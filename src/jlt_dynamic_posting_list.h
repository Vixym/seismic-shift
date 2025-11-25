#ifndef JLT_DYNAMIC_POSTING_LIST_H
#define JLT_DYNAMIC_POSTING_LIST_H

#include <vector>
#include <cstdint>
#include <utility>
#include <unordered_set>
#include <limits>

#include <cereal/types/vector.hpp>
#include <cereal/types/utility.hpp>   // <-- for std::pair
#include <cereal/types/memory.hpp>    // <-- for std::unique_ptr<SparseJLT>
#include <cereal/types/optional.hpp>  // if you use std::optional<Knn>

#include "sparse_dataset.h"
#include "sparse_jlt.cpp"
#include "configuration_strategies.h"

namespace seismic
{
/**
* Instead of string doc_ids we store their offsets in the forward_index and the lengths of the vectors.
* This allows us to save the random accesses that would be needed to access exactly these values from the
* forward index. The values of each doc are packed into a single u64 in `packed_postings`. We use 48 bits
* for the offset and 16 bits for the length. This choice limits the size of the dataset to be 1<<48-1.
* We use the forward index to convert the offsets of the top-k back to the id of the corresponding documents.
*/
struct JltPostingList : public SpaceUsage 
{
private:
    std::vector<uint64_t> packed_postings;
    std::vector<size_t> block_offsets;
    std::vector<std::vector<std::pair<size_t, float>>> summaries; // Summaries are stored as sparse centroid vectors
    std::vector<std::pair<size_t, size_t>> doc_block_pairs_;

public:
    JltPostingList() = default;
    JltPostingList(std::vector<uint64_t> packed_postings, std::vector<size_t> block_offsets,
                std::vector<std::vector<std::pair<size_t, float>>> summaries, std::vector<std::pair<size_t, size_t>> doc_block_pairs)
        : packed_postings(std::move(packed_postings)),
          block_offsets(std::move(block_offsets)),
          summaries(std::move(summaries)),
          doc_block_pairs_(std::move(doc_block_pairs)) {}

    std::vector<std::pair<size_t, size_t>> get_doc_block_pairs() { return doc_block_pairs_; };

    static void map_docs_to_blocks(std::vector<size_t>& postings, std::vector<size_t> offsets, std::vector<std::pair<size_t, size_t>>& doc_block_pairs)
    {   
        if (offsets.size() == 0 || postings.size() == 0) return;
        const size_t n_blocks = offsets.size() - 1;
        for (size_t block_id = 0; block_id < n_blocks; ++block_id) {
            size_t start = offsets[block_id];
            size_t end   = offsets[block_id + 1];

            for (size_t idx = start; idx < end; ++idx) {
                auto doc_id = postings[idx];
                doc_block_pairs.emplace_back(doc_id, block_id);
            }
        }
    }

    template <typename T>
    T dot_product_sparse_vectors(const std::vector<std::pair<size_t, T>>& v1, 
        const std::vector<std::pair<size_t, T>>& v2) const
    {
        size_t i = 0, j = 0;
        T result = T{0};

        while (i < v1.size() && j < v2.size()) {
            size_t idx1 = v1[i].first;
            size_t idx2 = v2[j].first;

            if (idx1 == idx2) {
                result += v1[i].second * v2[j].second;
                ++i;
                ++j;
            } else if (idx1 < idx2) {
                ++i;
            } else {
                ++j;
            }
        }

        return result;
    }

    template <typename T>
    std::vector<std::pair<size_t, T>> compute_summary_distances(const std::vector<std::pair<size_t, T>>& query) const
    {
        std::vector<std::pair<size_t, T>> results;
        results.reserve(summaries.size());

        for (size_t block = 0; block < summaries.size(); ++block) {
            T dot = dot_product_sparse_vectors(query, summaries[block]);
            results.emplace_back(block, dot);
        }

        return results;
    }

    // Search implementation
    template <typename T>
    void search(
        const std::vector<std::pair<size_t, T>>& query,
        size_t k,
        float heap_factor,
        utils::HeapFaiss& heap,
        std::unordered_set<size_t>& visited,
        const SparseDatasetMut<T>& forward_index,
        bool sort_summaries,
        SparseJLT& jlt) const 
    {    
        std::vector<std::vector<uint64_t>> blocks_to_evaluate;
        
        // Get distances between query and summaries
        auto distances = compute_summary_distances(query);
        
        // Sort summaries by dot product w.r.t. to the query. Useful only in the first list.
        if (sort_summaries) {
            std::sort(distances.begin(), distances.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });
        }
        
        for (const auto& [block_id, dot] : distances) {
            // Skip blocks that cannot contribute to the top-k
            if (heap.len() == k && dot < -heap_factor * heap.topk().front().first) {
                continue;
            }
            
            // Get the block of postings
            std::vector<uint64_t> packed_posting_block(
                packed_postings.begin() + block_offsets[block_id],
                packed_postings.begin() + block_offsets[block_id + 1]
            );
            
            // If we have accumulated blocks, evaluate them first
            if (blocks_to_evaluate.size() == 1) {
                for (const auto& cur_packed_posting : blocks_to_evaluate) {
                    evaluate_posting_block(
                        query,
                        cur_packed_posting,
                        heap,
                        visited,
                        forward_index,
                        jlt
                    );
                }
                blocks_to_evaluate.clear();
            }
            
            // Prefetch the block
            for (size_t i = 0; i < packed_posting_block.size(); i += 8) {
                // Prefetch the next elements (if available)
                if (i + 8 < packed_posting_block.size()) {
                    __builtin_prefetch(&packed_posting_block[i + 8], 0, 1);
                }
            }
            
            blocks_to_evaluate.push_back(std::move(packed_posting_block));
        }
        
        // Evaluate any remaining blocks
        for (const auto& cur_packed_posting : blocks_to_evaluate) {
            evaluate_posting_block(
                query,
                cur_packed_posting,
                heap,
                visited,
                forward_index,
                jlt
            );
        }
    }

    // Evaluate a block of postings
    template <typename T>
    void evaluate_posting_block(
        const std::vector<std::pair<size_t, T>>& query,
        const std::vector<uint64_t>& packed_posting_block,
        utils::HeapFaiss& heap,
        std::unordered_set<size_t>& visited,
        const SparseDatasetMut<T>& forward_index,
        SparseJLT& jlt) const 
    {    
        if (packed_posting_block.empty()) {
            return;
        }
        
        // Process the first posting
        auto [prev_offset, prev_len] = unpack_offset_len(packed_posting_block[0]);
        
        // Process the middle postings
        for (size_t i = 1; i < packed_posting_block.size(); ++i) {
            auto [offset, len] = unpack_offset_len(packed_posting_block[i]);
            
            // TODO: revisit, prefetching disabled for now
            
            // Process the current vector if not already visited
            if (visited.find(prev_offset) == visited.end()) {
                auto [v_components, v_values] = forward_index.get_with_offset(prev_offset, prev_len);
                auto transformed_vector = jlt.transform_vector(v_components, v_values);

                float distance = dot_product_sparse_vectors(query, transformed_vector);
                // if (query_term_ids.size() < THRESHOLD_BINARY_SEARCH) {
                //     distance = distances::dot_product_with_merge(
                //         query_term_ids, query_values, v_components, v_values);
                // } else {
                //     distance = distances::dot_product_dense_sparse(query, v_components, v_values);
                // }
                
                visited.insert(prev_offset);
                heap.push_with_id(-1.0f * distance, prev_offset);
            }
            
            prev_offset = offset;
            prev_len = len;
        }
        
        // Process the last posting if not already visited
        if (visited.find(prev_offset) == visited.end()) {
            auto [v_components, v_values] = forward_index.get_with_offset(prev_offset, prev_len);
            auto transformed_vector = jlt.transform_vector(v_components, v_values);
            
            float distance = dot_product_sparse_vectors(query, transformed_vector);
            
            visited.insert(prev_offset);
            std::cout << "Adding to heap" << std::endl;
            heap.push_with_id(-1.0f * distance, prev_offset);
        }
    }
    
    // Build a posting list from a dataset and a list of postings
    template <typename T>
    static JltPostingList build(
        const SparseDatasetMut<T>& dataset,
        const std::vector<std::pair<T, size_t>>& postings,
        const Configuration& config,
        const SparseJLT& jlt) 
    {    
        if (postings.size() == 0) {
            return JltPostingList{};
        }
        std::cout << "LOG: Building posting list with " << postings.size() << " postings" << std::endl;
        // Extract document IDs from postings
        std::vector<size_t> posting_list;
        posting_list.reserve(postings.size());
        for (const auto& [_, doc_id] : postings) {
            posting_list.push_back(doc_id);
        }

        // Apply blocking strategy
        std::vector<size_t> block_offsets;
        const auto& blocking = config.get_blocking();
        
        if (blocking.get_type() == BlockingStrategy::Type::FixedSize) {
            block_offsets = fixed_size_blocking(posting_list, blocking.get_block_size());
        } else if (blocking.get_type() == BlockingStrategy::Type::RandomKmeans) {
            block_offsets = blocking_with_random_kmeans(
                posting_list,
                blocking.get_centroid_fraction(),
                blocking.get_min_cluster_size(),
                dataset
            );
        }

        // Map documents to their block indices
        std::vector<std::pair<size_t, size_t>> doc_block_pairs;
        map_docs_to_blocks(posting_list, block_offsets, doc_block_pairs);
        
        // Create summaries(centroids) for each block
        std::vector<std::vector<std::pair<size_t, float>>> block_centroids;
        block_centroids.reserve(block_offsets.size() - 1);

        for (size_t i = 0; i < block_offsets.size() - 1; ++i) {
            std::vector<size_t> block(
                    posting_list.begin() + block_offsets[i],
                    posting_list.begin() + block_offsets[i + 1]);

            // Compute centroid for the block
            std::vector<float> centroid(jlt.row_matrix.rows, 0.0f);
            for (size_t doc_id : block) {
                auto doc = dataset.get(doc_id);
                auto& components = doc.first;
                auto& values     = doc.second;
                auto transformed_vector = jlt.transform_vector(components, values);
                for (const auto& [component, value] : transformed_vector) {
                    centroid[component] += value;
                }
            }

            // Normalize centroid
            float norm = 0.0f;
            for (float val : centroid) {
                norm += val * val;
            }
            norm = std::sqrt(norm);
            if (norm > 0) {
                for (float &val : centroid) {
                    val /= norm;
                }
            }

            std::vector<std::pair<size_t, float>> sparse_centroid;
            for (size_t component = 0; component < centroid.size(); ++component) {
                if (centroid[component] != 0.0f) {
                    sparse_centroid.emplace_back(component, centroid[component]);
                }
            }
            block_centroids.push_back(sparse_centroid);
        }

        // Pack document offsets and lengths
        std::vector<uint64_t> packed_postings;
        packed_postings.reserve(posting_list.size());
        
        for (size_t doc_id : posting_list) {
            packed_postings.push_back(
                pack_offset_len(
                    dataset.vector_offset(doc_id),
                    dataset.vector_len(doc_id)
                )
            );
        }

        return JltPostingList(
            std::move(packed_postings),
            std::move(block_offsets),
            std::move(block_centroids),
            std::move(doc_block_pairs)
        );
    }

    // Pack offset and length into a single uint64_t
    static uint64_t pack_offset_len(size_t offset, size_t len) { return ((static_cast<uint64_t>(offset) << 16) | (len & 0xFFFF)); }

    // Unpack offset and length from a single uint64_t
    static std::pair<size_t, size_t> unpack_offset_len(uint64_t pack) { return {static_cast<size_t>(pack >> 16), static_cast<size_t>(pack & 0xFFFF)}; }

    // Space usage calculation
    size_t space_usage_byte() const override
    {
        size_t summary_size = 0;
        for (const auto &summary : summaries) {
            summary_size += summary.size() * (sizeof(size_t) + sizeof(float));
        }
        return packed_postings.size() * sizeof(uint64_t) +
               block_offsets.size() * sizeof(size_t) +
               summary_size;
    }

    //////////////////////////
    //   Blocking strategies
    //////////////////////////
    static std::vector<size_t> fixed_size_blocking(const std::vector<size_t>& posting_list, size_t block_size) 
    {
        // Create block offsets for fixed-size blocks
        std::vector<size_t> block_offsets;
        
        for (size_t i = 0; i < posting_list.size(); i += block_size) {
            block_offsets.push_back(i);
        }
        
        // Add the final offset
        block_offsets.push_back(posting_list.size());
        
        return block_offsets;
    }

    template <typename T>
    static std::vector<size_t> blocking_with_random_kmeans(
        std::vector<size_t>& posting_list,
        float centroid_fraction,
        size_t min_cluster_size,
        const SparseDatasetMut<T>& dataset) 
    {
        if (posting_list.empty()) {
            return {};
        }
        
        // Calculate number of centroids
        size_t n_centroids = std::max(
            static_cast<size_t>(centroid_fraction * posting_list.size()),
            static_cast<size_t>(1)
        );
        
        // Ensure n_centroids doesn't exceed uint16_t max
        assert(n_centroids <= std::numeric_limits<uint16_t>::max() && 
                "In the current implementation the number of centroids cannot be greater than uint16_t::max. "
                "This is due to the quantized summary using uint16_t to store the centroids ids (aka summaries ids). "
                "Please, decrease centroid_fraction!");
        
        std::vector<size_t> reordered_posting_list;
        reordered_posting_list.reserve(posting_list.size());
        
        std::vector<size_t> block_offsets;
        block_offsets.reserve(n_centroids + 1);
        
        // TEMPORARY SOLUTION: Use simple round-robin clustering
        // This avoids the type compatibility issues between seismic::SparseDataset and utils::SparseDataset
        std::vector<std::pair<size_t, size_t>> clustering_results;
        clustering_results.reserve(posting_list.size());
        
        for (size_t i = 0; i < posting_list.size(); ++i) {
            size_t centroid_id = i % n_centroids;
            clustering_results.emplace_back(centroid_id, posting_list[i]);
        }
        
        // Mark unused parameters to avoid compiler warnings
        (void)min_cluster_size;
        (void)dataset;
        
        // Start with offset 0
        block_offsets.push_back(0);
        
        // Group by centroid ID
        std::sort(clustering_results.begin(), clustering_results.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
        
        size_t current_centroid = clustering_results[0].first;
        for (const auto& [centroid_id, doc_id] : clustering_results) {
            if (centroid_id != current_centroid) {
                // New centroid group
                block_offsets.push_back(reordered_posting_list.size());
                current_centroid = centroid_id;
            }
            reordered_posting_list.push_back(doc_id);
        }
        
        // Add final offset
        block_offsets.push_back(reordered_posting_list.size());
        
        // Copy reordered list back to original
        assert(reordered_posting_list.size() == posting_list.size());
        posting_list = std::move(reordered_posting_list);
        
        return block_offsets;
    }

    template <class Archive>
    void serialize(Archive& archive) { archive(packed_postings, block_offsets, summaries, doc_block_pairs_); }
};

} // namespace seismic

#endif // JLT_DYNAMIC_POSTING_LIST_H