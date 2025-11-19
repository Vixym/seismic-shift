#ifndef INVERTED_INDEX_H
#define INVERTED_INDEX_H

#include <vector>
#include <memory>
#include <optional>
#include <algorithm>
#include <cassert>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <cmath>
#include <fstream>
#include <chrono>
#include <iostream>

#include "distances.h"
#include "data_type.h"
#include "space_usage.h"
#include "sparse_dataset.h"
#include "quantized_summary.h"
#include "top_k_selector.h"
#include "elias_fano.h"
#include "configuration_strategies.h"

namespace seismic {

// Forward declaration of template class
template <typename T>
class InvertedIndex;

 /**
 * Instead of string doc_ids we store their offsets in the forward_index and the lengths of the vectors.
 * This allows us to save the random accesses that would be needed to access exactly these values from the
 * forward index. The values of each doc are packed into a single u64 in `packed_postings`. We use 48 bits 
 * for the offset and 16 bits for the length. This choice limits the size of the dataset to be 1<<48-1.
 * We use the forward index to convert the offsets of the top-k back to the id of the corresponding documents.
 */
 class PostingList : public SpaceUsage {
    private:
        std::vector<uint64_t> packed_postings;
        std::vector<size_t> block_offsets;
        QuantizedSummary summaries;

    public:
        // Default constructor
        PostingList() = default;
        
        // Constructor with parameters
        PostingList(std::vector<uint64_t> packed_postings, std::vector<size_t> block_offsets, 
                   QuantizedSummary summaries)
            : packed_postings(std::move(packed_postings)),
              block_offsets(std::move(block_offsets)),
              summaries(std::move(summaries)) {}
        
        // Pack offset and length into a single uint64_t
        static uint64_t pack_offset_len(size_t offset, size_t len) {
            return ((static_cast<uint64_t>(offset) << 16) | (len & 0xFFFF));
        }
    
        // Unpack offset and length from a single uint64_t
        static std::pair<size_t, size_t> unpack_offset_len(uint64_t pack) {
            return {static_cast<size_t>(pack >> 16), static_cast<size_t>(pack & 0xFFFF)};
        }
        
        // Space usage calculation
        size_t space_usage_byte() const override {
            return packed_postings.size() * sizeof(uint64_t) +
                   block_offsets.size() * sizeof(size_t) +
                   summaries.space_usage_byte();
        }
        
        // Search implementation
        template <typename T>
        void search(
            const std::vector<float>& query,
            const std::vector<uint16_t>& query_components,
            const std::vector<float>& query_values,
            size_t k,
            float heap_factor,
            utils::HeapFaiss& heap,
            std::unordered_set<size_t>& visited,
            const SparseDataset<T>& forward_index,
            bool sort_summaries) const {
            
            std::vector<std::vector<uint64_t>> blocks_to_evaluate;
            
            // Get distances between query and summaries
            auto dots = summaries.distances_iter(query_components, query_values);
            
            // Create pairs of (block_id, dot_product)
            std::vector<std::pair<size_t, float>> indexed_dots;
            indexed_dots.reserve(dots.size());
            for (size_t i = 0; i < dots.size(); ++i) {
                indexed_dots.emplace_back(i, dots[i].value_or(0.0f));
            }
            
            // Sort summaries by dot product w.r.t. to the query. Useful only in the first list.
            if (sort_summaries) {
                std::sort(indexed_dots.begin(), indexed_dots.end(),
                        [](const auto& a, const auto& b) { return a.second > b.second; });
            }
            
            for (const auto& [block_id, dot] : indexed_dots) {
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
                            query_components,
                            query_values,
                            cur_packed_posting,
                            heap,
                            visited,
                            forward_index
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
                    query_components,
                    query_values,
                    cur_packed_posting,
                    heap,
                    visited,
                    forward_index
                );
            }
        }
        
        // Build a posting list from a dataset and a list of postings
        template <typename T>
        static PostingList build(
            const SparseDataset<T>& dataset,
            const std::vector<std::pair<T, size_t>>& postings,
            const Configuration& config) {
            
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
                    dataset,
                    blocking.get_clustering_algorithm()
                );
            }
            
            // Create summaries for each block
            SparseDatasetMut<T> summaries_dataset;
            
            for (size_t i = 0; i < block_offsets.size() - 1; ++i) {
                std::vector<size_t> block(
                    posting_list.begin() + block_offsets[i],
                    posting_list.begin() + block_offsets[i + 1]
                );
                
                // Apply summarization strategy
                const auto& summarization = config.get_summarization();
                std::pair<std::vector<uint16_t>, std::vector<T>> summary;
                
                if (summarization.get_type() == SummarizationStrategy::Type::FixedSize) {
                    summary = fixed_size_summary(
                        dataset,
                        block,
                        summarization.get_n_components()
                    );
                } else if (summarization.get_type() == SummarizationStrategy::Type::EnergyPreserving) {
                    summary = energy_preserving_summary(
                        dataset,
                        block,
                        summarization.get_summary_energy()
                    );
                }
                
                summaries_dataset.push(summary.first, summary.second);
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
            
            return PostingList(
                std::move(packed_postings),
                std::move(block_offsets),
                QuantizedSummary::from_sparse_dataset(summaries_dataset.to_immutable())
            );
        }

        // Evaluate a block of postings
        template <typename T>
        void evaluate_posting_block(
            const std::vector<float>& query,
            const std::vector<uint16_t>& query_term_ids,
            const std::vector<float>& query_values,
            const std::vector<uint64_t>& packed_posting_block,
            utils::HeapFaiss& heap,
            std::unordered_set<size_t>& visited,
            const SparseDataset<T>& forward_index) const {
            
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
                    
                    float distance;
                    if (query_term_ids.size() < THRESHOLD_BINARY_SEARCH) {
                        distance = distances::dot_product_with_merge(
                            query_term_ids, query_values, v_components, v_values);
                    } else {
                        distance = distances::dot_product_dense_sparse(query, v_components, v_values);
                    }
                    
                    visited.insert(prev_offset);
                    heap.push_with_id(-1.0f * distance, prev_offset);
                }
                
                prev_offset = offset;
                prev_len = len;
            }
            
            // Process the last posting if not already visited
            if (visited.find(prev_offset) == visited.end()) {
                auto [v_components, v_values] = forward_index.get_with_offset(prev_offset, prev_len);
                
                float distance = distances::dot_product_dense_sparse(query, v_components, v_values);
                
                visited.insert(prev_offset);
                heap.push_with_id(-1.0f * distance, prev_offset);
            }
        }
    
        // Blocking strategies
        static std::vector<size_t> fixed_size_blocking(const std::vector<size_t>& posting_list, size_t block_size) {
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
            const SparseDataset<T>& dataset,
            const ClusteringAlgorithm& clustering_algorithm) {
            
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
            (void)clustering_algorithm;
            
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
    
        // Summarization strategies
        template <typename T>
        static std::pair<std::vector<uint16_t>, std::vector<T>> fixed_size_summary(
            const SparseDataset<T>& dataset,
            const std::vector<size_t>& block,
            size_t n_components) {
            
            std::unordered_map<uint16_t, T> hash;
            
            // For each document in the block
            for (size_t doc_id : block) {
                // For each component in the document, store the largest value seen so far
                auto [components, values] = dataset.get(doc_id);
                
                for (size_t i = 0; i < components.size(); ++i) {
                    uint16_t component_id = components[i];
                    T value = values[i];
                    auto it = hash.find(component_id);
                    if (it == hash.end() || it->second < value) {
                        hash[component_id] = value;
                    }
                }
            }
            
            // Convert to vector of pairs for sorting
            std::vector<std::pair<uint16_t, T>> components_values(hash.begin(), hash.end());
            
            // Sort by decreasing values
            std::sort(components_values.begin(), components_values.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
            
            // Take only up to n_components
            if (components_values.size() > n_components) {
                components_values.resize(n_components);
            }
            
            // Sort by component ID for binary search
            std::sort(components_values.begin(), components_values.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
            
            // Extract components and values
            std::vector<uint16_t> components;
            std::vector<T> values;
            
            components.reserve(components_values.size());
            values.reserve(components_values.size());
            
            for (const auto& [component_id, value] : components_values) {
                components.push_back(component_id);
                values.push_back(value);
            }
            
            return {components, values};
        }
    
        template <typename T>
        static std::pair<std::vector<uint16_t>, std::vector<T>> energy_preserving_summary(
            const SparseDataset<T>& dataset,
            const std::vector<size_t>& block,
            float fraction) {
            
            std::unordered_map<uint16_t, T> hash;
            
            // For each document in the block
            for (size_t doc_id : block) {
                // For each component in the document, store the largest value seen so far
                auto [components, values] = dataset.get(doc_id);
                
                for (size_t i = 0; i < components.size(); ++i) {
                    uint16_t component_id = components[i];
                    T value = values[i];
                    auto it = hash.find(component_id);
                    if (it == hash.end() || it->second < value) {
                        hash[component_id] = value;
                    }
                }
            }
            
            // Convert to vector of pairs for sorting
            std::vector<std::pair<uint16_t, T>> components_values(hash.begin(), hash.end());
            
            // Sort by decreasing values
            std::sort(components_values.begin(), components_values.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
            
            // Calculate total energy
            float total_sum = 0.0f;
            for (const auto& [_, value] : components_values) {
                total_sum += static_cast<float>(value);
            }
            
            // Select components that preserve the desired energy fraction
            std::vector<uint16_t> term_ids;
            std::vector<T> values;
            
            float acc = 0.0f;
            for (const auto& [tid, value] : components_values) {
                acc += static_cast<float>(value);
                term_ids.push_back(tid);
                values.push_back(value);
                
                if ((acc / total_sum) > fraction) {
                    break;
            }
            }
            
            // Sort term IDs
            std::sort(term_ids.begin(), term_ids.end());
            
            // Reorder values to match sorted term IDs
            std::vector<T> sorted_values;
            sorted_values.reserve(term_ids.size());
            
            for (uint16_t tid : term_ids) {
                sorted_values.push_back(hash[tid]);
            }
            
            return {term_ids, sorted_values};
        }

        template <class Archive>
        void serialize(Archive& archive) {
            archive(packed_postings, block_offsets, summaries);
        }
};

/**
 * The main inverted index class.
 */
 template <typename T>
 class InvertedIndex : public SpaceUsage {
 private:
     SparseDataset<T> forward_index_;
     std::vector<PostingList> posting_lists_;
     Configuration config_;
     std::optional<Knn> knn_;
 
 public:
     // Default constructor
     InvertedIndex() = default;
     
     // Constructor with parameters
     InvertedIndex(SparseDataset<T> forward_index, std::vector<PostingList> posting_lists, 
                  Configuration config, std::optional<Knn> knn = std::nullopt)
         : forward_index_(std::move(forward_index)), 
         posting_lists_(std::move(posting_lists)),
         config_(std::move(config)),
         knn_(std::move(knn)) {}
         
     // Search the index
     std::vector<std::pair<float, size_t>> search(
         const std::vector<uint16_t>& query_components,
         const std::vector<float>& query_values,
         size_t k,
         size_t query_cut,
         float heap_factor,
         size_t n_knn,
         bool first_sorted) const 
     {
         // Create a dense query vector
         std::vector<float> query(dim(), 0.0f);
         for (size_t i = 0; i < query_components.size(); ++i) {
             query[query_components[i]] = query_values[i];
         }
         
         utils::HeapFaiss heap(k);
         std::unordered_set<size_t> visited;
         visited.reserve(query_cut * 5000); // 5000 should be n_postings
         
         // Sort query terms by score and evaluate the posting list only for the top ones
         std::vector<std::pair<uint16_t, float>> sorted_components;
         sorted_components.reserve(query_components.size());
         
         for (size_t i = 0; i < query_components.size(); ++i) {
             sorted_components.emplace_back(query_components[i], query_values[i]);
         }
         
         // Sort by decreasing value
         std::sort(sorted_components.begin(), sorted_components.end(),
                   [](const auto& a, const auto& b) { return a.second > b.second; });
         
         // Take only up to query_cut components
         size_t components_to_use = std::min(query_cut, sorted_components.size());
         
         for (size_t i = 0; i < components_to_use; ++i) {
             uint16_t component_id = sorted_components[i].first;
             posting_lists_[component_id].search(
                 query,
                 query_components,
                 query_values,
                 k,
                 heap_factor,
                 heap,
                 visited,
                 forward_index_,
                 i == 0 && first_sorted
             );
         }
         
         if (knn_ && n_knn > 0) {
             knn_->refine<float>(query, heap, visited, forward_index_, n_knn);
         }
         
         auto results = heap.topk();
         std::vector<std::pair<float, size_t>> final_results;
         final_results.reserve(results.size());
         
         for (const auto& [dot, offset] : results) {
             final_results.emplace_back(std::abs(dot), forward_index_.offset_to_id(offset));
         }
         
         return final_results;
     }
 
     /// `n_postings`: minimum number of postings to select for each component
     static InvertedIndex<T> build(const SparseDataset<T>& dataset, const Configuration& config) { 
         // Distribute pairs (score, doc_id) to corresponding components for each chunk.
         // We use pairs because later each posting list will be sorted by score
         // by the pruning strategy.
         // The pruning strategy is applied to partial results, for Global Threshold strategy
         // the final fixed pruning is done only when all chunks have been parsed
 
         std::cout << "Distributing and pruning postings: ";
         auto time_start = std::chrono::high_resolution_clock::now();
         
         std::vector<std::vector<std::pair<T, size_t>>> inverted_pairs(dataset.dim());
         std::vector<std::vector<std::pair<T, size_t>>> chunk_inv_pairs(dataset.dim());
         
         size_t chunk_size = config.get_batched_indexing().value_or(dataset.len());
         
         for (size_t doc_id = 0; doc_id < dataset.len(); doc_id += chunk_size) {
             size_t end_id = std::min(doc_id + chunk_size, dataset.len());
             
             // Process each document in the chunk
             for (size_t i = doc_id; i < end_id; ++i) {
                 auto [components, values] = dataset.get(i);
                 
                 for (size_t j = 0; j < components.size(); ++j) {
                     uint16_t c = components[j];
                     T score = values[j];
                     chunk_inv_pairs[c].emplace_back(score, i);
                 }
             }
             
             // If not batched indexing, chunk_inv_pairs already contain all the pairs
             if (chunk_size == dataset.len()) {
                 inverted_pairs = std::move(chunk_inv_pairs);
             } else {
                 // Copy the pairs of the current chunk in the partial results
                 for (size_t c = 0; c < chunk_inv_pairs.size(); ++c) {
                     for (const auto& [score, doc_id] : chunk_inv_pairs[c]) {
                         inverted_pairs[c].emplace_back(score, doc_id);
                     }
                 }
             }
             
             // Pruning on partial result
             const auto& pruning = config.get_pruning();
             if (pruning.get_type() == PruningStrategy::Type::FixedSize) {
                 fixed_pruning(inverted_pairs, pruning.get_n_postings());
             } else if (pruning.get_type() == PruningStrategy::Type::GlobalThreshold) {
                 global_threshold_pruning(inverted_pairs, pruning.get_n_postings());
             }
             
             // Reset chunk_inv_pairs for next chunk
             if (chunk_size != dataset.len()) {
                 chunk_inv_pairs.clear();
                 chunk_inv_pairs.resize(dataset.dim());
             }
         }
         
         // Final pruning
         const auto& pruning = config.get_pruning();
         if (pruning.get_type() == PruningStrategy::Type::GlobalThreshold) {
             size_t max_postings = static_cast<size_t>(pruning.get_n_postings() * pruning.get_max_fraction());
             fixed_pruning(inverted_pairs, max_postings);
         }
         
         auto elapsed = std::chrono::high_resolution_clock::now() - time_start;
         std::cout << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() << " secs" << std::endl;
         
         std::cout << "\tNumber of posting lists: " << inverted_pairs.size() << std::endl;
         
         std::cout << "Building summaries: ";
         time_start = std::chrono::high_resolution_clock::now();
         
         // Build summaries and blocks for each posting list
         std::vector<PostingList> posting_lists(inverted_pairs.size());
         
         #pragma omp parallel for
         for (size_t i = 0; i < inverted_pairs.size(); ++i) {
             posting_lists[i] = PostingList::build(dataset, inverted_pairs[i], config);
         }
         
         InvertedIndex<T> index(dataset, std::move(posting_lists), config);
         
         elapsed = std::chrono::high_resolution_clock::now() - time_start;
         std::cout << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() << " secs" << std::endl;
         
         // Handle KNN if needed
         if (config.get_knn_config().get_nknn() == 0 && !config.get_knn_config().get_knn_path().has_value()) {
             return index;
         }
         
         time_start = std::chrono::high_resolution_clock::now();
         auto knn_config = config.get_knn_config();
         std::optional<Knn> knn;
         
         if (knn_config.get_knn_path().has_value()) {
             knn = Knn::new_from_serialized(knn_config.get_knn_path().value(), knn_config.get_nknn());
         } else {
             knn = Knn::new_from_index(index, knn_config.get_nknn());
         }
         
         elapsed = std::chrono::high_resolution_clock::now() - time_start;
         std::cout << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() << " secs" << std::endl;
         
         return InvertedIndex<T>(index.forward_index_, std::move(index.posting_lists_), config, std::move(knn));
     }
     
     // Add a KNN graph to the index
     void add_knn(Knn new_knn) {
         knn_ = std::move(new_knn);
     }
     
     // Get the KNN graph
     const std::optional<Knn>& knn() const { return knn_; }
     
     // Get the KNN graph
     const std::optional<Knn>& knn_graph() const { return knn_; }

     // Fixed pruning strategy
     static void fixed_pruning(std::vector<std::vector<std::pair<T, size_t>>>& inverted_pairs, size_t n_postings) {
         #pragma omp parallel for
         for (size_t i = 0; i < inverted_pairs.size(); ++i) {
             auto& posting_list = inverted_pairs[i];
             
             // Sort by decreasing score
             std::sort(posting_list.begin(), posting_list.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
             
             // Truncate to n_postings
             if (posting_list.size() > n_postings) {
                 posting_list.resize(n_postings);
             }
             
             // Shrink to fit
             posting_list.shrink_to_fit();
         }
     }
     
     // Global threshold pruning strategy
     static void global_threshold_pruning(std::vector<std::vector<std::pair<T, size_t>>>& inverted_pairs, size_t n_postings) {
         const size_t tot_postings = inverted_pairs.size() * n_postings; // overall number of postings to select
     
         constexpr size_t EQUALITY_THRESHOLD = 10;
         const size_t max_eq_postings = EQUALITY_THRESHOLD * tot_postings / 100; // maximum number of posting with score equal to the threshold
     
         // For every posting we create the tuple <score, docid, id_posting_list>
         std::vector<std::tuple<T, size_t, uint16_t>> postings;
         
         for (size_t id = 0; id < inverted_pairs.size(); ++id) {
             auto& posting_list = inverted_pairs[id];
             for (const auto& [score, docid] : posting_list) {
                 postings.emplace_back(score, docid, static_cast<uint16_t>(id));
             }
             posting_list.clear();
         }
         
         const size_t actual_tot_postings = std::min(tot_postings, postings.size() - 1);
         
         // Sort postings by decreasing score
         std::sort(postings.begin(), postings.end(),
                 [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
         
         // Find the threshold score
         const T threshold_score = std::get<0>(postings[actual_tot_postings]);
         
         // Count postings with score equal to the threshold
         size_t eq_count = 0;
         for (size_t i = actual_tot_postings; i < postings.size(); ++i) {
             if (std::get<0>(postings[i]) == threshold_score) {
                 eq_count++;
             } else {
                 break;
             }
         }
         
         if (eq_count > max_eq_postings) {
             std::cout << "A lot of entries have the same value. " << (eq_count - max_eq_postings) 
                     << " have been pruned, for more info look at DOC_REFERENCE" << std::endl;
         }
         
         // Add all postings above threshold and up to max_eq_postings with equal threshold
         for (size_t i = 0; i < actual_tot_postings; ++i) {
             const auto& [score, docid, id_posting] = postings[i];
             inverted_pairs[id_posting].emplace_back(score, docid);
         }
         
         // Add postings with scores equal to the threshold, up to max_eq_postings
         size_t eq_added = 0;
         for (size_t i = actual_tot_postings; i < postings.size() && eq_added < max_eq_postings; ++i) {
             if (std::get<0>(postings[i]) == threshold_score) {
                 const auto& [score, docid, id_posting] = postings[i];
                 inverted_pairs[id_posting].emplace_back(score, docid);
                 eq_added++;
             } else {
                 break;
             }
         }
     }
     
     // Getters
     const SparseDataset<T>& dataset() const { return forward_index_; }
 
     // TODO return to this after SparseDatasetIterator is implemented
     // const SparseDatasetIterator<T>& iterator() const { return forward_index_.iterator(); }
 
     // Returns (offset, len) of the "id"-th document
     std::pair<size_t, size_t> id_to_offset_len(size_t id) const {
         return forward_index_.id_to_offset_len(id);
     }
 
     // Returns the id of the largest component, i.e., the dimensionality of the vectors in the dataset.
     size_t dim() const {
         return forward_index_.dim();
     }
 
     // Returns the number of non-zero components in the dataset.
     size_t nnz() const {
         return forward_index_.nnz();
     }
 
     // Returns the number of vectors in the dataset
     size_t len() const {
         return forward_index_.len();
     }
 
     // Checks if the dataset is empty.
     bool is_empty() const {
         return forward_index_.len() == 0;
     }
 
     // Returns the number of neighbors in the knn graph, 0 if knn graph is not present.
     size_t knn_len() const {
         return knn_.has_value() ? knn_->get_d() : 0;
     }
 
     // Space usage
     size_t space_usage_byte() const override;
     
     // Space usage
     size_t print_space_usage_byte() const;

    template <class Archive>
    void serialize(Archive& archive) {
        archive(forward_index_, posting_lists_, config_, knn_);
    }

 };

} // namespace seismic


#endif // INVERTED_INDEX_H
