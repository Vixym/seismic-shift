#ifndef MY_POSTING_LIST_H
#define MY_POSTING_LIST_H

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
#include "summary_strategies.h"

namespace seismic
{

/**
 * Instead of string doc_ids we store their offsets in the forward_index and the lengths of the vectors.
 * This allows us to save the random accesses that would be needed to access exactly these values from the
 * forward index. The values of each doc are packed into a single u64 in `packed_postings`. We use 48 bits 
 * for the offset and 16 bits for the length. This choice limits the size of the dataset to be 1<<48-1.
 * We use the forward index to convert the offsets of the top-k back to the id of the corresponding documents.
 */
 class MyPostingList : public SpaceUsage {
    using SparseVector = std::vector<std::pair<uint16_t, float>>;
    private:
        std::vector<uint64_t> packed_postings_;
        std::vector<size_t> block_offsets_;
        std::vector<std::pair<size_t, SparseVector>> summaries_; // Summaries are stored as (size, summary)
        std::vector<std::pair<size_t, size_t>> doc_block_pairs_;
    
    public:
        // Default constructor
        MyPostingList() = default;
        
        // Constructor with parameters
        MyPostingList(std::vector<uint64_t> packed_postings, std::vector<size_t> block_offsets, 
                   std::vector<std::pair<size_t, SparseVector>> summaries, std::vector<std::pair<size_t, size_t>> doc_block_pairs)
            : packed_postings_(std::move(packed_postings)),
              block_offsets_(std::move(block_offsets)),
              summaries_(std::move(summaries)),
              doc_block_pairs_(std::move(doc_block_pairs)) {}
        
        // Build a posting list from a dataset and a list of postings
        template <typename T>
        static MyPostingList build(
            const SparseDatasetMut<T>& dataset,
            const std::vector<std::pair<T, size_t>>& postings,
            const Configuration& config,
            const SparseJLT& jlt) 
        {    
            (void)jlt;
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
    
            // Map documents to their block indices
            std::vector<std::pair<size_t, size_t>> doc_block_pairs;
            map_docs_to_blocks(posting_list, block_offsets, doc_block_pairs);

            // Create summaries for each block
            std::vector<std::pair<size_t, SparseVector>> summaries;
            summaries.reserve(block_offsets.size());

            for (size_t i = 0; i < block_offsets.size() - 1; ++i) {
                std::vector<size_t> block(
                    posting_list.begin() + block_offsets[i],
                    posting_list.begin() + block_offsets[i + 1]
                );

                std::vector<std::pair<uint16_t, float>> summary;
                // std::pair<std::vector<uint16_t>, std::vector<T>> summary;

                // // Apply summarization strategy
                // const auto& summarization = config.get_summarization();
                
                // if (summarization.get_type() == SummarizationStrategy::Type::FixedSize) {
                //     summary = fixed_size_centroid(
                //         dataset,
                //         block,
                //         summarization.get_n_components()
                //     );
                // } else if (summarization.get_type() == SummarizationStrategy::Type::EnergyPreserving) {
                //     summary = energy_preserving_centroid(
                //         dataset,
                //         block,
                //         summarization.get_summary_energy()
                //     );
                // }

                const auto& summarization = config.get_summarization();
                const auto& summarization_metric = config.get_summarization_metric();
                if (summarization_metric == "max") {
                    summary = MaxSummary::summary_init(dataset, block, summarization.get_n_components());
                } else if (summarization_metric == "centroid") {
                    summary = CentroidSummary::summary_init(dataset, block, summarization.get_n_components());
                }

                const auto& transform_function = config.get_transform_function();
                if (transform_function == "jlt") {
                    summary = jlt.transform(summary);
                }

                summaries.emplace_back(block.size(), summary);
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
            
            return MyPostingList(
                std::move(packed_postings),
                std::move(block_offsets),
                std::move(summaries),
                std::move(doc_block_pairs)
            );
        }

        template <typename T>
        void delete_doc(const size_t block_id, const size_t doc_id, const SparseJLT& jlt,
            const SparseDatasetMut<T>& dataset, const Configuration& config)
        {
            (void)jlt;
            (void)doc_id;
            (void)config;

            std::vector<uint64_t> block(
                packed_postings_.begin() + block_offsets_[block_id],
                packed_postings_.begin() + block_offsets_[block_id + 1]
            );

            // 1. Retrieve doc from dataset and convert to SparseVector format
            std::pair<std::vector<uint16_t>, std::vector<T>> doc_vec_pair = dataset.get(doc_id);
            std::vector<std::pair<uint16_t, float>> doc_vec;
            for (size_t i = 0; i < doc_vec_pair.first.size(); ++i) {
                doc_vec.emplace_back(doc_vec_pair.first[i], doc_vec_pair.second[i]);
            }

            std::vector<std::pair<uint16_t, float>>& summary = summaries_[block_id].second;
            size_t num_docs_in_block = summaries_[block_id].first;
    
            // 2. Transform if needed
            const auto& transform_function = config.get_transform_function();
            if (transform_function == "jlt") {
                doc_vec = jlt.transform(doc_vec);
            }

            // 3. Update the summaries
            const auto& summary_metric = config.get_summarization_metric();
            std::vector<std::pair<uint16_t, float>> new_summary;
            if (summary_metric == "max") {
                new_summary = MaxSummary::summary_delete(dataset, summary, doc_vec, block, config.get_summarization().get_n_components());
            } else if (summary_metric == "centroid") {
                new_summary = CentroidSummary::summary_delete(summary, doc_vec, num_docs_in_block);
            }

            summaries_[block_id] = {num_docs_in_block-1, new_summary};
        }

        size_t insert_doc(const std::vector<uint16_t>& components, const std::vector<float>& values,
             const SparseJLT& jlt, const Configuration& config, size_t offset, size_t length)
        {   
            (void)jlt;
            (void)config;
            // 0. Transforming representation and vector transformation if needed
            std::vector<std::pair<uint16_t, float>> doc_vec;
            for (size_t i = 0; i < components.size(); ++i) {
                doc_vec.emplace_back(components[i], values[i]);
            }
            const auto& transform_function = config.get_transform_function();
            if (transform_function == "jlt") {
                doc_vec = jlt.transform(doc_vec);
            }

            // 1. Find closest block
            std::vector<std::pair<uint16_t, float>> indexed_dots = get_distances(components, values, jlt, config);
            size_t best_block = 0;
            float best_val = 0.0f;
            for (const auto& [block_id, dot] : indexed_dots) {
                if (dot > best_val) {
                    best_block = block_id;
                    best_val = dot;
                }
            }

            // 2. Add document to block
            uint64_t packed_posting = pack_offset_len(offset, length);
            packed_postings_.insert(packed_postings_.begin()+block_offsets_[best_block+1], packed_posting);
            for (size_t block_id = best_block+1; block_id < block_offsets_.size(); ++block_id) {
                block_offsets_[block_id] += 1;
            }

            // 3. Update summary based on summary metric
            const auto& summary_metric = config.get_summarization_metric();
            std::vector<std::pair<uint16_t, float>> new_summary;

            if (summary_metric == "max") {
                new_summary = MaxSummary::summary_insert(summaries_[best_block].second, doc_vec);
            } else if (summary_metric == "centroid") {
                new_summary = CentroidSummary::summary_insert(summaries_[best_block].second, doc_vec, summaries_[best_block].first);
            } else {
                std::cout << "my_posting_list.h:insert_doc invalid summary metric" << std::endl;
            }

            summaries_[best_block] = {summaries_[best_block].first+1, new_summary};

            // 4. Return block id of closest block
            return best_block;
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
            const SparseDatasetMut<T>& forward_index,
            bool sort_summaries,
            const SparseJLT& jlt,
            const Configuration& config) const 
        {
            std::vector<std::vector<uint64_t>> blocks_to_evaluate;

            // Get distances between query and summaries vector of (idx, val) pairs
            std::vector<std::pair<uint16_t, float>> indexed_dots = get_distances(query_components, query_values, jlt, config);
            
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
                    packed_postings_.begin() + block_offsets_[block_id],
                    packed_postings_.begin() + block_offsets_[block_id + 1]
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

        std::vector<std::pair<uint16_t, float>> get_distances(const std::vector<uint16_t>& query_components, const std::vector<float>& query_values,
            const SparseJLT& jlt, const Configuration& config) const
        {
            (void) jlt;
            (void) config;
            const std::vector<uint16_t>* components = &query_components;
            const std::vector<float>* values = &query_values;

            std::vector<std::pair<uint16_t, float>> distances;
            distances.resize(summaries_.size());

            // Transform vectors into new space if needed
            std::vector<uint16_t> transformed_components;
            std::vector<float> transformed_values;

            if (config.get_transform_function() == "jlt") {
                auto transformed_vec = jlt.transform(query_components, query_values);

                transformed_components.reserve(transformed_vec.size());
                transformed_values.reserve(transformed_vec.size());
                for (const auto& [c, v] : transformed_vec) {
                    transformed_components.push_back(c);
                    transformed_values.push_back(v);
                }

                components = &transformed_components;
                values = &transformed_values;
            }
            
            for (size_t i = 0; i < summaries_.size(); ++i) {
                std::vector<std::pair<uint16_t, float>> summary = summaries_[i].second;
                // Two-pointer merge to compute dot product between sparse vectors
                size_t qi = 0;
                size_t si = 0;
                float acc = 0.0f;

                while (qi < components->size() && si < summary.size()) {
                    uint16_t qc = (*components)[qi];
                    uint16_t sc = summary[si].first;

                    if (qc == sc) {
                        // Matching component: accumulate product
                        acc += static_cast<float>((*values)[qi]) * static_cast<float>(summary[si].second);
                        ++qi;
                        ++si;
                    } else if (qc < sc) {
                        ++qi;
                    } else {
                        ++si;
                    }
                }

                // Store (summary_id, distance/similarity)
                distances[i] = {static_cast<uint16_t>(i), acc};
            }
            return distances;
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
            const SparseDatasetMut<T>& forward_index) const 
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
            const SparseDatasetMut<T>& dataset,
            const ClusteringAlgorithm& clustering_algorithm)
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
            // This avoids the type compatibility issues between seismic::SparseDatasetMut and utils::SparseDatasetMut
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
            const SparseDatasetMut<T>& dataset,
            const std::vector<size_t>& block,
            size_t n_components) 
        {    
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
        static std::pair<std::vector<uint16_t>, std::vector<T>> fixed_size_centroid(
            const SparseDatasetMut<T>& dataset,
            const std::vector<size_t>& block,
            size_t n_components) 
        {    
            std::unordered_map<uint16_t, T> hash;
            
            // For each document in the block
            for (size_t doc_id : block) {
                // For each component in the document, store the largest value seen so far
                auto [components, values] = dataset.get(doc_id);
                
                for (size_t i = 0; i < components.size(); ++i) {
                    uint16_t component_id = components[i];
                    T value = values[i];
                    auto it = hash.find(component_id);
                    if (it == hash.end()) {
                        hash[component_id] = value;
                    } else {
                        hash[component_id] += value;
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
                values.push_back(value/block.size());
            }
            
            return {components, values};
        }

        template <typename T>
        static std::pair<std::vector<uint16_t>, std::vector<T>> energy_preserving_centroid(
            const SparseDatasetMut<T>& dataset,
            const std::vector<size_t>& block,
            float fraction) 
        {
            
            std::unordered_map<uint16_t, T> hash;
            
            // For each document in the block
            for (size_t doc_id : block) {
                // For each component in the document, store the largest value seen so far
                auto [components, values] = dataset.get(doc_id);
                
                for (size_t i = 0; i < components.size(); ++i) {
                    uint16_t component_id = components[i];
                    T value = values[i];
                    auto it = hash.find(component_id);
                    if (it == hash.end()) {
                        hash[component_id] = value;
                    } else {
                        hash[component_id] += value;
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
                values.push_back(value/block.size());
                
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
    
        template <typename T>
        static std::pair<std::vector<uint16_t>, std::vector<T>> energy_preserving_summary(
            const SparseDatasetMut<T>& dataset,
            const std::vector<size_t>& block,
            float fraction) 
        {
            
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

        static void map_docs_to_blocks(std::vector<size_t>& postings, std::vector<size_t>& offsets, std::vector<std::pair<size_t, size_t>>& doc_block_pairs)
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

        std::vector<std::pair<size_t, size_t>> get_doc_block_pairs()
        {
            return doc_block_pairs_;
        }

        template <class Archive>
        void serialize(Archive& archive) {
            archive(packed_postings_, block_offsets_, summaries_);
        }

        // Pack offset and length into a single uint64_t
        static uint64_t pack_offset_len(size_t offset, size_t len) { return ((static_cast<uint64_t>(offset) << 16) | (len & 0xFFFF)); }
    
        // Unpack offset and length from a single uint64_t
        static std::pair<size_t, size_t> unpack_offset_len(uint64_t pack) { return {static_cast<size_t>(pack >> 16), static_cast<size_t>(pack & 0xFFFF)}; }
        
        // Space usage calculation
        size_t space_usage_byte() const override {
            return packed_postings_.size() * sizeof(uint64_t) +
                   block_offsets_.size() * sizeof(size_t);
        }
};

} // namespace seismic

#endif // MY_POSTING_LIST_H