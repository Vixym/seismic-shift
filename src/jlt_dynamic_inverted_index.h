#ifndef JLT_DYNAMIC_INVERTED_INDEX_H
#define JLT_DYNAMIC_INVERTED_INDEX_H

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <algorithm>
#include <optional>
#include <cstring>
#include <filesystem>

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/memory.hpp>

#include "sparse_dataset.h"
#include "configuration_strategies.h"
#include "jlt_dynamic_posting_list.h"

namespace seismic {

template <typename T>
class JltInvertedIndex : public SpaceUsage {
private:
    SparseDatasetMut<T> forward_index_;
    std::vector<JltPostingList> posting_lists_;
    Configuration config_;

    // doc_block_membership_[doc_id] = list of (component_id, block_id)
    std::vector<std::vector<std::pair<size_t, size_t>>> doc_block_membership_;

    // JLT transform matrix
    std::unique_ptr<SparseJLT> jlt_;

    std::optional<Knn> knn_;

public:
    JltInvertedIndex() = default;
    // Constructor
    JltInvertedIndex(SparseDatasetMut<T> forward_index, std::vector<JltPostingList> posting_lists, 
            Configuration config, std::vector<std::vector<std::pair<size_t, size_t>>> doc_block_membership,
            std::unique_ptr<SparseJLT> jlt, std::optional<Knn> knn = std::nullopt)
        : forward_index_(std::move(forward_index)), 
        posting_lists_(std::move(posting_lists)),
        config_(std::move(config)),
        doc_block_membership_(std::move(doc_block_membership)),
        jlt_(std::move(jlt)),
        knn_(std::move(knn)) {};
    
    static JltInvertedIndex<T> build(const SparseDatasetMut<T>& dataset, const Configuration& config)
    {
        // Distribute pairs (score, doc_id) to corresponding components for each chunk.
        // We use pairs because later each posting list will be sorted by score by the pruning strategy.
        // The pruning strategy is applied to partial results, for Global Threshold strategy
        // the final fixed pruning is done only when all chunks have been parsed
        std::cout << "Distributing and pruning postings: ";
        auto time_start = std::chrono::high_resolution_clock::now();
        
        auto jlt_ptr = std::make_unique<SparseJLT>(dataset.dim(), 1000);

        std::vector<std::vector<std::pair<T, size_t>>> inverted_pairs(jlt_ptr->row_matrix.rows);
        std::vector<std::vector<std::pair<T, size_t>>> chunk_inv_pairs(jlt_ptr->row_matrix.rows);
        
        size_t chunk_size = config.get_batched_indexing().value_or(dataset.len());

        for (size_t doc_id = 0; doc_id < dataset.len(); doc_id += chunk_size) {
            size_t end_id = std::min(doc_id + chunk_size, dataset.len());

            // Process each document in the chunk
            for (size_t i = doc_id; i < end_id; ++i) {
                auto [components, values] = dataset.get(i);
                auto transformed_vector = jlt_ptr->transform_vector(components, values);

                for (const auto& [component, value] : transformed_vector) {
                    chunk_inv_pairs[component].emplace_back(value, i);
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
        
        // Finished distributing and pruning dataset
        auto elapsed = std::chrono::high_resolution_clock::now() - time_start;
        std::cout << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() << " secs" << std::endl;
        std::cout << "\tNumber of posting lists: " << inverted_pairs.size() << std::endl;
        
        // Build summaries and blocks for each posting list
        std::cout << "Building summaries: " << std::endl;
        time_start = std::chrono::high_resolution_clock::now();
        std::vector<JltPostingList> posting_lists(inverted_pairs.size());
        
        // Initialize vector to map where each document exists in the inverted index (for efficient deletions)
        std::vector<std::vector<std::pair<size_t, size_t>>> doc_block_membership;
        doc_block_membership.resize(dataset.len());

        // TODO: Add parallelization back in #pragma omp parallel for
        for (size_t i = 0; i < inverted_pairs.size(); ++i) {
            JltPostingList posting_list = JltPostingList::build(dataset, inverted_pairs[i], config, *jlt_ptr);
            posting_lists[i] = posting_list;
            const auto& doc_block_pairs = posting_list.get_doc_block_pairs();
            for (const auto& [doc_id, block_id] : doc_block_pairs) {
                doc_block_membership[doc_id].emplace_back(i, block_id);
            }
        }
        
        elapsed = std::chrono::high_resolution_clock::now() - time_start;
        std::cout << "Finished building summaries: " << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() << " secs" << std::endl;
        
        JltInvertedIndex<T> index(dataset, std::move(posting_lists), config, std::move(doc_block_membership), std::move(jlt_ptr));
        
        // Handle KNN if needed
        if (config.get_knn_config().get_nknn() == 0 && !config.get_knn_config().get_knn_path().has_value()) {
            return index;
        }

        auto knn_config = config.get_knn_config();
        std::optional<Knn> knn;
        
        if (knn_config.get_knn_path().has_value()) {
            knn = Knn::new_from_serialized(knn_config.get_knn_path().value(), knn_config.get_nknn());
        } else {
            knn = Knn::new_from_index(index, knn_config.get_nknn());
        }
        
        elapsed = std::chrono::high_resolution_clock::now() - time_start;
        std::cout << "Finished building inverted index" << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() << " secs" << std::endl;
        
        return JltInvertedIndex<T>(index.forward_index_, std::move(index.posting_lists_), config, 
                                   std::move(index.doc_block_membership_), std::move(jlt_ptr), std::move(knn));
    }

    // Search the inverted index
    std::vector<std::pair<float, size_t>> search(
        const std::vector<uint16_t>& query_components,
        const std::vector<float>& query_values,
        size_t k,
        size_t query_cut,
        float heap_factor,
        size_t n_knn,
        bool first_sorted) const 
    {
        // TODO: Remove unused compiler logs
        (void) n_knn;

        // Transform the input vector
        auto transformed_vector = jlt_->transform_vector(query_components, query_values);
         
        // Sort query terms by score and evaluate the posting list only for the top ones
        utils::HeapFaiss heap(k);
        std::unordered_set<size_t> visited;
        visited.reserve(query_cut * 5000); // 5000 should be n_postings

        // Sort by decreasing value
        std::sort(transformed_vector.begin(), transformed_vector.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
         
        // Take only up to query_cut components
        size_t components_to_use = std::min(query_cut, transformed_vector.size());

        for (size_t i = 0; i < components_to_use; ++i) {
            uint16_t component_id = transformed_vector[i].first;
            posting_lists_[component_id].search(
                transformed_vector,
                k,
                heap_factor,
                heap,
                visited,
                forward_index_,
                i == 0 && first_sorted,
                *jlt_
            );
        }
         
        // TODO: Add back in refine
        // if (knn_ && n_knn > 0) {
        //     knn_->refine<float>(query, heap, visited, forward_index_, n_knn);
        // }
         
        auto results = heap.topk();
        std::vector<std::pair<float, size_t>> final_results;
        final_results.reserve(results.size());
         
        for (const auto& [dot, offset] : results) {
            final_results.emplace_back(std::abs(dot), forward_index_.offset_to_id(offset));
        }
         
        return final_results;
     }

    // Insert document to index
    void insert_document(const std::vector<uint16_t>& components, const std::vector<float>& values)
    {
        (void) values;
        // 0. Apply JLT transformation to document vector
        auto transformed_vector = jlt_->transform_vector(components, values);

        // 1. Add documemt to forward index
        forward_index_.push(transformed_vector);

        // 2. Iterate over each component and do the following:
        for (size_t component = 0; component < transformed_vector.size(); ++component) {
            if (component >= posting_lists_.size()) {
                // TODO: Modify size of posting list to fit new dimensions
            }
            auto posting_list = posting_lists_[component];
            // TODO 3. Find closest block

            // TODO 4. Add document to closest block

            // TODO 5. Update summary (if needed)

        }

        return;
    }

    // Remove document from index
    void delete_document(size_t doc_id)
    {
        // TODO 1. Change alive status in forward index

        // TODO 2. Update summary vectors that it contributes to

        // TODO 3. Check if need to resize
        (void) doc_id;
        return;
    }

    //////////////////////////
    //   Pruning strategies
    //////////////////////////
    static void fixed_pruning(std::vector<std::vector<std::pair<T, size_t>>>& inverted_pairs, size_t n_postings)
    {
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

    static void global_threshold_pruning(std::vector<std::vector<std::pair<T, size_t>>>& inverted_pairs, size_t n_postings)
    {
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

    size_t len() const {
        return forward_index_.len();
    }

    SparseDatasetMut<T> dataset() const {
        return forward_index_;
    }

    size_t space_usage_byte() const override {
        return 0;
    }
     // Returns the id of the largest component, i.e., the dimensionality of the vectors in the dataset.
     size_t dim() const {
         return forward_index_.dim();
     }
 
     // Returns the number of non-zero components in the dataset.
     size_t nnz() const {
         return forward_index_.nnz();
     }
    void print_space_usage_byte() {
        return;
    }

    // Archiving functionality
    template <class Archive>
    void serialize(Archive& archive) { archive(forward_index_, posting_lists_, config_, doc_block_membership_, jlt_, knn_); }
};
} // namespace seismic

#endif // JLT_DYNAMIC_INVERTED_INDEX_H