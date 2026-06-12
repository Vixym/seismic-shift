#ifndef MY_INVERTED_INDEX_H
#define MY_INVERTED_INDEX_H

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
#include "my_posting_list.h"

namespace seismic {

/**
 * The main inverted index class.
 */
template <typename T>
class MyInvertedIndex : public SpaceUsage {
private:
    SparseDatasetMut<T> forward_index_;
    std::vector<MyPostingList> posting_lists_;
    Configuration config_;

    // doc_block_membership_[doc_id] = list of (component_id, block_id)
    std::vector<std::vector<std::pair<size_t, size_t>>> doc_block_membership_;

    // JLT transform matrix
    std::unique_ptr<SparseJLT> jlt_;

    std::optional<Knn> knn_;

public:
    // Default constructor
    MyInvertedIndex() = default;
     
    // Constructor with parameters
    MyInvertedIndex(SparseDatasetMut<T> forward_index, std::vector<MyPostingList> posting_lists, 
                  Configuration config, std::vector<std::vector<std::pair<size_t, size_t>>> doc_block_membership, 
                  std::unique_ptr<SparseJLT> jlt, std::optional<Knn> knn = std::nullopt)
        : forward_index_(std::move(forward_index)), 
        posting_lists_(std::move(posting_lists)),
        config_(std::move(config)),
        doc_block_membership_(std::move(doc_block_membership)),
        jlt_(std::move(jlt)),
        knn_(std::move(knn)) {}
         
    void delete_doc(size_t doc_id)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        // 1. Remove doc from forward index (mark as dead)
        forward_index_.set_dead(doc_id);

        // 2. Remove doc from all the posting lists it belongs to (using doc_block_membership_)
        std::vector<std::pair<size_t, size_t>> doc_block_ids = doc_block_membership_[doc_id];
        for (const auto& [component, block_id] : doc_block_ids) {
            // 2.a. Update summary
            MyPostingList& posting_list = posting_lists_[component];
            posting_list.delete_doc(block_id, doc_id, *jlt_, forward_index_, config_);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time).count();
    }

    void insert_doc(const std::vector<uint16_t>& components, const std::vector<float>& values)
    {
        // 1. Add doc to forward index
        auto start_time = std::chrono::high_resolution_clock::now();
        size_t doc_id = forward_index_.len();
        forward_index_.push(components, values);

        std::vector<std::pair<size_t, size_t>> block_ids;
        block_ids.reserve(components.size());

        // 2. Add doc to posting lists for each of its components
        start_time = std::chrono::high_resolution_clock::now();
        //std::cout << "Adding document to " << components.size() << " postings lists." << std::endl;
        for (size_t c = 0; c < components.size(); ++c) {
            // 2.a. Find closest block, add document to block, and update summary (single function that returns block_id)
            if (components[c] >= posting_lists_.size()) break;
            MyPostingList& posting_list = posting_lists_[components[c]];
            size_t block_id = posting_list.insert_doc(components, values, *jlt_, config_, forward_index_.id_to_offset(doc_id), components.size());

            // 2.b. Keep track of which block was added to
            block_ids.emplace_back(components[c], block_id);
        }

        // 3. Update doc_block_membership_
        doc_block_membership_.push_back(block_ids);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time).count();
        //std::cout << "Index: Time to insert doc " << doc_id << ": " << elapsed << std::endl;
    }

    // Call this function if too empty (too many tombstones/deletes)
    void resize()
    {
        // 1. Iterate through posting lists
        for (size_t i = 0; i < posting_lists_.size(); ++i) {
            // 2. Delete from each posting list
            MyPostingList& posting_list = posting_lists_[i];
            posting_list.resize(forward_index_);
        }
    }

    // Search the index
    std::vector<std::pair<float, size_t>> search(
        const std::vector<uint16_t>& query_components,
        const std::vector<float>& query_values,
        size_t k,
        size_t query_cut,
        float heap_factor,
        size_t n_knn,
        bool first_sorted,
        float alpha = 0.0f,
        bool debug = false) const
    {
        // Create a dense query vector
        std::vector<float> query(dim(), 0.0f);
        for (size_t i = 0; i < query_components.size(); ++i) {
            if (query_components[i] >= query.size()) {
                std::cout << "Dense vector creation" << std::endl;
                continue;
            }
            query[query_components[i]] = query_values[i];
        }

        // L2 norm of the query, for the centroid+radius pruning bound.
        float qnorm = 0.0f;
        for (float v : query_values) qnorm += v * v;
        qnorm = std::sqrt(qnorm);
         
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
            if (component_id >= posting_lists_.size()) {
                std::cout << "Posting list component too large" << std::endl;
                continue;
            }

            posting_lists_[component_id].search(
                query,
                query_components,
                query_values,
                k,
                heap_factor,
                heap,
                visited,
                forward_index_,
                i == 0 && first_sorted,
                *jlt_,
                config_,
                alpha,
                qnorm,
                debug
            );
        }
         
        if (knn_ && n_knn > 0) {
            knn_->refine<float>(query, heap, visited, forward_index_, n_knn);
        }

        if (debug) {
            std::cout << "Docs evaluated: " << visited.size() << std::endl;
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
    static MyInvertedIndex<T> build(const SparseDatasetMut<T>& dataset, const Configuration& config) 
    { 
        auto time_start = std::chrono::high_resolution_clock::now();
        std::cout << "Building with the following settings:\n"
            << "DEBUG dynamic-support: " << config.get_dynamic_support() << "\n"
            << "DEBUG transform-function: " << config.get_transform_function() << "\n"
            << "DEBUG summarization-metric: " << config.get_summarization_metric() << std::endl;
        // Distribute pairs (score, doc_id) to corresponding components for each chunk.
        // We use pairs because later each posting list will be sorted by score
        // by the pruning strategy.
        // The pruning strategy is applied to partial results, for Global Threshold strategy
        // the final fixed pruning is done only when all chunks have been parsed

        // Initialize JLT matrix for shared use
        auto orig_d = dataset.len();
        auto transform_d = std::log(orig_d);

        std::unique_ptr<SparseJLT> jlt_ptr;
        if (config.get_transform_function() == "jlt") {
            jlt_ptr = std::make_unique<SparseJLT>(orig_d, transform_d);
        }

        auto jlt_elapsed = std::chrono::high_resolution_clock::now() - time_start;
        std::cout << "DEBUG JLT-matrix-initialization: " << std::chrono::duration_cast<std::chrono::seconds>(jlt_elapsed).count() << " seconds" << std::endl;

        std::vector<std::vector<std::pair<T, size_t>>> inverted_pairs(dataset.dim());
        std::vector<std::vector<std::pair<T, size_t>>> chunk_inv_pairs(dataset.dim());
         
        size_t chunk_size = config.get_batched_indexing().value_or(dataset.len());
        
        std::cout << "Distributing and pruning posting lists:" << std::endl;
        auto debug_start = std::chrono::high_resolution_clock::now();
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
                std::cout << "DEBUG Fixed-pruning" << std::endl;
                // Truncate each posting list to its top n_postings. This was previously
                // commented out, leaving lists fully unpruned -- which both exploded the
                // build cost (clustering ran on full lists) and made queries evaluate
                // ~all docs per list. Re-enabling restores paper-faithful pruning.
                fixed_pruning(inverted_pairs, pruning.get_n_postings());
            } else if (pruning.get_type() == PruningStrategy::Type::GlobalThreshold) {
                std::cout << "DEBUG Global-pruning" << std::endl;
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
        auto debug_elapsed = std::chrono::high_resolution_clock::now() - debug_start;
        std::cout << "DEBUG distribute-and-prune-postings: " << std::chrono::duration_cast<std::chrono::seconds>(debug_elapsed).count() << " seconds" << std::endl;

        std::cout << "\tNumber of posting lists: " << inverted_pairs.size() << std::endl;
         
        std::cout << "Building summaries: " << std::endl;
        debug_start = std::chrono::high_resolution_clock::now();
         
        // Build summaries and blocks for each posting list
        std::vector<MyPostingList> posting_lists(inverted_pairs.size());
        
        // Initialize vector to map where each document exists in the inverted index (for efficient deletions)
        std::vector<std::vector<std::pair<size_t, size_t>>> doc_block_membership;
        doc_block_membership.resize(dataset.len());

        // Dynamic scheduling: posting-list sizes are highly skewed (most lists are
        // tiny, a few are at the pruning cap and dominate build cost). Static
        // scheduling left threads idle at the tail; dynamic lets free threads grab
        // remaining heavy lists.
        #pragma omp parallel for schedule(dynamic, 16)
        for (size_t i = 0; i < inverted_pairs.size(); ++i) {
            posting_lists[i] = MyPostingList::build(dataset, inverted_pairs[i], config, *jlt_ptr);
        }
        std::cout << "Finished building all posting lists" << std::endl;

        // If dynamic, need to record doc to block -- otherwise we can skip in the static case
        if (config.get_dynamic_support()) {
            for (size_t i = 0; i < posting_lists.size(); ++i) {
                auto doc_block_pairs = posting_lists[i].get_doc_block_pairs();
                for (const auto& [doc_id, block_id] : doc_block_pairs) {
                    if (doc_id >= doc_block_membership.size()) {
                        std::cerr << "Bad doc_id " << doc_id << " (max " << doc_block_membership.size()-1 << ")\n";
                        continue;
                    }
                    doc_block_membership[doc_id].emplace_back(i, block_id);
                }
            }
        }

        debug_elapsed = std::chrono::high_resolution_clock::now() - debug_start;
        std::cout << "DEBUG build-summaries: " << std::chrono::duration_cast<std::chrono::seconds>(debug_elapsed).count() << " seconds" << std::endl;

        MyInvertedIndex<T> index(dataset, std::move(posting_lists), config, std::move(doc_block_membership), std::move(jlt_ptr));

        auto elapsed = std::chrono::high_resolution_clock::now() - time_start;
        std::cout << "DEBUG Total-time-to-build:" << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() << " secs" << std::endl;
         
        // Handle KNN if needed
        if (config.get_knn_config().get_nknn() == 0 && !config.get_knn_config().get_knn_path().has_value()) {
            std::cout << "No knn needed" << std::endl;
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
         
        return MyInvertedIndex<T>(index.forward_index_, std::move(index.posting_lists_), config, 
                                   std::move(index.doc_block_membership_), std::move(index.jlt_), std::move(knn));
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
    const SparseDatasetMut<T>& dataset() const { return forward_index_; }

    // Read-only access to posting lists (used by diagnostics).
    const std::vector<MyPostingList>& posting_lists() const { return posting_lists_; }
 
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
    size_t space_usage_byte() const { return 0; }
     
    // Space usage
    size_t print_space_usage_byte() { return 0; }

    template <class Archive>
    void serialize(Archive& archive) {
        archive(forward_index_, posting_lists_, config_, doc_block_membership_, jlt_, knn_);
    }
};

} // namespace seismic

#endif // MY_INVERTED_INDEX_H