#ifndef DYNAMIC_INVERTED_INDEX_H
#define DYNAMIC_INVERTED_INDEX_H

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

#include "sparse_dataset.h"
#include "configuration_strategies.h"
#include "dynamic_posting_list.h"

namespace seismic
{

template <typename T>
class DynamicInvertedIndex : public SpaceUsage {
public:
    SparseDatasetMut<T> forward_index_;
    std::vector<DynamicPostingList> posting_lists_;
    Configuration config_;
    std::optional<Knn> knn_;

    std::vector<uint8_t> alive_;    // same length as forward_index_.len()
    std::vector<std::vector<std::pair<uint16_t, size_t>>> doc_block_membership_;  // doc_block_membership_[doc_id] = list of (component_id, block_id)

public:
    // Default constructor
    DynamicInvertedIndex() = default;
     
    // Constructor with parameters
    DynamicInvertedIndex(SparseDatasetMut<T> forward_index, std::vector<DynamicPostingList> posting_lists, 
                Configuration config, std::optional<Knn> knn = std::nullopt)
        : forward_index_(std::move(forward_index)), 
        posting_lists_(std::move(posting_lists)),
        config_(std::move(config)),
        knn_(std::move(knn)) 
    {
        const size_t n = forward_index_.len();
        alive_.assign(n, 1);
        doc_block_membership_.resize(n);
    };
    
    static DynamicInvertedIndex<T> build(const SparseDatasetMut<T>& dataset, const Configuration& config);

    // Search the index
    std::vector<std::pair<float, size_t>> search(
        const std::vector<uint16_t>& query_components,
        const std::vector<float>& query_values,
        size_t k,
        size_t query_cut,
        float heap_factor,
        size_t n_knn,
        bool first_sorted) const;

    // Insert document to index
    void insert_document(const std::vector<uint16_t>& components, const std::vector<float>& values);

    // Remove document from index
    void delete_document(size_t doc_id);

    // Rebuild index from tombstones
    void rebuild_index();

    // Pruning strategies
    static void fixed_pruning(std::vector<std::vector<std::pair<T, size_t>>>& inverted_pairs, size_t n_postings);
    static void global_threshold_pruning(std::vector<std::vector<std::pair<T, size_t>>>& inverted_pairs, size_t n_postings);

    void add_knn(Knn new_knn){ knn_ = std::move(new_knn); }

    // Getters
    const std::optional<Knn>& knn() const { return knn_; }
    const SparseDatasetMut<T>& dataset() const { return forward_index_; }
 
    // Returns (offset, len) of the "id"-th document
    std::pair<size_t, size_t> id_to_offset_len(size_t id) const { return forward_index_.id_to_offset_len(id); }
 
    // Returns the id of the largest component, i.e., the dimensionality of the vectors in the dataset.
    size_t dim() const { return forward_index_.dim(); }
 
    // Returns the number of non-zero components in the dataset.
    size_t nnz() const { return forward_index_.nnz(); }
 
    // Returns the number of vectors in the dataset
    size_t len() const { return forward_index_.len(); }
 
    // Checks if the dataset is empty.
    bool is_empty() const { return forward_index_.len() == 0; }
 
    // Returns the number of neighbors in the knn graph, 0 if knn graph is not present.
    size_t knn_len() const { return knn_.has_value() ? knn_->get_d() : 0; }
 
    // Space usage
    size_t space_usage_byte() const override;
     
    // Space usage
    size_t print_space_usage_byte() const { return 0; }

    // Archiving functionality
    template <class Archive>
    void serialize(Archive& archive) { archive(forward_index_, posting_lists_, config_, knn_); }
};

template <typename T>
DynamicInvertedIndex<T> DynamicInvertedIndex<T>::build(const SparseDatasetMut<T>& dataset, const Configuration& config) { 
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

        auto test = std::chrono::high_resolution_clock::now() - time_start;
        std::cout << "Testing 1: " << std::chrono::duration_cast<std::chrono::seconds>(test).count() << " secs" << std::endl;
        
        // Process each document in the chunk
        for (size_t i = doc_id; i < end_id; ++i) {
            auto [components, values] = dataset.get(i);
            
            for (size_t j = 0; j < components.size(); ++j) {
                uint16_t c = components[j];
                T score = values[j];
                chunk_inv_pairs[c].emplace_back(score, i);
            }
        }

        test = std::chrono::high_resolution_clock::now() - time_start;
        std::cout << "Testing 2: " << std::chrono::duration_cast<std::chrono::seconds>(test).count() << " secs" << std::endl;

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
        
        test = std::chrono::high_resolution_clock::now() - time_start;
        std::cout << "Testing 3: " << std::chrono::duration_cast<std::chrono::seconds>(test).count() << " secs" << std::endl;

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
    

    auto test = std::chrono::high_resolution_clock::now() - time_start;
    std::cout << "Testing 4: " << std::chrono::duration_cast<std::chrono::seconds>(test).count() << " secs" << std::endl;

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
    std::vector<DynamicPostingList> posting_lists(inverted_pairs.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < inverted_pairs.size(); ++i) {
        posting_lists[i] = DynamicPostingList::build(dataset, inverted_pairs[i], config);
    }
    
    DynamicInvertedIndex<T> index(dataset, std::move(posting_lists), config);
    
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
    
    return DynamicInvertedIndex<T>(index.forward_index_, std::move(index.posting_lists_), config, std::move(knn));
}

template <typename T>
size_t DynamicInvertedIndex<T>::space_usage_byte() const {
    size_t bytes = 0;

    // Forward index
    bytes += forward_index_.space_usage_byte();   // if SparseDatasetMut<T> has this
    // If not, approximate:
    // bytes += sizeof(forward_index_);

    // Posting lists
    for (const auto& pl : posting_lists_) {
        bytes += pl.space_usage_byte();
    }

    // Config
    bytes += sizeof(config_);

    // KNN graph (if it has a space_usage_byte, otherwise sizeof)
    if (knn_) {
        bytes += knn_->space_usage_byte();   // or sizeof(*knn_);
    }

    // Dynamic metadata
    bytes += alive_.size() * sizeof(uint8_t);
    for (const auto& v : doc_block_membership_) {
        bytes += v.size() * sizeof(std::pair<uint16_t, size_t>);
    }

    return bytes;
}

template <typename T>
void DynamicInvertedIndex<T>::global_threshold_pruning(std::vector<std::vector<std::pair<T, size_t>>>& inverted_pairs, size_t n_postings)
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

template <typename T>
void DynamicInvertedIndex<T>::fixed_pruning(std::vector<std::vector<std::pair<T, size_t>>>& inverted_pairs, size_t n_postings)
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

// Search the index
template <typename T>
std::vector<std::pair<float, size_t>> DynamicInvertedIndex<T>::search(
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
            i == 0 && first_sorted,
            alive_
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

template <typename T>
void DynamicInvertedIndex<T>::insert_document(const std::vector<uint16_t>& components, const std::vector<float>& values)
{
    assert(components.size() == values.size());
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    // 1. Append to forward_index_ (or use whatever mutating API you have)
    // Assume forward_index_.append returns new doc_id:
    start_time = std::chrono::high_resolution_clock::now();
    size_t doc_id = forward_index_.add_document(components, values);
    end_time = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "1. Append to forward Index " << elapsed << std::endl;

    // 2. Ensure bookkeeping vectors are long enough
    start_time = std::chrono::high_resolution_clock::now();
    if (doc_id >= alive_.size()) {
        alive_.resize(doc_id + 1, 0);
        doc_block_membership_.resize(doc_id + 1);
    }
    alive_[doc_id] = 1;
    doc_block_membership_[doc_id].clear();
    end_time = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "2. Ensure vectors long enough " << elapsed << std::endl;

    // 3. For each component, assign to closest block and update max summary
    size_t d = forward_index_.dim();

    for (size_t i = 0; i < components.size(); ++i) {
        uint16_t comp = components[i];
        T val = static_cast<T>(values[i]);

        if (comp >= posting_lists_.size()) {
            // you might want to resize posting_lists_ if new components appear
            continue;
        }

        DynamicPostingList& pl = posting_lists_[comp];

        // ensure dynamic state is initialized
        if (pl.block_docs.empty()) {
            pl.initialize_dynamic_state(d);
        }
        start_time = std::chrono::high_resolution_clock::now();
        // 3.a pick closest block for this doc relative to this posting list’s summaries
        size_t block_id = pl.find_closest_block(components, values);
        end_time = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cout << "3.a Find closest block " << elapsed << std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        // 3.b add posting into that block
        pl.add_posting_to_block(block_id, doc_id, val);
        end_time = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cout << "3.b Add posting to block " << elapsed << std::endl;

        // 3.c record membership for later deletes
        start_time = std::chrono::high_resolution_clock::now();
        pl.block_docs[block_id].push_back(doc_id);
        doc_block_membership_[doc_id].emplace_back(comp, block_id);
        end_time = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cout << "3.c Record membership " << elapsed << std::endl;

        // 3.d update block_max = elementwise max(old, doc_vec)
        start_time = std::chrono::high_resolution_clock::now();
        auto& max_vec = pl.block_max[block_id];
        if (max_vec.size() < d) max_vec.resize(d, 0.0f);

        // sparse doc: only update coordinates present
        for (size_t j = 0; j < components.size(); ++j) {
            uint16_t c = components[j];
            float v = values[j];
            if (c < max_vec.size()) {
                if (v > max_vec[c]) {
                    max_vec[c] = v;
                }
            }
        }
        end_time = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cout << "3.d Update block_max " << elapsed << std::endl;

        // 3.e push updated max into QuantizedSummary
        start_time = std::chrono::high_resolution_clock::now();
        pl.set_block_summary_from_max(block_id);
        end_time = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cout << "3.e. Update summary " << elapsed << std::endl;
    }
}

template <typename T>
void DynamicInvertedIndex<T>::delete_document(size_t doc_id)
{
    if (doc_id >= alive_.size() || !alive_[doc_id]) {
        //return;
    }

    alive_[doc_id] = 0;

    // Get the doc’s block memberships
    auto& memberships = doc_block_membership_[doc_id];

    // We'll need doc vectors during recompute, but note:
    // recompute_block_max() will fetch all doc vectors directly from forward_index_.
    // No need to fetch this deleting doc's vector here.

    for (const auto& [comp, block_id] : memberships) {
        //if (comp >= posting_lists_.size()) continue;
        DynamicPostingList& pl = posting_lists_[comp];

        // Recompute block max from alive docs in this block
        std::cout << "Comp " << comp << std::endl;
        pl.recompute_block_max(block_id, forward_index_, alive_);
    }
}

} // namespace seismic

#endif // DYNAMIC_INVERTED_INDEX_H
