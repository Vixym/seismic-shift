#ifndef DYNAMIC_POSTING_LIST_H
#define DYNAMIC_POSTING_LIST_H

#include <vector>
#include <cstdint>
#include <utility>
#include <unordered_set>
#include <limits>

#include "sparse_dataset.h"
#include "quantized_summary.h"

#include "configuration_strategies.h"

namespace seismic
{

struct DynamicPostingList : public SpaceUsage {
    std::vector<uint64_t> packed_postings;
    std::vector<size_t> block_offsets;   // size = n_blocks + 1
    QuantizedSummary summaries;

    std::vector<size_t> block_counts;
    std::vector<std::vector<size_t>> block_docs;  // per-block doc ids (for this component’s list)
    std::vector<std::vector<float>> block_max;  // per-block max vectors (unquantized) [block][dim]

    // Default constructor
    DynamicPostingList() = default;
        
    // Constructor with parameters
    DynamicPostingList(std::vector<uint64_t> packed_postings, std::vector<size_t> block_offsets, QuantizedSummary summaries)
        : packed_postings(std::move(packed_postings)),
            block_offsets(std::move(block_offsets)),
            summaries(std::move(summaries)) {}

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
        std::vector<uint8_t>) const;

    template <typename T>
    static DynamicPostingList build(
        const SparseDatasetMut<T>& dataset,
        const std::vector<std::pair<T, size_t>>& postings,
        const Configuration& config);

    template <typename T>
    void evaluate_posting_block(
        const std::vector<float>& query,
        const std::vector<uint16_t>& query_term_ids,
        const std::vector<float>& query_values,
        const std::vector<uint64_t>& packed_posting_block,
        utils::HeapFaiss& heap,
        std::unordered_set<size_t>& visited,
        const SparseDatasetMut<T>& forward_index,
        const std::vector<uint8_t>& alive) const;

    static std::vector<size_t> fixed_size_blocking(const std::vector<size_t>& posting_list, size_t block_size);

    template <typename T>
    static std::vector<size_t> blocking_with_random_kmeans(
        std::vector<size_t>& posting_list,
        float centroid_fraction,
        size_t min_cluster_size,
        const SparseDatasetMut<T>& dataset,
        const ClusteringAlgorithm& clustering_algorithm);

    template <typename T>
    static std::pair<std::vector<uint16_t>, std::vector<T>> fixed_size_summary(
        const SparseDatasetMut<T>& dataset,
        const std::vector<size_t>& block,
        size_t n_components);

    template <typename T>
    static std::pair<std::vector<uint16_t>, std::vector<T>> energy_preserving_summary(
        const SparseDatasetMut<T>& dataset,
        const std::vector<size_t>& block,
        float fraction);

    void initialize_dynamic_state(size_t dim);

    // Pack offset and length into a single uint64_t
    static uint64_t pack_offset_len(size_t offset, size_t len) { return ((static_cast<uint64_t>(offset) << 16) | (len & 0xFFFF)); }

    // Unpack offset and length from a single uint64_t
    static std::pair<size_t, size_t> unpack_offset_len(uint64_t pack) { return {static_cast<size_t>(pack >> 16), static_cast<size_t>(pack & 0xFFFF)}; }
    
    // Space usage calculation
    size_t space_usage_byte() const override {
        return packed_postings.size() * sizeof(uint64_t) +
                block_offsets.size() * sizeof(size_t) +
                summaries.space_usage_byte();
    }

    // Helper APIs:
    size_t find_closest_block(const std::vector<uint16_t>& comps, const std::vector<float>& vals) const;

    template <typename T>
    void add_posting_to_block(size_t block_id, size_t doc_id, T value);

    // Recompute block_max[block_id] from scratch given forward_index & alive[]
    template <typename TDataset>
    void recompute_block_max(size_t block_id,
                             const TDataset& forward_index,
                             const std::vector<uint8_t>& alive);

    void set_block_summary_from_max(size_t block_id);

    // Archiving functionality
    template <class Archive>
    void serialize(Archive& archive) { archive(packed_postings, block_offsets, summaries); }
};

template <typename T>
DynamicPostingList DynamicPostingList::build(
    const SparseDatasetMut<T>& dataset,
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
            
    return DynamicPostingList(
        std::move(packed_postings),
        std::move(block_offsets),
        QuantizedSummary::from_sparse_dataset(summaries_dataset.to_immutable())
    );
}

// Blocking strategies
std::vector<size_t> DynamicPostingList::fixed_size_blocking(const std::vector<size_t>& posting_list, size_t block_size) {
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
std::vector<size_t> DynamicPostingList::blocking_with_random_kmeans(
    std::vector<size_t>& posting_list,
    float centroid_fraction,
    size_t min_cluster_size,
    const SparseDatasetMut<T>& dataset,
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
std::pair<std::vector<uint16_t>, std::vector<T>> DynamicPostingList::fixed_size_summary(
    const SparseDatasetMut<T>& dataset,
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
std::pair<std::vector<uint16_t>, std::vector<T>> DynamicPostingList::energy_preserving_summary(
    const SparseDatasetMut<T>& dataset,
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

template <typename T>
void DynamicPostingList::evaluate_posting_block(
    const std::vector<float>& query,
    const std::vector<uint16_t>& query_term_ids,
    const std::vector<float>& query_values,
    const std::vector<uint64_t>& packed_posting_block,
    utils::HeapFaiss& heap,
    std::unordered_set<size_t>& visited,
    const SparseDatasetMut<T>& forward_index,
    const std::vector<uint8_t>& alive) const 
{
    if (packed_posting_block.empty()) {
        return;
    }
    
    // First posting
    auto [prev_offset, prev_len] = unpack_offset_len(packed_posting_block[0]);
    
    // Middle postings
    for (size_t i = 1; i < packed_posting_block.size(); ++i) {
        auto [offset, len] = unpack_offset_len(packed_posting_block[i]);
        
        // ---- NEW: skip tombstoned docs ----
        // if (prev_offset >= alive.size() || !alive[prev_offset]) {
        //     prev_offset = offset;
        //     prev_len = len;
        //     continue;
        // }
        
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
    
    // Last posting
    // ---- NEW: skip tombstoned docs ----
    // if (prev_offset >= alive.size() || !alive[prev_offset]) {
    //     return;
    // }

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
}

// Search implementation
template <typename T>
void DynamicPostingList::search(
    const std::vector<float>& query,
    const std::vector<uint16_t>& query_components,
    const std::vector<float>& query_values,
    size_t k,
    float heap_factor,
    utils::HeapFaiss& heap,
    std::unordered_set<size_t>& visited,
    const SparseDatasetMut<T>& forward_index,
    bool sort_summaries,
    std::vector<uint8_t> alive) const 
{
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
                    forward_index,
                    alive
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
            forward_index,
            alive
        );
    }
}

void DynamicPostingList::initialize_dynamic_state(size_t dim) {
    size_t n_blocks = block_offsets.empty() ? 0 : block_offsets.size() - 1;
    block_docs.assign(n_blocks, {});
    block_max.assign(n_blocks, std::vector<float>(dim, 0.0f));
}

size_t DynamicPostingList::find_closest_block(const std::vector<uint16_t>& comps, const std::vector<float>& vals) const
{
    auto scores = summaries.distances_iter(comps, vals); // one score per block

    size_t best_block = 0;
    float best_score = std::numeric_limits<float>::lowest();

    for (size_t b = 0; b < scores.size(); ++b) {
        float s = scores[b].value_or(std::numeric_limits<float>::lowest());
        if (s > best_score) {
            best_score = s;
            best_block = b;
        }
    }
    return best_block;
}

void DynamicPostingList::set_block_summary_from_max(size_t block_id)
{
    const auto& max_vec = block_max[block_id];
    // This should quantize max_vec into the summary’s representation
    summaries.set_block_from_dense(block_id, max_vec);
}

template <typename T>
void DynamicPostingList::add_posting_to_block(size_t block_id, size_t doc_id, T value)
{
    // Pack (doc_id, value) however your code does it:
    uint64_t packed = pack_offset_len(doc_id, value); // TODO: use your real packer

    // Compute insertion position: end of that block
    size_t start = block_offsets[block_id];
    size_t end   = block_offsets[block_id + 1];

    // Insert at `end`
    packed_postings.insert(packed_postings.begin() + end, packed);

    // Update block_offsets for subsequent blocks
    for (size_t b = block_id + 1; b < block_offsets.size(); ++b) {
        block_offsets[b] += 1;
    }

    // Update counts (for summary updates)
    if (block_id >= block_counts.size()) {
        block_counts.resize(block_id + 1, 0);
    }
    block_counts[block_id] += 1;
}

template <typename TDataset>
void DynamicPostingList::recompute_block_max(size_t block_id,
                                      const TDataset& forward_index,
                                      const std::vector<uint8_t>& alive)
{
    std::cout << "block " << block_id
          << " has " << block_docs[block_id].size() << " docs\n";
    size_t d = forward_index.dim();
    //if (block_id >= block_docs.size()) return;

    std::vector<float> new_max(d, 0.0f);

    std::vector<uint16_t> comps;
    std::vector<float> vals;

    for (size_t doc_id : block_docs[block_id]) {
        if (doc_id >= alive.size() || !alive[doc_id]) {
            continue;
        }

        // get sparse doc vector from forward_index
        auto pair = forward_index.get(doc_id);
        comps = pair.first;
        vals = pair.second;
        for (size_t i = 0; i < comps.size(); ++i) {
            uint16_t c = comps[i];
            float v = vals[i];
            if (c < new_max.size() && v > new_max[c]) {
                new_max[c] = v;
            }
        }
    }

    block_max[block_id].swap(new_max);

    // NEW: also update the quantized summary so search sees the new max
    set_block_summary_from_max(block_id);
}

} // namespace seismic

#endif // DYNAMIC_POSTING_LIST_H