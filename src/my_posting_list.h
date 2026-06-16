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
        std::vector<float> block_radii_; // per-block residual radius R = max_d ||d - mu|| (for centroid+radius pruning)

    public:
        // Default constructor
        MyPostingList() = default;

        // Constructor with parameters
        MyPostingList(std::vector<uint64_t> packed_postings, std::vector<size_t> block_offsets,
                   std::vector<std::pair<size_t, SparseVector>> summaries, std::vector<std::pair<size_t, size_t>> doc_block_pairs,
                   std::vector<float> block_radii = {})
            : packed_postings_(std::move(packed_postings)),
              block_offsets_(std::move(block_offsets)),
              summaries_(std::move(summaries)),
              doc_block_pairs_(std::move(doc_block_pairs)),
              block_radii_(std::move(block_radii)) {}

        // Read-only accessors (used by diagnostics).
        const std::vector<uint64_t>& packed_postings() const { return packed_postings_; }
        const std::vector<size_t>& block_offsets() const { return block_offsets_; }
        size_t num_blocks() const { return block_offsets_.empty() ? 0 : block_offsets_.size() - 1; }

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
                block_offsets = fixed_size_blocking(posting_list, blocking.get_block_size(), dataset);
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
            std::vector<float> block_radii;
            block_radii.reserve(block_offsets.size());

            for (size_t i = 0; i < block_offsets.size() - 1; ++i) {
                std::vector<size_t> block(
                    posting_list.begin() + block_offsets[i],
                    posting_list.begin() + block_offsets[i + 1]
                );

                std::vector<std::pair<uint16_t, float>> summary;

                const auto& summarization = config.get_summarization();
                const auto& summarization_metric = config.get_summarization_metric();
                if (summarization_metric == "max") {
                    summary = MaxSummary::summary_init_energy_preserving(dataset, block, summarization.get_summary_energy());
                } else if (summarization_metric == "centroid") {
                    summary = CentroidSummary::summary_init_energy_preserving(dataset, block, summarization.get_summary_energy());
                }

                // Per-block residual radius R = max_d ||d - mu||, computed in the
                // summary's (pre-transform) space. Used for the centroid+radius
                // pruning bound U = q.mu + alpha*||q||*R. Meaningful when transform
                // == "none"; with JLT the doc/summary live in different spaces, so
                // we store 0 (radius pruning disabled) for now.
                float radius = 0.0f;
                if (config.get_transform_function() != "jlt") {
                    radius = compute_block_radius(dataset, block, summary);
                }
                block_radii.push_back(radius);

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
                std::move(doc_block_pairs),
                std::move(block_radii)
            );
        }

        // Compute the block radius R = max_{d in block} ||d - mu||_2 (mu given as a
        // sparse vector). Used at build time for the centroid+radius pruning bound.
        template <typename T>
        static float compute_block_radius(const SparseDatasetMut<T>& dataset,
                                          const std::vector<size_t>& block,
                                          const SparseVector& mu) {
            std::unordered_map<uint16_t, float> mumap;
            double munorm2 = 0.0;
            for (const auto& [c, v] : mu) { mumap[c] = v; munorm2 += double(v) * v; }
            double R2 = 0.0;
            for (size_t doc_id : block) {
                auto dv = dataset.get_view(doc_id);
                double r2 = munorm2;
                for (size_t j = 0; j < dv.len; ++j) {
                    double d = dv.values[j];
                    auto it = mumap.find(dv.components[j]);
                    double m = (it != mumap.end()) ? it->second : 0.0;
                    r2 += d * d - 2.0 * d * m;
                }
                if (r2 < 0) r2 = 0;
                R2 = std::max(R2, r2);
            }
            return static_cast<float>(std::sqrt(R2));
        }

        // ||a - b||_2 for two sparse vectors sorted by component id.
        static float sparse_residual_norm(const SparseVector& a, const SparseVector& b) {
            size_t i = 0, j = 0; double s = 0.0;
            while (i < a.size() && j < b.size()) {
                if (a[i].first == b[j].first) { double d = double(a[i].second) - b[j].second; s += d * d; ++i; ++j; }
                else if (a[i].first < b[j].first) { s += double(a[i].second) * a[i].second; ++i; }
                else { s += double(b[j].second) * b[j].second; ++j; }
            }
            for (; i < a.size(); ++i) s += double(a[i].second) * a[i].second;
            for (; j < b.size(); ++j) s += double(b[j].second) * b[j].second;
            return static_cast<float>(std::sqrt(s));
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
                new_summary = MaxSummary::summary_delete(dataset, summary, doc_vec, block, config.get_summarization().get_n_components(), config.get_summarization().get_summary_energy());
            } else if (summary_metric == "centroid") {
                new_summary = CentroidSummary::summary_delete(summary, doc_vec, num_docs_in_block);
            } else {
                std::cout << "my_posting_list.h:delete_doc invalid summary metric" << std::endl;
            }

            // Maintain the block radius for the centroid+radius pruning bound, O(nnz),
            // no block scan: the centroid shifts by delta = (mu - d)/(n-1), and by the
            // triangle inequality every surviving residual grows by at most ||delta||,
            // so R_new <= R_old + ||delta|| stays a valid upper bound. It only loosens
            // (never wrongly skips), and resize() periodically recomputes a tight R.
            if (summary_metric == "centroid" && transform_function != "jlt" &&
                block_id < block_radii_.size() && num_docs_in_block > 1) {
                float resid = sparse_residual_norm(doc_vec, summary); // ||d - mu_old||
                block_radii_[block_id] += resid / float(num_docs_in_block - 1);
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
            float best_val = -std::numeric_limits<float>::infinity();
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

            // Maintain block radius (centroid+radius bound), O(nnz): inserting d shifts
            // the centroid by delta_ins = (d - mu)/(n+1); existing residuals grow by at
            // most ||delta_ins|| and the new doc has residual ||d - mu_new||, so
            // R_new = max(R_old + ||delta_ins||, ||d - mu_new||).
            if (summary_metric == "centroid" && transform_function != "jlt" &&
                best_block < block_radii_.size()) {
                size_t n = summaries_[best_block].first; // docs before this insert
                float res_old = sparse_residual_norm(doc_vec, summaries_[best_block].second); // ||d - mu_old||
                float res_new = sparse_residual_norm(doc_vec, new_summary);                   // ||d - mu_new||
                float delta = res_old / float(n + 1);
                block_radii_[best_block] = std::max(block_radii_[best_block] + delta, res_new);
            }

            summaries_[best_block] = {summaries_[best_block].first+1, new_summary};

            // 4. Return block id of closest block
            return best_block;
        }

        template <typename T>
        void resize(const SparseDatasetMut<T>& dataset)
        {
            std::vector<uint64_t> new_packed_postings;
            std::vector<size_t> new_block_offsets;
            new_block_offsets.push_back(0);

            for (size_t block_id = 0; block_id < block_offsets_.size()-1; ++block_id) {
                const size_t starting_id = block_offsets_[block_id];
                const size_t ending_id = block_offsets_[block_id+1];

                for (size_t posting_idx = starting_id; posting_idx < ending_id; ++posting_idx) {
                    uint64_t packed_posting = packed_postings_[posting_idx];
                    auto [offset, len] = unpack_offset_len(packed_posting);
                    size_t id = dataset.offset_to_id(offset);
                    if (dataset.is_alive(id)) {
                        new_packed_postings.push_back(packed_posting);
                    }
                }

                new_block_offsets.push_back(new_packed_postings.size());
            }

            packed_postings_ = new_packed_postings;
            block_offsets_ = new_block_offsets;
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
            const Configuration& config,
            float alpha = 0.0f,
            float qnorm = 0.0f,
            const bool debug = false) const
        {
            auto start = std::chrono::high_resolution_clock::now();
            if (debug) {
                start = std::chrono::high_resolution_clock::now();
            }
            std::vector<std::vector<uint64_t>> blocks_to_evaluate;

            // Get distances between query and summaries vector of (idx, val) pairs
            std::vector<std::pair<uint16_t, float>> indexed_dots = get_distances(query_components, query_values, jlt, config, debug);
            
            // Always process blocks best-first (by query·summary). This lets the top-k
            // heap fill with high-scoring docs early, which raises the pruning threshold
            // quickly so most remaining (lower-scoring) blocks can be skipped. Processing
            // blocks in arbitrary order leaves the threshold low and ends up evaluating
            // nearly every doc in the list (the cause of the latency regression).
            (void)sort_summaries;
            std::sort(indexed_dots.begin(), indexed_dots.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });
            const size_t block_threshold = indexed_dots.size() * .75;
            size_t num_blocks_processed = 0;
            for (const auto& [block_id, dot] : indexed_dots) {
                num_blocks_processed += 1;
                // Skip blocks that cannot contribute to the top-k
                if (config.get_summarization_metric() == "max"){
                    if (heap.len() == k && dot < -heap_factor * heap.front_distance()) continue;
                } else if (config.get_summarization_metric() == "centroid") {
                    if (alpha > 0.0f && block_id < block_radii_.size()) {
                        // Centroid + residual-radius bound: an upper bound on any doc's
                        // score in the block is  U = q.mu + alpha*||q||*R  (alpha<1 makes
                        // it an aggressive/approximate bound; see diagnostics). Reuse the
                        // max-style threshold skip.
                        float U = dot + alpha * qnorm * block_radii_[block_id];
                        if (heap.len() == k && U < -heap_factor * heap.front_distance()) continue;
                    } else {
                        // Legacy count-based heuristic (no radius available).
                        if (heap.len() == k && num_blocks_processed >= block_threshold) continue;
                    }
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
            if (debug) {
                auto elapsed = std::chrono::high_resolution_clock::now() - start;
                std::cout << "Time to search:" << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " microseconds" << std::endl;
            }
        }

        std::vector<std::pair<uint16_t, float>> get_distances(const std::vector<uint16_t>& query_components, const std::vector<float>& query_values,
            const SparseJLT& jlt, const Configuration& config, const bool debug = false) const
        {
            (void) jlt;
            (void) config;
            auto start = std::chrono::high_resolution_clock::now();
            auto transform_start = std::chrono::high_resolution_clock::now();
            if (debug) {
                start = std::chrono::high_resolution_clock::now();
            }
            const std::vector<uint16_t>* components = &query_components;
            const std::vector<float>* values = &query_values;

            std::vector<std::pair<uint16_t, float>> distances;
            distances.resize(summaries_.size());

            // Transform vectors into new space if needed
            if (debug) {
                transform_start = std::chrono::high_resolution_clock::now();
            }
            std::vector<uint16_t> transformed_components;
            std::vector<float> transformed_values;
            if (config.get_transform_function() == "jlt") {
                auto transformed_vec = jlt.transform(query_components, query_values);

                for (const auto& [c, v] : transformed_vec) {
                    transformed_components.push_back(c);
                    transformed_values.push_back(v);
                }

                components = &transformed_components;
                values = &transformed_values;
            }
            if (debug) {
                auto elapsed = std::chrono::high_resolution_clock::now() - transform_start;
                std::cout << "Time to transform:" << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " microseconds" << std::endl;
            }
            
            for (size_t i = 0; i < summaries_.size(); ++i) {
                const std::vector<std::pair<uint16_t, float>>& summary = summaries_[i].second;
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

            if (debug) {
                auto elapsed = std::chrono::high_resolution_clock::now() - start;
                std::cout << "Time to get distances:" << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " microsecs" << std::endl;
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
                if (visited.find(prev_offset) == visited.end() && forward_index.is_alive_at_offset(prev_offset)) {
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
            
            // Process the last posting if not already visited (and still alive)
            if (visited.find(prev_offset) == visited.end() && forward_index.is_alive_at_offset(prev_offset)) {
                auto [v_components, v_values] = forward_index.get_with_offset(prev_offset, prev_len);
                
                float distance = distances::dot_product_dense_sparse(query, v_components, v_values);
                
                visited.insert(prev_offset);
                heap.push_with_id(-1.0f * distance, prev_offset);
            }
        }

        // Blocking strategies
        static std::vector<size_t> fixed_size_blocking(std::vector<size_t>& posting_list, size_t num_blocks, const SparseDatasetMut<float>& dataset) 
        {
            if (posting_list.empty()) return {0};
            // `num_blocks` (= --block-size) is the number of centroids/blocks for
            // this posting list, bounded by the list length. Paper-faithful blocking
            // uses ~0.1*L (=400 at the n_postings=4000 cap); a smaller value makes the
            // build much faster but coarsens blocks, which hurts query latency/recall.
            // Blocking cost is O(L * num_blocks * nnz), so this is the main build-time
            // vs query-quality knob. Set it via build_inverted_index --block-size.
            num_blocks = std::min(num_blocks, posting_list.size());
            std::vector<size_t> reordered_posting_list;
            reordered_posting_list.reserve(posting_list.size());       
            
            std::vector<size_t> block_offsets;
            block_offsets.reserve(num_blocks + 1);

            // Randomly select cluster representatives
            std::vector<size_t> shuffled = posting_list;
            {
                std::random_device rd;
                std::mt19937 rng(rd());
                std::shuffle(shuffled.begin(), shuffled.end(), rng);
            }
            std::vector<size_t> centroid_docs(shuffled.begin(), shuffled.begin() + num_blocks);

            // Assign each x to argmax_j <x, μ(j)>
            std::vector<std::pair<size_t, size_t>> clustering_results;
            clustering_results.resize(posting_list.size());

            // Helper: dot product between two sorted sparse vectors given as raw
            // pointers (zero-copy views) so the hot loop allocates nothing.
            auto sparse_dot = [](const uint16_t* a_idx, const float* a_val, size_t a_n,
                                 const uint16_t* b_idx, const float* b_val, size_t b_n) -> float
            {
                size_t i = 0, j = 0;
                float acc = 0.0f;
                while (i < a_n && j < b_n) {
                    if (a_idx[i] == b_idx[j]) {
                        acc += a_val[i] * b_val[j];
                        ++i; ++j;
                    } else if (a_idx[i] < b_idx[j]) {
                        ++i;
                    } else {
                        ++j;
                    }
                }
                return acc;
            };

            // Pre-fetch the centroid views ONCE (zero-copy into the dataset arrays).
            // The earlier version called dataset.get() for every (doc, centroid) pair,
            // allocating and copying two vectors each time, which made blocking
            // allocator-bound. Views remove that entirely.
            std::vector<SparseDatasetMut<float>::VectorView> centroid_views;
            centroid_views.reserve(centroid_docs.size());
            for (size_t j = 0; j < centroid_docs.size(); ++j) {
                centroid_views.push_back(dataset.get_view(centroid_docs[j]));
            }

            // No inner `#pragma omp parallel for` here: this runs inside the outer
            // parallel-for over posting lists in MyInvertedIndex::build. A nested
            // region (OpenMP nesting is off by default) serialized the largest
            // lists onto a single thread, creating a long single-threaded tail.
            for (size_t i = 0; i < posting_list.size(); ++i) {
                size_t doc_id = posting_list[i];
                assert(doc_id < dataset.len());

                const auto xv = dataset.get_view(doc_id);

                size_t best_j = 0;
                float best_score = -std::numeric_limits<float>::infinity();

                for (size_t j = 0; j < centroid_views.size(); ++j) {
                    const auto& mu = centroid_views[j];
                    float s = sparse_dot(xv.components, xv.values, xv.len,
                                         mu.components, mu.values, mu.len);
                    if (s > best_score) {
                        best_score = s;
                        best_j = j;
                    }
                }
                clustering_results[i] = {best_j, doc_id};
            }

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

            // // Create block offsets for fixed-size blocks
            // std::vector<size_t> block_offsets;
            
            // for (size_t i = 0; i < posting_list.size(); i += block_size) {
            //     block_offsets.push_back(i);
            // }
            
            // // Add the final offset
            // block_offsets.push_back(posting_list.size());
            
            // return block_offsets;
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
            
            // Begin approximate knn neighbors
            // // Helper function to compute dot between sparse vectors
            // auto sparse_dot = [&](const auto& a_idx, const auto& a_val,
            //         const auto& b_idx, const auto& b_val) -> float 
            // {
            //     size_t i = 0, j = 0;
            //     float acc = 0.0;
            //     while (i < a_idx.size() && j < b_idx.size()) {
            //         if (a_idx[i] == b_idx[j]) {
            //             acc += static_cast<float>(a_val[i]) * static_cast<float>(b_val[j]);
            //             ++i; ++j;
            //         } else if (a_idx[i] < b_idx[j]) {
            //             ++i;
            //         } else {
            //             ++j;
            //         }
            //     }
            //     return acc;
            // };

            // // Randomly select centroids
            // std::vector<size_t> shuffled = posting_list;
            // {
            //     std::random_device rd;
            //     std::mt19937 rng(rd());
            //     std::shuffle(shuffled.begin(), shuffled.end(), rng);
            // }
            // std::vector<size_t> centroid_docs(shuffled.begin(), shuffled.begin() + n_centroids);

            // // assign each x to argmax_j <x, μ(j)>
            // std::vector<std::pair<size_t, size_t>> clustering_results;
            // clustering_results.reserve(posting_list.size());

            // for (size_t doc_id : posting_list) {
            //     const auto& x = dataset.get(doc_id);
            //     const auto& x_idx = x.first;
            //     const auto& x_val = x.second;

            //     size_t best_j = 0;
            //     double best_score = -std::numeric_limits<double>::infinity();

            //     for (size_t j = 0; j < centroid_docs.size(); ++j) {
            //         const auto& mu = dataset.get(centroid_docs[j]);
            //         const auto& mu_idx = mu.first;
            //         const auto& mu_val = mu.second;

            //         double s = sparse_dot(x_idx, x_val, mu_idx, mu_val);
            //         if (s > best_score) {
            //             best_score = s;
            //             best_j = j;
            //         }
            //     }
            //     clustering_results.emplace_back(best_j, doc_id);
            // }
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
            archive(packed_postings_, block_offsets_, summaries_, block_radii_);
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