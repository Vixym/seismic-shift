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
#include "sparse_jlt.h"

namespace seismic
{

    // Forward declaration of template class
    template <typename T>
    class InvertedIndex;

    const size_t THRESHOLD_BINARY_SEARCH = 10;

    /**
     * Clustering algorithm types for the inverted index.
     */
    class ClusteringAlgorithm
    {
    public:
        enum class Type
        {
            RandomKmeans,
            RandomKmeansInvertedIndex,
            RandomKmeansInvertedIndexApprox
        };

        // Default constructor - RandomKmeansInvertedIndexApprox with doc_cut = 15
        ClusteringAlgorithm()
            : type(Type::RandomKmeansInvertedIndexApprox),
              pruning_factor(0.0f),
              doc_cut(15) {}

        // RandomKmeans constructor
        static ClusteringAlgorithm random_kmeans()
        {
            ClusteringAlgorithm algorithm;
            algorithm.type = Type::RandomKmeans;
            return algorithm;
        }

        // RandomKmeansInvertedIndex constructor
        static ClusteringAlgorithm random_kmeans_inverted_index(float pruning_factor, size_t doc_cut)
        {
            ClusteringAlgorithm algorithm;
            algorithm.type = Type::RandomKmeansInvertedIndex;
            algorithm.pruning_factor = pruning_factor;
            algorithm.doc_cut = doc_cut;
            return algorithm;
        }

        // RandomKmeansInvertedIndexApprox constructor
        static ClusteringAlgorithm random_kmeans_inverted_index_approx(size_t doc_cut)
        {
            ClusteringAlgorithm algorithm;
            algorithm.type = Type::RandomKmeansInvertedIndexApprox;
            algorithm.doc_cut = doc_cut;
            return algorithm;
        }

        // Getters
        Type get_type() const { return type; }
        float get_pruning_factor() const { return pruning_factor; }
        size_t get_doc_cut() const { return doc_cut; }

        // Serialization
        template <class Archive>
        void serialize(Archive &archive)
        {
            archive(type, pruning_factor, doc_cut);
        }

    private:
        Type type;
        float pruning_factor;
        size_t doc_cut;
    };

    /**
     * Represents the possible choices for the strategy used to prune the posting
     * lists at building time.
     * There are the following possible strategies:
     * - `Fixed  { n_postings: usize }`: Every posting list is pruned by taking its top-`n_postings`
     * - `GlobalThreshold { n_postings: usize, max_fraction: f32 }`: We globally select a threshold and
     * we prune all the postings with smaller score. The threshold is chosen so that every posting list has
     * `n_postings` on average. We limit the number of postings per list to `max_fraction*n_postings`.
     */
    class PruningStrategy
    {
    public:
        enum class Type
        {
            FixedSize,
            GlobalThreshold
        };

        // Default constructor - FixedSize with n_postings = 3500
        PruningStrategy() : type(Type::FixedSize), n_postings(3500), max_fraction(0.0f) {}

        // FixedSize constructor
        static PruningStrategy fixed_size(size_t n_postings)
        {
            PruningStrategy strategy;
            strategy.type = Type::FixedSize;
            strategy.n_postings = n_postings;
            return strategy;
        }

        // GlobalThreshold constructor
        static PruningStrategy global_threshold(size_t n_postings, float max_fraction)
        {
            PruningStrategy strategy;
            strategy.type = Type::GlobalThreshold;
            strategy.n_postings = n_postings;
            strategy.max_fraction = max_fraction;
            return strategy;
        }

        // Getters
        Type get_type() const { return type; }
        size_t get_n_postings() const { return n_postings; }
        float get_max_fraction() const { return max_fraction; }

        // Serialization
        template <class Archive>
        void serialize(Archive &archive)
        {
            archive(type, n_postings, max_fraction);
        }

    private:
        Type type;
        size_t n_postings;
        float max_fraction;
    };

    /**
     * Represents the possible choices for the strategy used to block the posting lists.
     */
    class BlockingStrategy
    {
    public:
        enum class Type
        {
            FixedSize,
            RandomKmeans
        };

        // Default constructor - RandomKmeans with default parameters
        BlockingStrategy()
            : type(Type::RandomKmeans),
              block_size(0),
              centroid_fraction(0.1f),
              min_cluster_size(2),
              clustering_algorithm(ClusteringAlgorithm()) {}

        // FixedSize constructor
        static BlockingStrategy fixed_size(size_t block_size)
        {
            BlockingStrategy strategy;
            strategy.type = Type::FixedSize;
            strategy.block_size = block_size;
            return strategy;
        }

        // RandomKmeans constructor
        static BlockingStrategy random_kmeans(float centroid_fraction, size_t min_cluster_size,
                                              const ClusteringAlgorithm &algorithm)
        {
            BlockingStrategy strategy;
            strategy.type = Type::RandomKmeans;
            strategy.centroid_fraction = centroid_fraction;
            strategy.min_cluster_size = min_cluster_size;
            strategy.clustering_algorithm = algorithm;
            return strategy;
        }

        // Getters
        Type get_type() const { return type; }
        size_t get_block_size() const { return block_size; }
        float get_centroid_fraction() const { return centroid_fraction; }
        size_t get_min_cluster_size() const { return min_cluster_size; }
        const ClusteringAlgorithm &get_clustering_algorithm() const { return clustering_algorithm; }

        // Serialization
        template <class Archive>
        void serialize(Archive &archive)
        {
            archive(type, block_size, centroid_fraction, min_cluster_size, clustering_algorithm);
        }

    private:
        Type type;
        size_t block_size;
        float centroid_fraction;
        size_t min_cluster_size;
        ClusteringAlgorithm clustering_algorithm;
    };

    /**
     * Represents the possible choices for the strategy used to summarize blocks.
     */
    class SummarizationStrategy
    {
    public:
        enum class Type
        {
            FixedSize,
            EnergyPreserving
        };

        // Default constructor - EnergyPreserving with summary_energy = 0.4
        SummarizationStrategy() : type(Type::EnergyPreserving), n_components(0), summary_energy(0.4f) {}

        // FixedSize constructor
        static SummarizationStrategy fixed_size(size_t n_components)
        {
            SummarizationStrategy strategy;
            strategy.type = Type::FixedSize;
            strategy.n_components = n_components;
            return strategy;
        }

        // EnergyPreserving constructor
        static SummarizationStrategy energy_preserving(float summary_energy)
        {
            SummarizationStrategy strategy;
            strategy.type = Type::EnergyPreserving;
            strategy.summary_energy = summary_energy;
            return strategy;
        }

        // Getters
        Type get_type() const { return type; }
        size_t get_n_components() const { return n_components; }
        float get_summary_energy() const { return summary_energy; }

        // Serialization
        template <class Archive>
        void serialize(Archive &archive)
        {
            archive(type, n_components, summary_energy);
        }

    private:
        Type type;
        size_t n_components;
        float summary_energy;
    };

    /**
     * KNN configuration class for the inverted index.
     */
    class KnnConfiguration
    {
    private:
        size_t nknn;
        std::optional<std::string> knn_path;

    public:
        // Default constructor
        KnnConfiguration() : nknn(0), knn_path(std::nullopt) {}

        // Constructor with parameters
        KnnConfiguration(size_t nknn, std::optional<std::string> knn_path = std::nullopt)
            : nknn(nknn), knn_path(std::move(knn_path)) {}

        // Getters
        size_t get_nknn() const { return nknn; }
        const std::optional<std::string> &get_knn_path() const { return knn_path; }

        // Serialization
        template <class Archive>
        void serialize(Archive &archive)
        {
            archive(nknn, knn_path);
        }
    };

    /**
     * Configuration for the inverted index.
     * Contains parameters for pruning, blocking, summarization, and kNN.
     */
    class Configuration
    {
    public:
        // Default constructor
        Configuration() : pruning(PruningStrategy()), blocking(BlockingStrategy()),
                          summarization(SummarizationStrategy()), knn_config(KnnConfiguration()),
                          jlt_target_dim(0), jlt_sparsity(-1) {}

        // Builder pattern methods
        Configuration &pruning_strategy(const PruningStrategy &pruning);
        Configuration &blocking_strategy(const BlockingStrategy &blocking);
        Configuration &summarization_strategy(const SummarizationStrategy &summarization);
        Configuration &knn(const KnnConfiguration &knn_config);
        Configuration &batched_indexing(const std::optional<size_t> &batched_indexing);
        Configuration &jlt(size_t target_dim, int sparsity = -1)
        {
            jlt_target_dim = target_dim;
            jlt_sparsity = sparsity;
            return *this;
        }

        // Getters
        const PruningStrategy &get_pruning() const { return pruning; }
        const BlockingStrategy &get_blocking() const { return blocking; }
        const SummarizationStrategy &get_summarization() const { return summarization; }
        const KnnConfiguration &get_knn_config() const { return knn_config; }
        const std::optional<size_t> &get_batched_indexing() const { return batched_indexing_size; }
        size_t get_jlt_target_dim() const { return jlt_target_dim; }
        int get_jlt_sparsity() const { return jlt_sparsity; }

        // Serialization
        template <class Archive>
        void serialize(Archive &archive)
        {
            archive(pruning, blocking, summarization, knn_config, batched_indexing_size, jlt_target_dim, jlt_sparsity);
        }

    private:
        PruningStrategy pruning;
        BlockingStrategy blocking;
        SummarizationStrategy summarization;
        KnnConfiguration knn_config;
        std::optional<size_t> batched_indexing_size;
        size_t jlt_target_dim;
        int jlt_sparsity;
    };

    /**
     * Instead of string doc_ids we store their offsets in the forward_index and the lengths of the vectors.
     * This allows us to save the random accesses that would be needed to access exactly these values from the
     * forward index. The values of each doc are packed into a single u64 in `packed_postings`. We use 48 bits
     * for the offset and 16 bits for the length. This choice limits the size of the dataset to be 1<<48-1.
     * We use the forward index to convert the offsets of the top-k back to the id of the corresponding documents.
     */
    class PostingList : public SpaceUsage
    {
    private:
        std::vector<uint64_t> packed_postings;
        std::vector<size_t> block_offsets;
        std::vector<std::vector<float>> centroids; // Replace QuantizedSummary with centroids

    public:
        // Default constructor
        PostingList() = default;

        // Constructor with parameters
        PostingList(std::vector<uint64_t> packed_postings, std::vector<size_t> block_offsets,
                    std::vector<std::vector<float>> centroids)
            : packed_postings(std::move(packed_postings)),
              block_offsets(std::move(block_offsets)),
              centroids(std::move(centroids)) {}

        // Pack offset and length into a single uint64_t
        static uint64_t pack_offset_len(size_t offset, size_t len)
        {
            return ((static_cast<uint64_t>(offset) << 16) | (len & 0xFFFF));
        }

        // Unpack offset and length from a single uint64_t
        static std::pair<size_t, size_t> unpack_offset_len(uint64_t pack)
        {
            return {static_cast<size_t>(pack >> 16), static_cast<size_t>(pack & 0xFFFF)};
        }

        // Space usage calculation
        size_t space_usage_byte() const override
        {
            size_t centroid_size = 0;
            for (const auto &centroid : centroids)
            {
                centroid_size += centroid.size() * sizeof(float);
            }
            return packed_postings.size() * sizeof(uint64_t) +
                   block_offsets.size() * sizeof(size_t) +
                   centroid_size;
        }

        // Search implementation
        template <typename T>
        void search(
            const std::vector<float> &query,
            const std::vector<uint16_t> &query_components,
            const std::vector<float> &query_values,
            size_t k,
            float heap_factor,
            utils::HeapFaiss &heap,
            std::unordered_set<size_t> &visited,
            const SparseDataset<T> &forward_index,
            bool sort_summaries) const
        {
            std::vector<std::vector<uint64_t>> blocks_to_evaluate;

            // Get distances between query and centroids
            std::vector<std::pair<size_t, float>> indexed_dots;
            indexed_dots.reserve(centroids.size());

            for (size_t i = 0; i < centroids.size(); ++i)
            {
                float dot = 0.0f;
                for (size_t j = 0; j < query.size(); ++j)
                {
                    dot += query[j] * centroids[i][j];
                }
                indexed_dots.emplace_back(i, dot);
            }

            // Sort centroids by dot product w.r.t. to the query if requested
            if (sort_summaries)
            {
                std::sort(indexed_dots.begin(), indexed_dots.end(),
                          [](const auto &a, const auto &b)
                          { return a.second > b.second; });
            }

            for (const auto &[block_id, dot] : indexed_dots)
            {
                // Skip blocks that cannot contribute to the top-k
                if (heap.len() == k && dot < -heap_factor * heap.topk().front().first)
                {
                    continue;
                }

                // Get the block of postings
                std::vector<uint64_t> packed_posting_block(
                    packed_postings.begin() + block_offsets[block_id],
                    packed_postings.begin() + block_offsets[block_id + 1]);

                // If we have accumulated blocks, evaluate them first
                if (blocks_to_evaluate.size() == 1)
                {
                    for (const auto &cur_packed_posting : blocks_to_evaluate)
                    {
                        evaluate_posting_block(
                            query,
                            query_components,
                            query_values,
                            cur_packed_posting,
                            heap,
                            visited,
                            forward_index);
                    }
                    blocks_to_evaluate.clear();
                }

                // Prefetch the block
                for (size_t i = 0; i < packed_posting_block.size(); i += 8)
                {
                    if (i + 8 < packed_posting_block.size())
                    {
                        __builtin_prefetch(&packed_posting_block[i + 8], 0, 1);
                    }
                }

                blocks_to_evaluate.push_back(std::move(packed_posting_block));
            }

            // Evaluate any remaining blocks
            for (const auto &cur_packed_posting : blocks_to_evaluate)
            {
                evaluate_posting_block(
                    query,
                    query_components,
                    query_values,
                    cur_packed_posting,
                    heap,
                    visited,
                    forward_index);
            }
        }

        // Build a posting list from a dataset and a list of postings
        template <typename T>
        static PostingList build(const std::vector<size_t> &posting_list, const SparseDataset<T> &dataset, const Configuration &config)
        {
            // Create block offsets
            std::vector<size_t> block_offsets;
            const auto &blocking = config.get_blocking();

            if (blocking.get_type() == BlockingStrategy::Type::FixedSize)
            {
                block_offsets = fixed_size_blocking(posting_list, blocking.get_block_size());
            }
            else if (blocking.get_type() == BlockingStrategy::Type::RandomKmeans)
            {
                block_offsets = blocking_with_random_kmeans(
                    posting_list,
                    blocking.get_centroid_fraction(),
                    blocking.get_min_cluster_size(),
                    dataset,
                    blocking.get_clustering_algorithm());
            }

            // Create centroids for each block
            std::vector<std::vector<float>> block_centroids;
            block_centroids.reserve(block_offsets.size() - 1);

            for (size_t i = 0; i < block_offsets.size() - 1; ++i)
            {
                std::vector<size_t> block(
                    posting_list.begin() + block_offsets[i],
                    posting_list.begin() + block_offsets[i + 1]);

                // Compute centroid for the block
                std::vector<float> centroid(dataset.dim(), 0.0f);
                for (size_t doc_id : block)
                {
                    auto [components, values] = dataset.get(doc_id);
                    for (size_t j = 0; j < components.size(); ++j)
                    {
                        centroid[components[j]] += values[j];
                    }
                }

                // Normalize centroid
                float norm = 0.0f;
                for (float val : centroid)
                {
                    norm += val * val;
                }
                norm = std::sqrt(norm);
                if (norm > 0)
                {
                    for (float &val : centroid)
                    {
                        val /= norm;
                    }
                }

                block_centroids.push_back(std::move(centroid));
            }

            // Pack document offsets and lengths
            std::vector<uint64_t> packed_postings;
            packed_postings.reserve(posting_list.size());

            for (size_t i = 0; i < posting_list.size(); ++i)
            {
                auto [offset, len] = dataset.id_to_offset_len(posting_list[i]);
                packed_postings.push_back(pack_offset_len(offset, len));
            }

            return PostingList(std::move(packed_postings), std::move(block_offsets), std::move(block_centroids));
        }

        // Evaluate a block of postings
        template <typename T>
        void evaluate_posting_block(
            const std::vector<float> &query,
            const std::vector<uint16_t> &query_components,
            const std::vector<float> &query_values,
            const std::vector<uint64_t> &packed_posting_block,
            utils::HeapFaiss &heap,
            std::unordered_set<size_t> &visited,
            const SparseDataset<T> &forward_index) const
        {
            if (packed_posting_block.empty())
            {
                return;
            }

            // Process the first posting
            auto [prev_offset, prev_len] = unpack_offset_len(packed_posting_block[0]);

            // Process the middle postings
            for (size_t i = 1; i < packed_posting_block.size(); ++i)
            {
                auto [offset, len] = unpack_offset_len(packed_posting_block[i]);

                // Process the current vector if not already visited
                if (visited.find(prev_offset) == visited.end())
                {
                    auto [v_components, v_values] = forward_index.get_with_offset(prev_offset, prev_len);

                    float distance;
                    if (jlt_.has_value())
                    {
                        // Transform both query and document vectors using JLT
                        auto transformed_query = jlt_->transform_sparse(query_components, query_values);
                        auto transformed_doc = jlt_->transform_sparse(v_components, v_values);

                        // Compute dot product in the reduced space
                        distance = 0.0f;
                        for (size_t j = 0; j < transformed_query.size(); ++j)
                        {
                            distance += transformed_query[j] * transformed_doc[j];
                        }
                    }
                    else if (query_components.size() < THRESHOLD_BINARY_SEARCH)
                    {
                        distance = distances::dot_product_with_merge(
                            query_components, query_values, v_components, v_values);
                    }
                    else
                    {
                        distance = distances::dot_product_dense_sparse(query, v_components, v_values);
                    }

                    visited.insert(prev_offset);
                    heap.push_with_id(-1.0f * distance, prev_offset);
                }

                prev_offset = offset;
                prev_len = len;
            }

            // Process the last posting if not already visited
            if (visited.find(prev_offset) == visited.end())
            {
                auto [v_components, v_values] = forward_index.get_with_offset(prev_offset, prev_len);

                float distance;
                if (jlt_.has_value())
                {
                    // Transform both query and document vectors using JLT
                    auto transformed_query = jlt_->transform_sparse(query_components, query_values);
                    auto transformed_doc = jlt_->transform_sparse(v_components, v_values);

                    // Compute dot product in the reduced space
                    distance = 0.0f;
                    for (size_t j = 0; j < transformed_query.size(); ++j)
                    {
                        distance += transformed_query[j] * transformed_doc[j];
                    }
                }
                else
                {
                    distance = distances::dot_product_dense_sparse(query, v_components, v_values);
                }

                visited.insert(prev_offset);
                heap.push_with_id(-1.0f * distance, prev_offset);
            }
        }

        // Blocking strategies
        static std::vector<size_t> fixed_size_blocking(const std::vector<size_t> &posting_list, size_t block_size)
        {
            // Create block offsets for fixed-size blocks
            std::vector<size_t> block_offsets;

            for (size_t i = 0; i < posting_list.size(); i += block_size)
            {
                block_offsets.push_back(i);
            }

            // Add the final offset
            block_offsets.push_back(posting_list.size());

            return block_offsets;
        }

        template <typename T>
        static std::vector<size_t> blocking_with_random_kmeans(
            std::vector<size_t> &posting_list,
            float centroid_fraction,
            size_t min_cluster_size,
            const SparseDataset<T> &dataset,
            const ClusteringAlgorithm &clustering_algorithm)
        {

            if (posting_list.empty())
            {
                return {};
            }

            // Calculate number of centroids
            size_t n_centroids = std::max(
                static_cast<size_t>(centroid_fraction * posting_list.size()),
                static_cast<size_t>(1));

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

            for (size_t i = 0; i < posting_list.size(); ++i)
            {
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
                      [](const auto &a, const auto &b)
                      { return a.first < b.first; });

            size_t current_centroid = clustering_results[0].first;
            for (const auto &[centroid_id, doc_id] : clustering_results)
            {
                if (centroid_id != current_centroid)
                {
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
            const SparseDataset<T> &dataset,
            const std::vector<size_t> &block,
            size_t n_components)
        {

            std::unordered_map<uint16_t, T> hash;

            // For each document in the block
            for (size_t doc_id : block)
            {
                // For each component in the document, store the largest value seen so far
                auto [components, values] = dataset.get(doc_id);

                for (size_t i = 0; i < components.size(); ++i)
                {
                    uint16_t component_id = components[i];
                    T value = values[i];
                    auto it = hash.find(component_id);
                    if (it == hash.end() || it->second < value)
                    {
                        hash[component_id] = value;
                    }
                }
            }

            // Convert to vector of pairs for sorting
            std::vector<std::pair<uint16_t, T>> components_values(hash.begin(), hash.end());

            // Sort by decreasing values
            std::sort(components_values.begin(), components_values.end(),
                      [](const auto &a, const auto &b)
                      { return a.second > b.second; });

            // Take only up to n_components
            if (components_values.size() > n_components)
            {
                components_values.resize(n_components);
            }

            // Sort by component ID for binary search
            std::sort(components_values.begin(), components_values.end(),
                      [](const auto &a, const auto &b)
                      { return a.first < b.first; });

            // Extract components and values
            std::vector<uint16_t> components;
            std::vector<T> values;

            components.reserve(components_values.size());
            values.reserve(components_values.size());

            for (const auto &[component_id, value] : components_values)
            {
                components.push_back(component_id);
                values.push_back(value);
            }

            return {components, values};
        }

        template <typename T>
        static std::pair<std::vector<uint16_t>, std::vector<T>> energy_preserving_summary(
            const SparseDataset<T> &dataset,
            const std::vector<size_t> &block,
            float fraction)
        {

            std::unordered_map<uint16_t, T> hash;

            // For each document in the block
            for (size_t doc_id : block)
            {
                // For each component in the document, store the largest value seen so far
                auto [components, values] = dataset.get(doc_id);

                for (size_t i = 0; i < components.size(); ++i)
                {
                    uint16_t component_id = components[i];
                    T value = values[i];
                    auto it = hash.find(component_id);
                    if (it == hash.end() || it->second < value)
                    {
                        hash[component_id] = value;
                    }
                }
            }

            // Convert to vector of pairs for sorting
            std::vector<std::pair<uint16_t, T>> components_values(hash.begin(), hash.end());

            // Sort by decreasing values
            std::sort(components_values.begin(), components_values.end(),
                      [](const auto &a, const auto &b)
                      { return a.second > b.second; });

            // Calculate total energy
            float total_sum = 0.0f;
            for (const auto &[_, value] : components_values)
            {
                total_sum += static_cast<float>(value);
            }

            // Select components that preserve the desired energy fraction
            std::vector<uint16_t> term_ids;
            std::vector<T> values;

            float acc = 0.0f;
            for (const auto &[tid, value] : components_values)
            {
                acc += static_cast<float>(value);
                term_ids.push_back(tid);
                values.push_back(value);

                if ((acc / total_sum) > fraction)
                {
                    break;
                }
            }

            // Sort term IDs
            std::sort(term_ids.begin(), term_ids.end());

            // Reorder values to match sorted term IDs
            std::vector<T> sorted_values;
            sorted_values.reserve(term_ids.size());

            for (uint16_t tid : term_ids)
            {
                sorted_values.push_back(hash[tid]);
            }

            return {term_ids, sorted_values};
        }

        // Getters
        const std::vector<uint64_t> &packed_postings() const { return packed_postings; }
        const std::vector<size_t> &block_offsets() const { return block_offsets; }
        const std::vector<std::vector<float>> &get_centroids() const { return centroids; }

        template <class Archive>
        void serialize(Archive &archive)
        {
            archive(packed_postings, block_offsets, centroids);
        }
    };

    /**
     * KNN graph implementation for the inverted index.
     */
    class Knn
    {
        // TODO implement space usage
    private:
        size_t n_vecs;
        size_t d;
        BitVector neighbours;
        size_t nbits;

    public:
        static constexpr size_t KNN_QUERY_CUT = 10;
        static constexpr float KNN_HEAP_FACTOR = 0.7f;

        // Default constructor
        Knn() : n_vecs(0), d(0), nbits(0) {}

        // Constructor with parameters
        Knn(size_t n_vecs, size_t d, BitVector neighbours, size_t nbits)
            : n_vecs(n_vecs), d(d), neighbours(std::move(neighbours)), nbits(nbits) {}

        // Build a KNN graph from an inverted index
        template <typename T>
        static Knn new_from_index(const InvertedIndex<T> &index, size_t d)
        {
            const size_t n_vecs = index.len();
            const size_t nbits = static_cast<size_t>(std::ceil(std::log2(static_cast<double>(n_vecs))));
            BitVectorMut neighbours(n_vecs * nbits * d);

            // For each document, find its k nearest neighbors
            for (size_t i = 0; i < n_vecs; ++i)
            {
                // Get the document vector
                auto [components, values] = index.dataset().get(i);

                // Search for similar documents
                std::vector<std::pair<float, size_t>> results;
                if (index.jlt_.has_value())
                {
                    // Transform the document vector using JLT
                    auto transformed = index.jlt_->transform_sparse(components, values);
                    std::vector<float> transformed_query(transformed.begin(), transformed.end());
                    results = index.search(components, values, d + 1, components.size(), 0.7f, 0, false);
                }
                else
                {
                    results = index.search(components, values, d + 1, components.size(), 0.7f, 0, false);
                }

                // Store the neighbors
                for (size_t j = 0; j < d; ++j)
                {
                    size_t bit_offset = i * d * nbits + j * nbits;
                    neighbours.append_bits(results[j + 1].second, nbits);
                }
            }

            return Knn(n_vecs, d, neighbours.to_immutable(), nbits);
        }

        // Load a KNN graph from a serialized file
        static Knn new_from_serialized(const std::string &input_file, size_t d)
        {
            // Open the input file
            std::ifstream input(input_file, std::ios::binary);
            if (!input)
            {
                throw std::runtime_error("Failed to open input file: " + input_file);
            }

            // Read the number of vectors
            size_t n_vecs;
            input.read(reinterpret_cast<char *>(&n_vecs), sizeof(n_vecs));

            // Read the number of bits per vector
            size_t nbits;
            input.read(reinterpret_cast<char *>(&nbits), sizeof(nbits));

            // Read the bit vector data
            std::vector<uint64_t> data;
            size_t data_size = (n_vecs * d * nbits + 63) / 64;
            data.resize(data_size);
            input.read(reinterpret_cast<char *>(data.data()), data_size * sizeof(uint64_t));

            // Create the bit vector
            BitVectorMut neighbours(n_vecs * nbits * d);
            for (uint64_t x : data)
            {
                neighbours.append_bits(x, 64);
            }

            return Knn(n_vecs, d, neighbours.to_immutable(), nbits);
        }

        // Serialize the KNN graph to a file
        void serialize(const std::string &output_file) const
        {
            // Open the output file
            std::ofstream output(output_file, std::ios::binary);
            if (!output)
            {
                throw std::runtime_error("Failed to open output file: " + output_file);
            }

            // Write the number of vectors
            output.write(reinterpret_cast<const char *>(&n_vecs), sizeof(n_vecs));

            // Write the number of bits per vector
            output.write(reinterpret_cast<const char *>(&nbits), sizeof(nbits));

            // Write the bit vector data
            size_t data_size = (n_vecs * d * nbits + 63) / 64;
            std::vector<uint64_t> data(data_size);
            for (size_t i = 0; i < data_size; ++i)
            {
                data[i] = neighbours.get_bits_unchecked(i * 64, 64);
            }
            output.write(reinterpret_cast<const char *>(data.data()), data_size * sizeof(uint64_t));
        }

        // Compress a sequence of document IDs into a bit vector
        static std::pair<BitVector, size_t> compress_into_bitvector(
            const std::vector<uint64_t> &data, size_t n_vecs, [[maybe_unused]] size_t d)
        {

            // Calculate number of bits needed to represent n_vecs
            size_t nbits = static_cast<size_t>(std::ceil(std::log2(static_cast<double>(n_vecs))));

            // Create a bit vector to store the compressed data
            BitVectorMut neighbours(data.size() * nbits);

            // Append each document ID to the bit vector
            for (uint64_t x : data)
            {
                neighbours.append_bits(x, nbits);
            }

            return {neighbours.into(), nbits};
        }

        // Refine the search results using the KNN graph
        template <typename T>
        void refine(const std::vector<float> &query,
                    utils::HeapFaiss &heap,
                    std::unordered_set<size_t> &visited,
                    const SparseDataset<T> &forward_index,
                    size_t n_knn) const
        {
            // Use the minimum of the requested number of neighbors and the available number
            size_t n_knn_to_use = std::min(d, n_knn);

            // Get the current top-k results
            auto neighbours_results = heap.topk();

            // For each result, explore its neighbors
            for (const auto &[_distance, offset] : neighbours_results)
            {
                size_t id = forward_index.offset_to_id(offset);

                // For each neighbor of the current result
                for (size_t i = 0; i < n_knn_to_use; ++i)
                {
                    size_t bit_offset = id * d * nbits + i * nbits;
                    uint64_t neighbour = neighbours.get_bits_unchecked(bit_offset, nbits);

                    // Get the offset and length of the neighbor's vector
                    size_t offset = forward_index.vector_offset(static_cast<size_t>(neighbour));
                    size_t len = forward_index.vector_len(static_cast<size_t>(neighbour));

                    // If not already visited, compute the distance and add to the heap
                    if (visited.find(offset) == visited.end())
                    {
                        auto [v_components, v_values] = forward_index.get_with_offset(offset, len);

                        // If the query is transformed (JLT), we need to transform the document vector too
                        float distance;
                        if (query.size() != forward_index.dim())
                        {
                            // Query is transformed, so we need to transform the document vector
                            auto transformed_doc = SparseJLT(forward_index.dim(), query.size(), -1)
                                                       .transform_sparse(v_components, v_values);
                            // Compute dot product in reduced space
                            distance = 0.0f;
                            for (size_t j = 0; j < query.size(); ++j)
                            {
                                distance += query[j] * transformed_doc[j];
                            }
                        }
                        else
                        {
                            // Query is not transformed, use normal dot product
                            distance = distances::dot_product_dense_sparse(query, v_components, v_values);
                        }

                        visited.insert(offset);
                        heap.push_with_id(-1.0f * distance, offset);
                    }
                }
            }
        }

        // Getters
        size_t get_n_vecs() const { return n_vecs; }
        size_t get_d() const { return d; }
        const BitVector &get_neighbours() const { return neighbours; }
        size_t get_nbits() const { return nbits; }

        // Space usage calculation
        size_t space_usage_byte() const;

        template <class Archive>
        void serialize(Archive &archive)
        {
            archive(n_vecs, d, neighbours, nbits);
        }

        std::vector<size_t> get_neighbors(size_t id) const
        {
            std::vector<size_t> neighbors;
            neighbors.reserve(d);

            for (size_t i = 0; i < d; ++i)
            {
                size_t bit_offset = id * d * nbits + i * nbits;
                uint64_t neighbor = neighbours.get_bits_unchecked(bit_offset, nbits);
                neighbors.push_back(static_cast<size_t>(neighbor));
            }

            return neighbors;
        }
    };

    /**
     * The main inverted index class.
     */
    template <typename T>
    class InvertedIndex : public SpaceUsage
    {
    private:
        SparseDataset<T> forward_index_;
        std::vector<PostingList> posting_lists_;
        Configuration config_;
        std::optional<Knn> knn_;
        std::optional<SparseJLT> jlt_;

    public:
        // Default constructor
        InvertedIndex() = default;

        // Constructor with parameters
        InvertedIndex(SparseDataset<T> forward_index, std::vector<PostingList> posting_lists,
                      Configuration config, std::optional<Knn> knn = std::nullopt,
                      std::optional<SparseJLT> jlt = std::nullopt)
            : forward_index_(std::move(forward_index)),
              posting_lists_(std::move(posting_lists)),
              config_(std::move(config)),
              knn_(std::move(knn)),
              jlt_(std::move(jlt)) {}

        // Search the index
        std::vector<std::pair<float, size_t>> search(
            const std::vector<uint16_t> &query_components,
            const std::vector<float> &query_values,
            size_t k,
            size_t query_cut,
            float heap_factor,
            size_t n_knn,
            bool first_sorted) const
        {
            // Create a dense query vector
            std::vector<float> query(dim(), 0.0f);
            for (size_t i = 0; i < query_components.size(); ++i)
            {
                query[query_components[i]] = query_values[i];
            }

            utils::HeapFaiss heap(k);
            std::unordered_set<size_t> visited;
            visited.reserve(query_cut * 5000); // 5000 should be n_postings

            // Sort query terms by score and evaluate the posting list only for the top ones
            std::vector<std::pair<uint16_t, float>> sorted_components;
            sorted_components.reserve(query_components.size());

            for (size_t i = 0; i < query_components.size(); ++i)
            {
                sorted_components.emplace_back(query_components[i], query_values[i]);
            }

            // Sort by decreasing value
            std::sort(sorted_components.begin(), sorted_components.end(),
                      [](const auto &a, const auto &b)
                      { return a.second > b.second; });

            // Take only up to query_cut components
            size_t components_to_use = std::min(query_cut, sorted_components.size());
            sorted_components.resize(components_to_use);

            // If JLT is configured, transform the query
            std::vector<float> transformed_query;
            if (jlt_.has_value())
            {
                auto transformed = jlt_->transform_sparse(query_components, query_values);
                transformed_query = std::vector<float>(transformed.begin(), transformed.end());
            }

            // Evaluate each posting list
            for (const auto &[component, value] : sorted_components)
            {
                const auto &posting_list = posting_lists_[component];

                // Get the block scores
                std::vector<std::pair<float, size_t>> block_scores;
                block_scores.reserve(posting_list.block_offsets().size() - 1);

                for (size_t i = 0; i < posting_list.block_offsets().size() - 1; ++i)
                {
                    // Get the block of postings
                    std::vector<uint64_t> packed_posting_block(
                        posting_list.packed_postings().begin() + posting_list.block_offsets()[i],
                        posting_list.packed_postings().begin() + posting_list.block_offsets()[i + 1]);

                    // Get the summary for this block
                    auto [summary_components, summary_values] = posting_list.get_centroids()[i];

                    // Compute dot product with summary
                    float dot;
                    if (jlt_.has_value())
                    {
                        // Transform summary using JLT
                        auto transformed_summary = jlt_->transform_sparse(summary_components, summary_values);
                        // Compute dot product in reduced space
                        dot = 0.0f;
                        for (size_t j = 0; j < transformed_query.size(); ++j)
                        {
                            dot += transformed_query[j] * transformed_summary[j];
                        }
                    }
                    else
                    {
                        dot = distances::dot_product_dense_sparse(query, summary_components, summary_values);
                    }

                    block_scores.emplace_back(dot, i);
                }

                // Sort blocks by score
                std::sort(block_scores.begin(), block_scores.end(),
                          [](const auto &a, const auto &b)
                          { return a.first > b.first; });

                // Evaluate blocks in order
                for (const auto &[dot, block_id] : block_scores)
                {
                    // Skip blocks that cannot contribute to the top-k
                    if (heap.len() == k && dot < -heap_factor * heap.topk().front().first)
                    {
                        continue;
                    }

                    // Get the block of postings
                    std::vector<uint64_t> packed_posting_block(
                        posting_list.packed_postings().begin() + posting_list.block_offsets()[block_id],
                        posting_list.packed_postings().begin() + posting_list.block_offsets()[block_id + 1]);

                    // Evaluate the block
                    evaluate_posting_block(
                        query,
                        query_components,
                        query_values,
                        packed_posting_block,
                        heap,
                        visited,
                        forward_index_);
                }
            }

            // Apply KNN refinement if configured
            if (knn_ && n_knn > 0)
            {
                // If JLT is configured, transform the query before KNN refinement
                if (jlt_.has_value())
                {
                    auto transformed = jlt_->transform_sparse(query_components, query_values);
                    std::vector<float> transformed_query(transformed.begin(), transformed.end());
                    knn_->refine<float>(transformed_query, heap, visited, forward_index_, n_knn);
                }
                else
                {
                    knn_->refine<float>(query, heap, visited, forward_index_, n_knn);
                }
            }

            // Get the results
            std::vector<std::pair<float, size_t>> results;
            results.reserve(k);

            for (const auto &[dot, offset] : heap.topk())
            {
                results.emplace_back(std::abs(dot), forward_index_.offset_to_id(offset));
            }

            return results;
        }

        /// `n_postings`: minimum number of postings to select for each component
        static InvertedIndex<T> build(const SparseDataset<T> &dataset, const Configuration &config)
        {
            // Initialize JLT if configured
            std::optional<SparseJLT> jlt;
            if (config.get_jlt_target_dim() > 0)
            {
                jlt.emplace(dataset.dim(), config.get_jlt_target_dim(), config.get_jlt_sparsity());
            }

            // Create posting lists
            std::vector<PostingList> posting_lists;
            posting_lists.reserve(dataset.dim());

            // Process each dimension
            for (size_t i = 0; i < dataset.dim(); ++i)
            {
                // Get postings for this dimension
                std::vector<size_t> posting_list;
                posting_list.reserve(dataset.len());

                for (size_t j = 0; j < dataset.len(); ++j)
                {
                    auto [components, values] = dataset.get(j);
                    for (size_t k = 0; k < components.size(); ++k)
                    {
                        if (components[k] == i)
                        {
                            posting_list.push_back(j);
                            break;
                        }
                    }
                }

                // Apply pruning strategy
                const auto &pruning = config.get_pruning();
                if (pruning.get_type() == PruningStrategy::Type::FixedSize)
                {
                    if (posting_list.size() > pruning.get_n_postings())
                    {
                        posting_list.resize(pruning.get_n_postings());
                    }
                }
                else if (pruning.get_type() == PruningStrategy::Type::GlobalThreshold)
                {
                    // TODO: Implement global threshold pruning
                }

                // Create posting list
                posting_lists.emplace_back(PostingList::build(posting_list, dataset, config));

                // Print progress
                if (i % 1000 == 0)
                {
                    std::cout << "Processed " << i << " dimensions" << std::endl;
                }
            }

            // Create KNN if configured
            std::optional<Knn> knn;
            if (config.get_knn_config().get_nknn() > 0)
            {
                knn.emplace(dataset, config.get_knn_config().get_nknn());
            }

            return InvertedIndex<T>(dataset, std::move(posting_lists), config, std::move(knn), std::move(jlt));
        }

        // Add a KNN graph to the index
        void add_knn(Knn new_knn)
        {
            knn_ = std::move(new_knn);
        }

        // Get the KNN graph
        const std::optional<Knn> &knn() const { return knn_; }

        // Get the KNN graph
        const std::optional<Knn> &knn_graph() const { return knn_; }

        // Fixed pruning strategy
        static void fixed_pruning(std::vector<std::vector<std::pair<T, size_t>>> &inverted_pairs, size_t n_postings)
        {
#pragma omp parallel for
            for (size_t i = 0; i < inverted_pairs.size(); ++i)
            {
                auto &posting_list = inverted_pairs[i];

                // Sort by decreasing score
                std::sort(posting_list.begin(), posting_list.end(),
                          [](const auto &a, const auto &b)
                          { return a.first > b.first; });

                // Truncate to n_postings
                if (posting_list.size() > n_postings)
                {
                    posting_list.resize(n_postings);
                }

                // Shrink to fit
                posting_list.shrink_to_fit();
            }
        }

        // Global threshold pruning strategy
        static void global_threshold_pruning(std::vector<std::vector<std::pair<T, size_t>>> &inverted_pairs, size_t n_postings)
        {
            const size_t tot_postings = inverted_pairs.size() * n_postings; // overall number of postings to select

            constexpr size_t EQUALITY_THRESHOLD = 10;
            const size_t max_eq_postings = EQUALITY_THRESHOLD * tot_postings / 100; // maximum number of posting with score equal to the threshold

            // For every posting we create the tuple <score, docid, id_posting_list>
            std::vector<std::tuple<T, size_t, uint16_t>> postings;

            for (size_t id = 0; id < inverted_pairs.size(); ++id)
            {
                auto &posting_list = inverted_pairs[id];
                for (const auto &[score, docid] : posting_list)
                {
                    postings.emplace_back(score, docid, static_cast<uint16_t>(id));
                }
                posting_list.clear();
            }

            const size_t actual_tot_postings = std::min(tot_postings, postings.size() - 1);

            // Sort postings by decreasing score
            std::sort(postings.begin(), postings.end(),
                      [](const auto &a, const auto &b)
                      { return std::get<0>(a) > std::get<0>(b); });

            // Find the threshold score
            const T threshold_score = std::get<0>(postings[actual_tot_postings]);

            // Count postings with score equal to the threshold
            size_t eq_count = 0;
            for (size_t i = actual_tot_postings; i < postings.size(); ++i)
            {
                if (std::get<0>(postings[i]) == threshold_score)
                {
                    eq_count++;
                }
                else
                {
                    break;
                }
            }

            if (eq_count > max_eq_postings)
            {
                std::cout << "A lot of entries have the same value. " << (eq_count - max_eq_postings)
                          << " have been pruned, for more info look at DOC_REFERENCE" << std::endl;
            }

            // Add all postings above threshold and up to max_eq_postings with equal threshold
            for (size_t i = 0; i < actual_tot_postings; ++i)
            {
                const auto &[score, docid, id_posting] = postings[i];
                inverted_pairs[id_posting].emplace_back(score, docid);
            }

            // Add postings with scores equal to the threshold, up to max_eq_postings
            size_t eq_added = 0;
            for (size_t i = actual_tot_postings; i < postings.size() && eq_added < max_eq_postings; ++i)
            {
                if (std::get<0>(postings[i]) == threshold_score)
                {
                    const auto &[score, docid, id_posting] = postings[i];
                    inverted_pairs[id_posting].emplace_back(score, docid);
                    eq_added++;
                }
                else
                {
                    break;
                }
            }
        }

        // Add JLT configuration and initialization
        void set_jlt(size_t target_dim, int sparsity = -1)
        {
            jlt_ = SparseJLT(forward_index_.dim(), target_dim, sparsity);
        }

        // Transform a sparse vector using JLT if configured
        std::vector<double> transform_vector(const std::vector<uint16_t> &components, const std::vector<float> &values)
        {
            if (jlt_.has_value())
            {
                return jlt_->transform_sparse(components, values);
            }
            // If no JLT configured, return original vector as dense
            std::vector<double> result(forward_index_.dim(), 0.0);
            for (size_t i = 0; i < components.size(); ++i)
            {
                result[components[i]] = values[i];
            }
            return result;
        }

        // Getters
        const SparseDataset<T> &dataset() const { return forward_index_; }

        // TODO return to this after SparseDatasetIterator is implemented
        // const SparseDatasetIterator<T>& iterator() const { return forward_index_.iterator(); }

        // Returns (offset, len) of the "id"-th document
        std::pair<size_t, size_t> id_to_offset_len(size_t id) const
        {
            return forward_index_.id_to_offset_len(id);
        }

        // Returns the id of the largest component, i.e., the dimensionality of the vectors in the dataset.
        size_t dim() const
        {
            return forward_index_.dim();
        }

        // Returns the number of non-zero components in the dataset.
        size_t nnz() const
        {
            return forward_index_.nnz();
        }

        // Returns the number of vectors in the dataset
        size_t len() const
        {
            return forward_index_.len();
        }

        // Checks if the dataset is empty.
        bool is_empty() const
        {
            return forward_index_.len() == 0;
        }

        // Returns the number of neighbors in the knn graph, 0 if knn graph is not present.
        size_t knn_len() const
        {
            return knn_.has_value() ? knn_->get_d() : 0;
        }

        // Space usage
        size_t space_usage_byte() const override;

        // Space usage
        size_t print_space_usage_byte() const;

        // Dynamic update methods
        void add_document_vector(const std::vector<uint16_t> &components, const std::vector<float> &values)
        {
            // TODO: Implement dynamic document addition
            // 1. Add document to forward index
            // 2. Update posting lists for each component
            // 3. Update centroids for affected blocks
            // 4. Update KNN graph if present
            throw std::runtime_error("Dynamic document addition not yet implemented");
        }

        void delete_document_vector(size_t doc_id)
        {
            // TODO: Implement dynamic document deletion
            // 1. Remove document from forward index
            // 2. Update posting lists for each component
            // 3. Update centroids for affected blocks
            // 4. Update KNN graph if present
            throw std::runtime_error("Dynamic document deletion not yet implemented");
        }

        template <class Archive>
        void serialize(Archive &archive)
        {
            archive(forward_index_, posting_lists_, config_, knn_, jlt_);
        }
    };

} // namespace seismic

#endif // INVERTED_INDEX_H
