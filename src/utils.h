// utils.h
#ifndef UTILS_H
#define UTILS_H
#include <vector>
#include <unordered_set>
#include <string>
#include <functional>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <random>
#include <utility>
#include <numeric>
#include <optional>
#include <compare>

namespace utils
{
    /**
     * SpaceUsage trait (similar to Rust trait)
     */
    class SpaceUsage
    {
    public:
        virtual size_t size_in_bytes() const = 0;
        virtual ~SpaceUsage() = default;
    };

    /**
     * AsPrimitive trait - equivalent to Rust's AsPrimitive
     */
    template <typename T>
    class AsPrimitive
    {
    public:
        virtual T as_() const = 0;
        virtual ~AsPrimitive() = default;
    };

    /**
     * ToPrimitive trait - equivalent to Rust's ToPrimitive
     */
    class ToPrimitive
    {
    public:
        virtual std::optional<float> to_f32() const = 0;
        virtual std::optional<double> to_f64() const = 0;
        virtual std::optional<int32_t> to_i32() const = 0;
        virtual std::optional<int64_t> to_i64() const = 0;
        virtual std::optional<uint32_t> to_u32() const = 0;
        virtual std::optional<uint64_t> to_u64() const = 0;
        virtual ~ToPrimitive() = default;
    };

    /**
     * Zero trait - equivalent to Rust's Zero
     */
    template <typename T>
    class Zero
    {
    public:
        static T zero()
        {
            return T();
        }
    };

    /**
     * FromPrimitive trait - equivalent to Rust's FromPrimitive
     */
    class FromPrimitive
    {
    public:
        virtual bool from_f32(float value) = 0;
        virtual bool from_f64(double value) = 0;
        virtual bool from_i32(int32_t value) = 0;
        virtual bool from_i64(int64_t value) = 0;
        virtual bool from_u32(uint32_t value) = 0;
        virtual bool from_u64(uint64_t value) = 0;
        virtual ~FromPrimitive() = default;
    };

    /**
     * DataType concept (C++ equivalent of Rust trait)
     * Based on the Rust trait:
     * pub trait DataType:
     *     SpaceUsage + Copy + AsPrimitive + ToPrimitive + Zero + Send + Sync + PartialOrd + FromPrimitive
     * {}
     */
    template <typename T>
    concept DataType = std::is_copy_constructible_v<T> &&
                       std::is_copy_assignable_v<T> &&
                       std::is_move_constructible_v<T> &&
                       std::is_move_assignable_v<T> &&
                       std::is_default_constructible_v<T> &&
                       requires(T a, T b) {
                           { a.size_in_bytes() } -> std::convertible_to<size_t>;               // SpaceUsage
                           { T::zero() } -> std::convertible_to<T>;                            // Zero
                           { a.to_f32() } -> std::convertible_to<std::optional<float>>;        // ToPrimitive
                           { a.partial_cmp(b) } -> std::convertible_to<std::partial_ordering>; // PartialOrd
                           // Note: Send and Sync are Rust-specific thread safety traits
                           // C++ doesn't have direct equivalents that can be checked at compile time
                       };

    /**
     * Float wrapper complying with DataType concept
     */
    class Float : public SpaceUsage, public ToPrimitive, public FromPrimitive, public AsPrimitive<float>, public Zero<Float>
    {
    private:
        float value;

    public:
        Float() : value(0.0f) {}
        Float(float v) : value(v) {}

        // SpaceUsage implementation
        size_t size_in_bytes() const override
        {
            return sizeof(Float);
        }

        // Zero implementation - static method already defined in Zero<Float>
        static Float zero()
        {
            return Float(0.0f);
        }

        // Basic operations
        Float operator+(const Float &other) const
        {
            return Float(value + other.value);
        }

        Float operator*(const Float &other) const
        {
            return Float(value * other.value);
        }

        // AsPrimitive implementation
        float as_() const override
        {
            return value;
        }

        // ToPrimitive implementation
        std::optional<float> to_f32() const override
        {
            return value;
        }

        std::optional<double> to_f64() const override
        {
            return static_cast<double>(value);
        }

        std::optional<int32_t> to_i32() const override
        {
            return static_cast<int32_t>(value);
        }

        std::optional<int64_t> to_i64() const override
        {
            return static_cast<int64_t>(value);
        }

        std::optional<uint32_t> to_u32() const override
        {
            return value < 0 ? std::nullopt : std::optional<uint32_t>(static_cast<uint32_t>(value));
        }

        std::optional<uint64_t> to_u64() const override
        {
            return value < 0 ? std::nullopt : std::optional<uint64_t>(static_cast<uint64_t>(value));
        }

        // FromPrimitive implementation
        bool from_f32(float val) override
        {
            value = val;
            return true;
        }

        bool from_f64(double val) override
        {
            value = static_cast<float>(val);
            return true;
        }

        bool from_i32(int32_t val) override
        {
            value = static_cast<float>(val);
            return true;
        }

        bool from_i64(int64_t val) override
        {
            value = static_cast<float>(val);
            return true;
        }

        bool from_u32(uint32_t val) override
        {
            value = static_cast<float>(val);
            return true;
        }

        bool from_u64(uint64_t val) override
        {
            value = static_cast<float>(val);
            return true;
        }

        // Comparison (PartialOrd)
        std::partial_ordering partial_cmp(const Float &other) const
        {
            if (std::isnan(value) || std::isnan(other.value))
                return std::partial_ordering::unordered;
            if (value < other.value)
                return std::partial_ordering::less;
            if (value > other.value)
                return std::partial_ordering::greater;
            return std::partial_ordering::equivalent;
        }

        // Conversion operator
        operator float() const
        {
            return value;
        }

        // Equality operator
        bool operator==(const Float &other) const
        {
            return value == other.value;
        }
    };

    /**
     * Float16 wrapper (similar to Rust's f16 from half crate)
     * This is a simplified implementation as full f16 support would require a proper half-precision library
     */
    class Float16 : public SpaceUsage, public ToPrimitive, public FromPrimitive, public AsPrimitive<float>, public Zero<Float16>
    {
    private:
        // This is just a simplified representation - real f16 would use uint16_t
        float value; // We store as float for simplicity
    public:
        Float16() : value(0.0f) {}
        Float16(float v) : value(v) {}

        // SpaceUsage implementation
        size_t size_in_bytes() const override
        {
            return 2; // f16 is 2 bytes
        }

        // Zero implementation
        static Float16 zero()
        {
            return Float16(0.0f);
        }

        // Basic operations
        Float16 operator+(const Float16 &other) const
        {
            return Float16(value + other.value);
        }

        Float16 operator*(const Float16 &other) const
        {
            return Float16(value * other.value);
        }

        // AsPrimitive implementation
        float as_() const override
        {
            return value;
        }

        // ToPrimitive implementation
        std::optional<float> to_f32() const override
        {
            return value;
        }

        std::optional<double> to_f64() const override
        {
            return static_cast<double>(value);
        }

        std::optional<int32_t> to_i32() const override
        {
            return static_cast<int32_t>(value);
        }

        std::optional<int64_t> to_i64() const override
        {
            return static_cast<int64_t>(value);
        }

        std::optional<uint32_t> to_u32() const override
        {
            return value < 0 ? std::nullopt : std::optional<uint32_t>(static_cast<uint32_t>(value));
        }

        std::optional<uint64_t> to_u64() const override
        {
            return value < 0 ? std::nullopt : std::optional<uint64_t>(static_cast<uint64_t>(value));
        }

        // FromPrimitive implementation
        bool from_f32(float val) override
        {
            value = val;
            return true;
        }

        bool from_f64(double val) override
        {
            value = static_cast<float>(val);
            return true;
        }

        bool from_i32(int32_t val) override
        {
            value = static_cast<float>(val);
            return true;
        }

        bool from_i64(int64_t val) override
        {
            value = static_cast<float>(val);
            return true;
        }

        bool from_u32(uint32_t val) override
        {
            value = static_cast<float>(val);
            return true;
        }

        bool from_u64(uint64_t val) override
        {
            value = static_cast<float>(val);
            return true;
        }

        // Comparison (PartialOrd)
        std::partial_ordering partial_cmp(const Float16 &other) const
        {
            if (std::isnan(value) || std::isnan(other.value))
                return std::partial_ordering::unordered;
            if (value < other.value)
                return std::partial_ordering::less;
            if (value > other.value)
                return std::partial_ordering::greater;
            return std::partial_ordering::equivalent;
        }

        // Conversion operator
        operator float() const
        {
            return value;
        }

        // Equality operator
        bool operator==(const Float16 &other) const
        {
            return value == other.value;
        }
    };

    // Make Float and Float16 comply with DataType concept
    static_assert(DataType<Float>);
    static_assert(DataType<Float16>);

    /**
     * SparseDataset class for sparse data representation
     */
    template <typename T>
        requires DataType<T>
    class SparseDataset
    {
    private:
        std::vector<std::vector<std::pair<uint32_t, T>>> data;
        size_t dimensions;

    public:
        SparseDataset(size_t dim) : dimensions(dim) {}

        void add(const std::vector<std::pair<uint32_t, T>> &vec)
        {
            data.push_back(vec);
        }

        size_t size() const { return data.size(); }
        size_t dim() const { return dimensions; }

        const std::vector<std::pair<uint32_t, T>> &iter_vector(size_t idx) const
        {
            return data[idx];
        }

        std::pair<const uint32_t *, const T *> get(size_t idx) const
        {
            static std::vector<uint32_t> components;
            static std::vector<T> values;
            components.clear();
            values.clear();
            for (const auto &[comp, val] : data[idx])
            {
                components.push_back(comp);
                values.push_back(val);
            }
            // Add terminator
            components.push_back(0);
            values.push_back(T::zero());
            return {components.data(), values.data()};
        }

        // Calculate space usage
        size_t size_in_bytes() const
        {
            size_t total = sizeof(SparseDataset<T>);
            for (const auto &vec : data)
            {
                total += sizeof(std::vector<std::pair<uint32_t, T>>);
                total += vec.size() * sizeof(std::pair<uint32_t, T>);
            }
            return total;
        }
    };

    /**
     * Computes the size of the intersection of two unsorted lists.
     */
    template <typename T>
    inline size_t intersection(const std::vector<T> &s, const std::vector<T> &groundtruth)
    {
        std::unordered_set<T> s_set(s.begin(), s.end());
        size_t size = 0;
        for (const auto &v : groundtruth)
        {
            if (s_set.count(v) > 0)
            {
                size++;
            }
        }
        return size;
    }

    /**
     * Prefetch data for read with NTA (non-temporal access) hint
     */
    template <typename T>
    inline void prefetch_read_NTA(const T *data, size_t offset)
    {
        [[maybe_unused]] const int8_t *p = reinterpret_cast<const int8_t *>(data + offset);
#if defined(PREFETCH_ENABLED) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#include <immintrin.h>
        _mm_prefetch(reinterpret_cast<const char *>(p), _MM_HINT_NTA);
#elif defined(PREFETCH_ENABLED) && defined(__aarch64__)
#include <arm_neon.h>
        __builtin_prefetch(p, 0, 0); // PRFM PLDL1KEEP
#endif
    }

    /**
     * Returns the typename of the argument.
     */
    template <typename T>
    inline std::string_view type_of(const T &)
    {
        return typeid(T).name();
    }

    /**
     * Binary search implementation with reduced branching.
     */
    inline size_t binary_search_branchless(const std::vector<uint16_t> &data, uint16_t target)
    {
        if (data.empty())
            return 0;
        size_t base = 0;
        size_t size = data.size();
        while (size > 1)
        {
            size_t mid = size / 2;
            bool cmp = data[base + mid - 1] < target;
            base += cmp * mid; // Avoids a branch by using multiplication
            size -= mid;
        }
        return base;
    }

    /**
     * Dot product between dense and sparse vectors
     */
    template <typename T>
        requires DataType<T>
    float dot_product_dense_sparse(const std::vector<T> &dense, const uint32_t *components, const T *values)
    {
        float result = 0.0f;
        size_t idx = 0;
        while (components[idx] != 0 || !(values[idx] == T::zero()))
        {
            uint32_t component = components[idx];
            if (component < dense.size())
            {
                result += dense[component].to_f32().value_or(0.0f) * values[idx].to_f32().value_or(0.0f);
            }
            idx++;
        }
        return result;
    }

    /**
     * Compute centroid assignments based on dot product
     */
    template <typename T>
        requires DataType<T>
    std::vector<std::pair<size_t, size_t>> compute_centroid_assignments_approx_dot_product(
        const std::vector<size_t> &doc_ids,
        const std::vector<std::vector<std::pair<size_t, T>>> &inverted_index,
        const SparseDataset<T> &dataset,
        const std::vector<size_t> &centroids,
        const std::unordered_set<size_t> &to_avoid,
        size_t doc_cut)
    {
        std::vector<std::pair<size_t, size_t>> centroid_assignments;
        centroid_assignments.reserve(doc_ids.size());
        std::vector<float> scores(centroids.size(), 0.0f);

        for (const auto &doc_id : doc_ids)
        {
            std::fill(scores.begin(), scores.end(), 0.0f);

            // Get document components and sort them by value (descending)
            auto doc_components = dataset.iter_vector(doc_id);
            std::sort(doc_components.begin(), doc_components.end(),
                      [](const auto &a, const auto &b)
                      {
                          return b.second.partial_cmp(a.second) == std::partial_ordering::less;
                      });

            // Take only top doc_cut components
            size_t components_to_process = std::min(doc_cut, doc_components.size());

            for (size_t i = 0; i < components_to_process; i++)
            {
                const auto &[component_id, value] = doc_components[i];

                // Fix: Check if component_id is within the valid range
                if (component_id < inverted_index.size())
                {
                    for (const auto &[centroid_id, score] : inverted_index[component_id])
                    {
                        // Fix: Check if centroid_id is within valid range
                        if (centroid_id < scores.size())
                        {
                            scores[centroid_id] += score.to_f32().value_or(0.0f) * value.to_f32().value_or(0.0f);
                        }
                    }
                }
            }

            // Fix: Initialize with first valid centroid
            float max = -1.0f; // Start with negative value to ensure valid assignments
            size_t max_centroid_id = 0;
            bool found_valid = false;

            for (size_t centroid_id = 0; centroid_id < scores.size(); centroid_id++)
            {
                if (to_avoid.find(centroid_id) != to_avoid.end())
                {
                    continue;
                }

                if (!found_valid)
                {
                    max_centroid_id = centroid_id;
                    max = scores[centroid_id];
                    found_valid = true;
                    continue;
                }

                if (scores[centroid_id] > max)
                {
                    max = scores[centroid_id];
                    max_centroid_id = centroid_id;
                }
            }

            centroid_assignments.emplace_back(max_centroid_id, doc_id);
        }

        return centroid_assignments;
    }

    /**
     * Perform random k-means clustering with approximate dot product
     */
    template <typename T>
    std::vector<std::pair<size_t, size_t>> do_random_kmeans_on_docids_ii_approx_dot_product(
        const std::vector<size_t> &doc_ids,
        size_t n_clusters,
        const SparseDataset<T> &dataset,
        size_t min_cluster_size,
        size_t doc_cut)
    {
        // Initialize random number generator with fixed seed
        std::mt19937 rng(1142);

        // Select random centroids
        std::vector<size_t> centroid_ids;
        centroid_ids.reserve(n_clusters);

        std::sample(doc_ids.begin(), doc_ids.end(),
                    std::back_inserter(centroid_ids),
                    n_clusters, rng);

        // Build inverted index for centroids
        std::vector<std::vector<std::pair<size_t, T>>> inverted_index(dataset.dim());

        for (size_t i = 0; i < centroid_ids.size(); i++)
        {
            const auto &centroid_id = centroid_ids[i];
            for (const auto &[component, score] : dataset.iter_vector(centroid_id))
            {
                inverted_index[component].emplace_back(i, score);
            }
        }

        // Compute initial assignments
        auto centroid_assignments = compute_centroid_assignments_approx_dot_product(
            doc_ids, inverted_index, dataset, centroid_ids,
            std::unordered_set<size_t>(), doc_cut);

        // Sort assignments by centroid ID
        std::sort(centroid_assignments.begin(), centroid_assignments.end());

        // Group by centroid ID and handle small clusters
        std::vector<std::pair<size_t, size_t>> final_assignments;
        final_assignments.reserve(doc_ids.size());

        std::vector<size_t> to_be_reassigned;
        std::unordered_set<size_t> removed_centroids;

        size_t i = 0;
        while (i < centroid_assignments.size())
        {
            size_t start = i;
            size_t centroid_id = centroid_assignments[i].first;

            // Find end of current group
            while (i < centroid_assignments.size() && centroid_assignments[i].first == centroid_id)
            {
                i++;
            }

            // Check if cluster is too small
            if (i - start <= min_cluster_size)
            {
                for (size_t j = start; j < i; j++)
                {
                    to_be_reassigned.push_back(centroid_assignments[j].second);
                }
                removed_centroids.insert(centroid_id);
            }
            else
            {
                for (size_t j = start; j < i; j++)
                {
                    final_assignments.push_back(centroid_assignments[j]);
                }
            }
        }

        // Reassign documents from small clusters
        if (!to_be_reassigned.empty())
        {
            auto reassignments = compute_centroid_assignments_approx_dot_product(
                to_be_reassigned, inverted_index, dataset, centroid_ids,
                removed_centroids, doc_cut);

            final_assignments.insert(final_assignments.end(),
                                     reassignments.begin(),
                                     reassignments.end());
        }

        // Verify size matches
        assert(final_assignments.size() == doc_ids.size() &&
               "Final assignment size mismatch");

        // Sort final assignments
        std::sort(final_assignments.begin(), final_assignments.end());

        return final_assignments;
    }

    /**
     * Compute centroid assignments using exact dot product calculation
     */
    template <typename T>
    std::vector<std::pair<size_t, size_t>> compute_centroid_assignments_dot_product(
        const std::vector<size_t> &doc_ids,
        const std::vector<std::vector<std::pair<T, size_t>>> &inverted_index,
        const SparseDataset<T> &dataset,
        const std::vector<size_t> &centroids,
        const std::unordered_set<size_t> &to_avoid,
        size_t doc_cut)
    {
        std::vector<std::pair<size_t, size_t>> centroid_assignments;
        centroid_assignments.reserve(doc_ids.size());

        std::unordered_set<size_t> centroid_set(centroids.begin(), centroids.end());

        for (const auto &doc_id : doc_ids)
        {
            // If document is also a centroid, assign to itself
            if (centroid_set.count(doc_id) > 0 && to_avoid.count(doc_id) == 0)
            {
                centroid_assignments.emplace_back(doc_id, doc_id);
                continue;
            }

            // Densify the vector
            std::vector<T> dense_vector(dataset.dim(), T::zero());
            for (const auto &[component, value] : dataset.iter_vector(doc_id))
            {
                dense_vector[component] = value;
            }

            float max = 0.0f;
            size_t max_centroid_id = centroids[0];
            std::unordered_set<size_t> visited = to_avoid;

            // Sort query terms by score and evaluate posting list for top ones
            auto doc_components = dataset.iter_vector(doc_id);
            std::sort(doc_components.begin(), doc_components.end(),
                      [](const auto &a, const auto &b)
                      {
                          return b.second.partial_cmp(a.second) == std::partial_ordering::less;
                      });

            size_t components_to_process = std::min(doc_cut, doc_components.size());

            for (size_t i = 0; i < components_to_process; i++)
            {
                const auto &[component_id, _] = doc_components[i];

                for (const auto &[_, centroid_id] : inverted_index[component_id])
                {
                    if (visited.count(centroid_id) > 0)
                    {
                        continue;
                    }

                    visited.insert(centroid_id);
                    auto [v_components, v_values] = dataset.get(centroid_id);
                    float dot = dot_product_dense_sparse(dense_vector, v_components, v_values);

                    if (dot > max)
                    {
                        max = dot;
                        max_centroid_id = centroid_id;
                    }
                }
            }

            centroid_assignments.emplace_back(max_centroid_id, doc_id);
        }

        return centroid_assignments;
    }

    /**
     * Perform random k-means clustering with exact dot product calculation
     */
    template <typename T>
    std::vector<std::pair<size_t, size_t>> do_random_kmeans_on_docids_ii_dot_product(
        const std::vector<size_t> &doc_ids,
        size_t n_clusters,
        const SparseDataset<T> &dataset,
        size_t min_cluster_size,
        float pruning_factor,
        size_t doc_cut)
    {
        // Initialize random number generator with fixed seed
        std::mt19937 rng(42);

        // Select random centroids
        std::vector<size_t> centroid_ids;
        centroid_ids.reserve(n_clusters);

        std::sample(doc_ids.begin(), doc_ids.end(),
                    std::back_inserter(centroid_ids),
                    n_clusters, rng);

        // Calculate pruned list size
        size_t pruned_list_size = std::max<size_t>(5, static_cast<size_t>(doc_ids.size() * pruning_factor));

        // Build inverted index for centroids
        std::vector<std::vector<std::pair<T, size_t>>> inverted_index(dataset.dim());

        for (const auto &centroid_id : centroid_ids)
        {
            for (const auto &[component, score] : dataset.iter_vector(centroid_id))
            {
                inverted_index[component].emplace_back(score, centroid_id);
            }
        }

        // Sort and prune inverted index lists
        for (auto &list : inverted_index)
        {
            std::sort(list.begin(), list.end(),
                      [](const auto &a, const auto &b)
                      {
                          return b.first.partial_cmp(a.first) == std::partial_ordering::less;
                      });

            if (list.size() > pruned_list_size)
            {
                list.resize(pruned_list_size);
            }
        }

        // Compute initial assignments
        auto centroid_assignments = compute_centroid_assignments_dot_product(
            doc_ids, inverted_index, dataset, centroid_ids,
            std::unordered_set<size_t>(), doc_cut);

        // Sort assignments by centroid ID
        std::sort(centroid_assignments.begin(), centroid_assignments.end());

        // Group by centroid ID and handle small clusters
        std::vector<std::pair<size_t, size_t>> final_assignments;
        final_assignments.reserve(doc_ids.size());

        std::vector<size_t> to_be_reassigned;
        std::unordered_set<size_t> removed_centroids;

        size_t i = 0;
        while (i < centroid_assignments.size())
        {
            size_t start = i;
            size_t centroid_id = centroid_assignments[i].first;

            // Find end of current group
            while (i < centroid_assignments.size() && centroid_assignments[i].first == centroid_id)
            {
                i++;
            }

            // Check if cluster is too small
            if (i - start <= min_cluster_size)
            {
                for (size_t j = start; j < i; j++)
                {
                    to_be_reassigned.push_back(centroid_assignments[j].second);
                }
                removed_centroids.insert(centroid_id);
            }
            else
            {
                for (size_t j = start; j < i; j++)
                {
                    final_assignments.push_back(centroid_assignments[j]);
                }
            }
        }

        // Reassign documents from small clusters
        if (!to_be_reassigned.empty())
        {
            auto reassignments = compute_centroid_assignments_dot_product(
                to_be_reassigned, inverted_index, dataset, centroid_ids,
                removed_centroids, doc_cut);

            final_assignments.insert(final_assignments.end(),
                                     reassignments.begin(),
                                     reassignments.end());
        }

        // Verify size matches
        assert(final_assignments.size() == doc_ids.size() &&
               "Final assignment size mismatch");

        // Sort final assignments
        std::sort(final_assignments.begin(), final_assignments.end());

        return final_assignments;
    }

    /**
     * Compute centroid assignments using basic method
     */
    template <typename T>
    std::vector<std::pair<size_t, size_t>> compute_centroid_assignments(
        const std::vector<size_t> &doc_ids,
        const SparseDataset<T> &dataset,
        const std::vector<size_t> &centroids,
        const std::unordered_set<size_t> &to_avoid)
    {
        std::vector<std::pair<size_t, size_t>> centroid_assignments;
        centroid_assignments.reserve(doc_ids.size());

        std::unordered_set<size_t> centroid_set;
        for (const auto &centroid_id : centroids)
        {
            if (to_avoid.count(centroid_id) == 0)
            {
                centroid_set.insert(centroid_id);
            }
        }

        for (const auto &doc_id : doc_ids)
        {
            // If document is also a centroid, assign to itself
            if (centroid_set.count(doc_id) > 0)
            {
                centroid_assignments.emplace_back(doc_id, doc_id);
                continue;
            }

            // Densify the vector
            std::vector<T> dense_vector(dataset.dim(), T::zero());
            for (const auto &[component, value] : dataset.iter_vector(doc_id))
            {
                dense_vector[component] = value;
            }

            // Fix: Initialize with first available centroid
            size_t centroid_max = 0;
            float max = -1.0f; // Start with negative value to ensure any valid dot product will be greater
            bool found_first = false;

            for (const auto &centroid_id : centroids)
            {
                if (to_avoid.count(centroid_id) > 0)
                    continue;

                if (!found_first)
                {
                    centroid_max = centroid_id;
                    found_first = true;
                }

                auto [v_components, v_values] = dataset.get(centroid_id);
                float dot = dot_product_dense_sparse(dense_vector, v_components, v_values);

                if (dot > max)
                {
                    max = dot;
                    centroid_max = centroid_id;
                }
            }

            centroid_assignments.emplace_back(centroid_max, doc_id);
        }

        return centroid_assignments;
    }
    /**
     * Perform random k-means clustering on document IDs
     */
    template <typename T>
    std::vector<std::pair<size_t, size_t>> do_random_kmeans_on_docids(
        const std::vector<size_t> &doc_ids,
        size_t n_clusters,
        const SparseDataset<T> &dataset,
        size_t min_cluster_size)
    {
        // Initialize random number generator with fixed seed
        std::mt19937 rng(42);

        // Select random centroids
        std::vector<size_t> centroid_ids;
        centroid_ids.reserve(n_clusters);

        std::sample(doc_ids.begin(), doc_ids.end(),
                    std::back_inserter(centroid_ids),
                    n_clusters, rng);

        // Compute initial assignments
        auto centroid_assignments = compute_centroid_assignments(
            doc_ids, dataset, centroid_ids, std::unordered_set<size_t>());

        // Sort assignments by centroid ID
        std::sort(centroid_assignments.begin(), centroid_assignments.end());

        // Group by centroid ID and handle small clusters
        std::vector<std::pair<size_t, size_t>> final_assignments;
        final_assignments.reserve(doc_ids.size());

        std::vector<size_t> to_be_reassigned;
        std::unordered_set<size_t> removed_centroids;

        size_t i = 0;
        while (i < centroid_assignments.size())
        {
            size_t start = i;
            size_t centroid_id = centroid_assignments[i].first;

            // Find end of current group
            while (i < centroid_assignments.size() && centroid_assignments[i].first == centroid_id)
            {
                i++;
            }

            // Check if cluster is too small
            if (i - start <= min_cluster_size)
            {
                for (size_t j = start; j < i; j++)
                {
                    to_be_reassigned.push_back(centroid_assignments[j].second);
                }
                removed_centroids.insert(centroid_id);
            }
            else
            {
                for (size_t j = start; j < i; j++)
                {
                    final_assignments.push_back(centroid_assignments[j]);
                }
            }
        }

        // Reassign documents from small clusters
        if (!to_be_reassigned.empty())
        {
            auto reassignments = compute_centroid_assignments(
                to_be_reassigned, dataset, centroid_ids, removed_centroids);

            final_assignments.insert(final_assignments.end(),
                                     reassignments.begin(),
                                     reassignments.end());
        }

        // Verify size matches
        assert(final_assignments.size() == doc_ids.size() &&
               "Final assignment size mismatch");

        // Sort final assignments
        std::sort(final_assignments.begin(), final_assignments.end());

        return final_assignments;
    }

} // namespace utils

#endif // UTILS_H