// distances.h
#ifndef DISTANCES_H
#define DISTANCES_H

#include <vector>
#include <cstdint>
#include "utils.h"

namespace distances
{

    /**
     * Computes the dot product between a dense query and a sparse vector.
     * Before using this function, the query must be made dense. This is much faster
     * than computing the dot product with a "merge" style.
     *
     * @param query The dense query vector.
     * @param v_components The indices of the non-zero components in the vector.
     * @param v_values The values of the non-zero components in the vector.
     *
     * @return The dot product between the query and the vector.
     *
     * Example:
     * @code
     * std::vector<float> query = {1.0, 2.0, 3.0, 0.0};
     * std::vector<uint16_t> v_components = {0, 2, 3};
     * std::vector<float> v_values = {1.0, 1.0, 1.5};
     *
     * float result = dot_product_dense_sparse(query, v_components, v_values);
     * assert(result == 4.0);
     * @endcode
     */
    template <typename Q, typename V>
        requires utils::DataType<Q> && utils::DataType<V>
    inline float dot_product_dense_sparse(const std::vector<Q> &query,
                                          const std::vector<uint16_t> &v_components,
                                          const std::vector<V> &v_values)
    {
        constexpr size_t N_LANES = 4;
        std::array<float, N_LANES> result = {0.0f, 0.0f, 0.0f, 0.0f};

        // Process chunks of 4 elements at a time
        size_t chunks = v_components.size() / N_LANES;
        for (size_t i = 0; i < chunks; i++)
        {
            size_t base_idx = i * N_LANES;

            result[0] += query[v_components[base_idx]].to_f32().value_or(0.0f) * v_values[base_idx].to_f32().value_or(0.0f);
            result[1] += query[v_components[base_idx + 1]].to_f32().value_or(0.0f) * v_values[base_idx + 1].to_f32().value_or(0.0f);
            result[2] += query[v_components[base_idx + 2]].to_f32().value_or(0.0f) * v_values[base_idx + 2].to_f32().value_or(0.0f);
            result[3] += query[v_components[base_idx + 3]].to_f32().value_or(0.0f) * v_values[base_idx + 3].to_f32().value_or(0.0f);
        }

        // Handle remaining elements
        size_t len = v_components.size();
        size_t rem = len % N_LANES;

        if (rem > 0)
        {
            for (size_t i = len - rem; i < len; i++)
            {
                result[0] += query[v_components[i]].to_f32().value_or(0.0f) * v_values[i].to_f32().value_or(0.0f);
            }
        }

        // Sum all the partial results
        return result[0] + result[1] + result[2] + result[3];
    }

    /**
     * Computes the dot product between a sparse query and a sparse vector using binary search.
     * This function should be used when the query has just a few components.
     * Both the query's and vector's terms must be sorted by id.
     *
     * @param query_term_ids The ids of the query terms.
     * @param query_values The values of the query terms.
     * @param v_terms_ids The ids of the vector terms.
     * @param v_values The values of the vector terms.
     *
     * @return The dot product between the query and the vector.
     *
     * Example:
     * @code
     * std::vector<uint16_t> query_term_ids = {1, 2, 7};
     * std::vector<float> query_values = {1.0, 1.0, 1.0};
     * std::vector<uint16_t> v_term_ids = {0, 1, 2, 3, 4};
     * std::vector<float> v_values = {0.1, 1.0, 1.0, 1.0, 0.5};
     *
     * float result = dot_product_with_binary_search(query_term_ids, query_values, v_term_ids, v_values);
     * assert(result == 2.0);
     * @endcode
     */
    template <typename Q, typename V>
        requires utils::DataType<Q> && utils::DataType<V>
    inline float dot_product_with_binary_search(
        const std::vector<uint16_t> &query_term_ids,
        const std::vector<Q> &query_values,
        const std::vector<uint16_t> &v_terms_ids,
        const std::vector<V> &v_values)
    {

        float result = 0.0f;

        for (size_t j = 0; j < query_term_ids.size(); j++)
        {
            uint16_t term_id = query_term_ids[j];
            const Q &value = query_values[j];

            // Use branchless binary search
            size_t i = utils::binary_search_branchless(v_terms_ids, term_id);

            // Check if the term was found
            if (i < v_terms_ids.size() && v_terms_ids[i] == term_id)
            {
                result += value.to_f32().value_or(0.0f) * v_values[i].to_f32().value_or(0.0f);
            }
        }

        return result;
    }

    /**
     * Computes the dot product between a query and a vector using merge style.
     * This function should be used when the query has just a few components.
     * Both the query's and vector's terms must be sorted by id.
     *
     * @param query_term_ids The ids of the query terms.
     * @param query_values The values of the query terms.
     * @param v_term_ids The ids of the vector terms.
     * @param v_values The values of the vector terms.
     *
     * @return The dot product between the query and the vector.
     *
     * Example:
     * @code
     * std::vector<uint16_t> query_term_ids = {1, 2, 7};
     * std::vector<float> query_values = {1.0, 1.0, 1.0};
     * std::vector<uint16_t> v_term_ids = {0, 1, 2, 3, 4};
     * std::vector<float> v_values = {0.1, 1.0, 1.0, 1.0, 0.5};
     *
     * float result = dot_product_with_merge(query_term_ids, query_values, v_term_ids, v_values);
     * assert(result == 2.0);
     * @endcode
     */
    template <typename Q, typename V>
        requires utils::DataType<Q> && utils::DataType<V>
    inline float dot_product_with_merge(
        const std::vector<uint16_t> &query_term_ids,
        const std::vector<Q> &query_values,
        const std::vector<uint16_t> &v_term_ids,
        const std::vector<V> &v_values)
    {

        float result = 0.0f;
        size_t i = 0;

        for (size_t q = 0; q < query_term_ids.size(); q++)
        {
            uint16_t q_id = query_term_ids[q];
            const Q &q_v = query_values[q];

            // Find matching term ID using merge-join style
            while (i < v_term_ids.size() && v_term_ids[i] < q_id)
            {
                i++;
            }

            // End of vector terms or term ID greater than query term ID
            if (i >= v_term_ids.size())
            {
                break;
            }

            // If there's a match, include in dot product
            if (v_term_ids[i] == q_id)
            {
                result += v_values[i].to_f32().value_or(0.0f) * q_v.to_f32().value_or(0.0f);
            }
        }

        return result;
    }

} // namespace distances

#endif // DISTANCES_H