// distances_test.cpp
#include <gtest/gtest.h>
#include "../src/distances.h"
#include "../src/utils.h"
#include <vector>
#include <cmath>

// Test fixture for distance functions
class DistancesTest : public ::testing::Test
{
protected:
    // Set up common test data
    void SetUp() override
    {
        // Convert basic types to DataType objects
        for (float f : float_query)
        {
            query.push_back(utils::Float(f));
        }

        for (float f : float_values)
        {
            values.push_back(utils::Float(f));
        }

        for (float f : float_query_values)
        {
            query_values.push_back(utils::Float(f));
        }

        for (float f : float_sparse_values)
        {
            sparse_values.push_back(utils::Float(f));
        }
    }

    // Test data for dense-sparse dot product
    std::vector<float> float_query = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.0f, 7.0f, 8.0f};
    std::vector<utils::Float> query;
    std::vector<uint16_t> components = {0, 2, 3, 7};
    std::vector<float> float_values = {1.5f, 2.5f, 3.5f, 0.5f};
    std::vector<utils::Float> values;

    // Test data for sparse-sparse dot product
    std::vector<uint16_t> query_term_ids = {1, 2, 5, 7};
    std::vector<float> float_query_values = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<utils::Float> query_values;
    std::vector<uint16_t> sparse_term_ids = {0, 1, 2, 5, 8};
    std::vector<float> float_sparse_values = {0.5f, 1.5f, 2.5f, 1.0f, 3.0f};
    std::vector<utils::Float> sparse_values;

    // Helper function to calculate expected dot product for dense-sparse
    float calculate_expected_dense_sparse()
    {
        float expected = 0.0f;
        for (size_t i = 0; i < components.size(); i++)
        {
            expected += float_query[components[i]] * float_values[i];
        }
        return expected;
    }

    // Helper function to calculate expected dot product for sparse-sparse
    float calculate_expected_sparse_sparse()
    {
        float expected = 0.0f;
        for (size_t i = 0; i < query_term_ids.size(); i++)
        {
            for (size_t j = 0; j < sparse_term_ids.size(); j++)
            {
                if (query_term_ids[i] == sparse_term_ids[j])
                {
                    expected += float_query_values[i] * float_sparse_values[j];
                }
            }
        }
        return expected;
    }
};

// Test dense-sparse dot product with normal vectors
TEST_F(DistancesTest, DenseSparseNormalCase)
{
    float result = distances::dot_product_dense_sparse(query, components, values);
    float expected = calculate_expected_dense_sparse();

    ASSERT_NEAR(result, expected, 1e-5);
    // Expected: 1.0*1.5 + 3.0*2.5 + 4.0*3.5 + 8.0*0.5 = 1.5 + 7.5 + 14.0 + 4.0 = 27.0
    ASSERT_NEAR(result, 27.0f, 1e-5);
}

// Test dense-sparse dot product with empty components
TEST_F(DistancesTest, DenseSparseEmptyComponents)
{
    std::vector<uint16_t> empty_components;
    std::vector<utils::Float> empty_values;

    float result = distances::dot_product_dense_sparse(query, empty_components, empty_values);

    ASSERT_NEAR(result, 0.0f, 1e-5);
}

// Test dense-sparse dot product with non-divisible-by-4 length
TEST_F(DistancesTest, DenseSparseNonDivisibleLength)
{
    std::vector<uint16_t> components_odd = {0, 2, 3, 4, 7};
    std::vector<utils::Float> values_odd;
    std::vector<float> float_values_odd = {1.5f, 2.5f, 3.5f, 1.0f, 0.5f};

    for (float f : float_values_odd)
    {
        values_odd.push_back(utils::Float(f));
    }

    float result = distances::dot_product_dense_sparse(query, components_odd, values_odd);
    float expected = 1.0f * 1.5f + 3.0f * 2.5f + 4.0f * 3.5f + 5.0f * 1.0f + 8.0f * 0.5f;

    ASSERT_NEAR(result, expected, 1e-5);
    ASSERT_NEAR(result, 32.0f, 1e-5);
}

// Test binary search dot product with normal vectors
TEST_F(DistancesTest, BinarySearchNormalCase)
{
    float result = distances::dot_product_with_binary_search(
        query_term_ids, query_values, sparse_term_ids, sparse_values);
    float expected = calculate_expected_sparse_sparse();

    ASSERT_NEAR(result, expected, 1e-5);
    // Expected: 1.0*1.5 + 2.0*2.5 + 3.0*1.0 = 1.5 + 5.0 + 3.0 = 9.5
    ASSERT_NEAR(result, 9.5f, 1e-5);
}

// Test binary search dot product with empty query
TEST_F(DistancesTest, BinarySearchEmptyQuery)
{
    std::vector<uint16_t> empty_query_ids;
    std::vector<utils::Float> empty_query_values;

    float result = distances::dot_product_with_binary_search(
        empty_query_ids, empty_query_values, sparse_term_ids, sparse_values);

    ASSERT_NEAR(result, 0.0f, 1e-5);
}

// Test binary search dot product with no overlap
TEST_F(DistancesTest, BinarySearchNoOverlap)
{
    std::vector<uint16_t> no_overlap_ids = {10, 11, 12};
    std::vector<utils::Float> no_overlap_values;
    std::vector<float> float_no_overlap = {1.0f, 2.0f, 3.0f};

    for (float f : float_no_overlap)
    {
        no_overlap_values.push_back(utils::Float(f));
    }

    float result = distances::dot_product_with_binary_search(
        no_overlap_ids, no_overlap_values, sparse_term_ids, sparse_values);

    ASSERT_NEAR(result, 0.0f, 1e-5);
}

// Test merge style dot product with normal vectors
TEST_F(DistancesTest, MergeStyleNormalCase)
{
    float result = distances::dot_product_with_merge(
        query_term_ids, query_values, sparse_term_ids, sparse_values);
    float expected = calculate_expected_sparse_sparse();

    ASSERT_NEAR(result, expected, 1e-5);
    // Expected: 1.0*1.5 + 2.0*2.5 + 3.0*1.0 = 1.5 + 5.0 + 3.0 = 9.5
    ASSERT_NEAR(result, 9.5f, 1e-5);
}

// Test merge style dot product with empty query
TEST_F(DistancesTest, MergeStyleEmptyQuery)
{
    std::vector<uint16_t> empty_query_ids;
    std::vector<utils::Float> empty_query_values;

    float result = distances::dot_product_with_merge(
        empty_query_ids, empty_query_values, sparse_term_ids, sparse_values);

    ASSERT_NEAR(result, 0.0f, 1e-5);
}

// Test merge style dot product with no overlap
TEST_F(DistancesTest, MergeStyleNoOverlap)
{
    std::vector<uint16_t> no_overlap_ids = {10, 11, 12};
    std::vector<utils::Float> no_overlap_values;
    std::vector<float> float_no_overlap = {1.0f, 2.0f, 3.0f};

    for (float f : float_no_overlap)
    {
        no_overlap_values.push_back(utils::Float(f));
    }

    float result = distances::dot_product_with_merge(
        no_overlap_ids, no_overlap_values, sparse_term_ids, sparse_values);

    ASSERT_NEAR(result, 0.0f, 1e-5);
}

// Test all three methods with the same data to ensure they produce the same results
TEST_F(DistancesTest, AllMethodsConsistency)
{
    // Create overlapping data that all methods can use
    std::vector<uint16_t> common_components = {0, 1, 2};
    std::vector<utils::Float> common_values;
    std::vector<float> float_common_values = {1.0f, 2.0f, 3.0f};

    for (float f : float_common_values)
    {
        common_values.push_back(utils::Float(f));
    }

    std::vector<utils::Float> dense_query;
    std::vector<float> float_dense_query = {1.0f, 2.0f, 3.0f, 0.0f};

    for (float f : float_dense_query)
    {
        dense_query.push_back(utils::Float(f));
    }

    // Run all three methods
    float result1 = distances::dot_product_dense_sparse(dense_query, common_components, common_values);
    float result2 = distances::dot_product_with_binary_search(common_components, common_values, common_components, common_values);
    float result3 = distances::dot_product_with_merge(common_components, common_values, common_components, common_values);

    // Calculate expected manually: (1*1 + 2*2 + 3*3) = 1 + 4 + 9 = 14
    float expected = 14.0f;

    // All should be equal
    ASSERT_NEAR(result1, expected, 1e-5);
    ASSERT_NEAR(result2, expected, 1e-5);
    ASSERT_NEAR(result3, expected, 1e-5);
    ASSERT_NEAR(result1, result2, 1e-5);
    ASSERT_NEAR(result2, result3, 1e-5);
}

// Test with Float16 type
TEST_F(DistancesTest, Float16Type)
{
    // Create vectors with Float16 type
    std::vector<utils::Float16> query16;
    std::vector<float> float_query16 = {1.0f, 2.0f, 3.0f, 4.0f};

    for (float f : float_query16)
    {
        query16.push_back(utils::Float16(f));
    }

    std::vector<uint16_t> components16 = {0, 2};
    std::vector<utils::Float16> values16;
    std::vector<float> float_values16 = {1.5f, 2.5f};

    for (float f : float_values16)
    {
        values16.push_back(utils::Float16(f));
    }

    float result = distances::dot_product_dense_sparse(query16, components16, values16);
    float expected = 1.0f * 1.5f + 3.0f * 2.5f;

    ASSERT_NEAR(result, expected, 1e-5);
    ASSERT_NEAR(result, 9.0f, 1e-5);
}

// Test performance characteristics (this is not a strict test)
TEST_F(DistancesTest, PerformanceComparison)
{
    // Create larger test data
    constexpr size_t large_size = 1000;
    std::vector<uint16_t> large_components;
    std::vector<utils::Float> large_values;
    std::vector<utils::Float> large_query(large_size * 2, utils::Float(1.0f));

    // Fill with some pattern
    for (size_t i = 0; i < large_size; i++)
    {
        if (i % 2 == 0)
        { // Add even indices to make it sparse
            large_components.push_back(static_cast<uint16_t>(i));
            large_values.push_back(utils::Float(static_cast<float>(i % 10)));
        }
    }

    // Time the different methods
    auto start1 = std::chrono::high_resolution_clock::now();
    float result1 = distances::dot_product_dense_sparse(large_query, large_components, large_values);
    auto end1 = std::chrono::high_resolution_clock::now();

    auto start2 = std::chrono::high_resolution_clock::now();
    float result2 = distances::dot_product_with_binary_search(large_components, large_values, large_components, large_values);
    auto end2 = std::chrono::high_resolution_clock::now();

    auto start3 = std::chrono::high_resolution_clock::now();
    float result3 = distances::dot_product_with_merge(large_components, large_values, large_components, large_values);
    auto end3 = std::chrono::high_resolution_clock::now();

    // Calculate durations
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3).count();

    // We expect all methods to return valid results
    ASSERT_GT(result1, 0.0f);
    ASSERT_GT(result2, 0.0f);
    ASSERT_GT(result3, 0.0f);

    // Log timings for informational purposes
    std::cout << "Dense-Sparse time: " << duration1 << " microseconds\n";
    std::cout << "Binary Search time: " << duration2 << " microseconds\n";
    std::cout << "Merge Style time: " << duration3 << " microseconds\n";

    // We don't assert on relative performance as it may vary by environment
}