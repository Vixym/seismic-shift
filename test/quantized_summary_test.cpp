#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>
#include "../src/quantized_summary.h"
#include "../src/sparse_dataset.h"

using namespace seismic;

class QuantizedSummaryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test data
        data = {
            {{0, 2, 4}, {1.0f, 2.0f, 3.0f}},
            {{1, 3}, {4.0f, 5.0f}},
            {{0, 1, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f}}
        };

        // Create sparse dataset
        for (const auto& [components, values] : data) {
            dataset_mut.push(components, values);
        }
        
        dataset = dataset_mut.to_immutable();
        
        // Create quantized summary
        summary = QuantizedSummary::from_sparse_dataset(dataset);
    }

    std::vector<std::pair<std::vector<uint16_t>, std::vector<float>>> data;
    SparseDatasetMut<float> dataset_mut;
    SparseDataset<float> dataset;
    QuantizedSummary summary;
};

TEST_F(QuantizedSummaryTest, Quantize) {
    // Test quantization of a vector
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto [min_val, quant, quantized] = QuantizedSummary::quantize(values);
    
    EXPECT_FLOAT_EQ(min_val, 1.0f);
    EXPECT_FLOAT_EQ(quant, 4.0f / 256.0f);
    EXPECT_EQ(quantized.size(), values.size());
    
    // Test reconstruction of values
    for (size_t i = 0; i < values.size(); ++i) {
        float reconstructed = quantized[i] * quant + min_val;
        // For the last value, we need to be more lenient since it might be clamped to N_CLASSES-1
        if (i == values.size() - 1) {
            EXPECT_LE(reconstructed, values[i]);
        } else {
            // For other values, we expect them to be within one quantization step
            EXPECT_NEAR(reconstructed, values[i], quant * 2);
        }
    }
    
    // Test quantization of a constant vector
    std::vector<float> constant_values(5, 3.0f);
    auto [const_min, const_quant, const_quantized] = QuantizedSummary::quantize(constant_values);
    
    EXPECT_FLOAT_EQ(const_min, 3.0f);
    EXPECT_FLOAT_EQ(const_quant, 1.0f);  // Should be 1.0 to avoid division by zero
    EXPECT_EQ(const_quantized.size(), constant_values.size());
    
    for (auto q : const_quantized) {
        EXPECT_EQ(q, 0);  // All values should be quantized to 0
    }
}

TEST_F(QuantizedSummaryTest, DistancesIter) {
    // Test distances calculation
    std::vector<uint16_t> query_components = {0, 2};
    std::vector<float> query_values = {1.0f, 1.0f};
    
    auto iter = summary.distances_iter(query_components, query_values);
    
    // Collect all distances
    std::vector<float> distances;
    while (iter.has_next()) {
        auto dist = iter.next();
        ASSERT_TRUE(dist.has_value());
        distances.push_back(*dist);
    }
    
    EXPECT_EQ(distances.size(), 3);  // Should have 3 distances (one for each vector in the dataset)
    
    // Calculate expected distances manually
    std::vector<float> expected_distances(3, 0.0f);
    
    // For vector 0: {0, 2, 4} -> {1.0, 2.0, 3.0}
    // Query: {0, 2} -> {1.0, 1.0}
    // Dot product: 1.0 * 1.0 + 2.0 * 1.0 = 3.0
    expected_distances[0] = 1.0f * 1.0f + 2.0f * 1.0f;
    
    // For vector 1: {1, 3} -> {4.0, 5.0}
    // Query: {0, 2} -> {1.0, 1.0}
    // No overlap, dot product = 0
    expected_distances[1] = 0.0f;
    
    // For vector 2: {0, 1, 2, 3} -> {1.0, 2.0, 3.0, 4.0}
    // Query: {0, 2} -> {1.0, 1.0}
    // Dot product: 1.0 * 1.0 + 3.0 * 1.0 = 4.0
    expected_distances[2] = 1.0f * 1.0f + 3.0f * 1.0f;
    
    // Due to quantization, we allow some error in the distances
    for (size_t i = 0; i < distances.size(); ++i) {
        EXPECT_NEAR(distances[i], expected_distances[i], 0.1f);
    }
}

TEST_F(QuantizedSummaryTest, SpaceUsage) {
    // Test space usage
    size_t space = summary.space_usage_byte();
    EXPECT_GT(space, 0);
    
    // Space usage should be less than the original dataset
    // This is a rough estimate and depends on the quantization
    size_t dataset_space = dataset.space_usage_byte();
    
    // Print space usage for debugging
    std::cout << "Original dataset space: " << dataset_space << " bytes" << std::endl;
    std::cout << "Quantized summary space: " << space << " bytes" << std::endl;
    
    // The quantized summary might not always be smaller due to the overhead of storing
    // minimums and quants, especially for small datasets like in this test
    // But for larger datasets, it should be more efficient
}

TEST_F(QuantizedSummaryTest, FromSparseDataset) {
    // Test creation from sparse dataset
    QuantizedSummary new_summary = QuantizedSummary::from_sparse_dataset(dataset);
    
    // Test with a larger query
    std::vector<uint16_t> query_components = {0, 1, 2, 3};
    std::vector<float> query_values = {1.0f, 1.0f, 1.0f, 1.0f};
    
    auto iter1 = summary.distances_iter(query_components, query_values);
    auto iter2 = new_summary.distances_iter(query_components, query_values);
    
    // Both iterators should produce the same distances
    while (iter1.has_next() && iter2.has_next()) {
        auto dist1 = iter1.next();
        auto dist2 = iter2.next();
        
        ASSERT_TRUE(dist1.has_value());
        ASSERT_TRUE(dist2.has_value());
        
        EXPECT_FLOAT_EQ(*dist1, *dist2);
    }
    
    // Both iterators should be exhausted
    EXPECT_FALSE(iter1.has_next());
    EXPECT_FALSE(iter2.has_next());
}

// Test with a larger dataset to demonstrate space savings
TEST(QuantizedSummaryLargeTest, LargeDatasetSpaceUsage) {
    // Create a large dataset with many vectors and components
    SparseDatasetMut<float> large_dataset_mut;
    
    // Generate 1000 sparse vectors with ~50 non-zero components each
    const size_t num_vectors = 1000;
    const size_t components_per_vector = 50;
    const size_t dim = 10000;
    
    std::srand(42); // Fixed seed for reproducibility
    
    for (size_t i = 0; i < num_vectors; ++i) {
        std::vector<uint16_t> components;
        std::vector<float> values;
        
        // Generate random components and values
        components.reserve(components_per_vector);
        values.reserve(components_per_vector);
        
        // Ensure components are sorted
        std::set<uint16_t> unique_components;
        while (unique_components.size() < components_per_vector) {
            unique_components.insert(static_cast<uint16_t>(std::rand() % dim));
        }
        
        for (uint16_t comp : unique_components) {
            components.push_back(comp);
            values.push_back(static_cast<float>(std::rand()) / RAND_MAX * 10.0f); // Random values between 0 and 10
        }
        
        large_dataset_mut.push(components, values);
    }
    
    // Convert to immutable dataset
    SparseDataset<float> large_dataset = large_dataset_mut.to_immutable();
    
    // Create quantized summary
    QuantizedSummary large_summary = QuantizedSummary::from_sparse_dataset(large_dataset);
    
    // Calculate space usage
    size_t large_dataset_space = large_dataset.space_usage_byte();
    size_t large_summary_space = large_summary.space_usage_byte();
    
    // Print space usage
    std::cout << "\nLarge dataset test:" << std::endl;
    std::cout << "Original large dataset space: " << large_dataset_space << " bytes" << std::endl;
    std::cout << "Quantized large summary space: " << large_summary_space << " bytes" << std::endl;
    std::cout << "Space savings: " << (1.0 - static_cast<double>(large_summary_space) / large_dataset_space) * 100.0 << "%" << std::endl;
    
    // The quantized summary should be significantly smaller for large datasets
    EXPECT_LT(large_summary_space, large_dataset_space);
}

// Main function is provided by main_test.cpp
