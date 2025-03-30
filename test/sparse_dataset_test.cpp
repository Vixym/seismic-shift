#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "../src/sparse_dataset.h"

using namespace seismic;

class SparseDatasetTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test data
        data = {
            {{0, 2, 4}, {1.0f, 2.0f, 3.0f}},
            {{1, 3}, {4.0f, 5.0f}},
            {{0, 1, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f}}
        };

        // Create mutable dataset
        for (const auto& [components, values] : data) {
            mutable_dataset.push(components, values);
        }

        // Convert to immutable dataset
        immutable_dataset = mutable_dataset.to_immutable();
    }

    std::vector<std::pair<std::vector<uint16_t>, std::vector<float>>> data;
    SparseDatasetMut<float> mutable_dataset;
    SparseDataset<float> immutable_dataset;
};

TEST_F(SparseDatasetTest, BasicProperties) {
    // Test immutable dataset properties
    EXPECT_EQ(immutable_dataset.len(), 3);
    EXPECT_EQ(immutable_dataset.dim(), 5);
    EXPECT_EQ(immutable_dataset.nnz(), 9);
    EXPECT_FALSE(immutable_dataset.is_empty());

    // Test mutable dataset properties
    EXPECT_EQ(mutable_dataset.len(), 3);
    EXPECT_EQ(mutable_dataset.dim(), 5);
    EXPECT_EQ(mutable_dataset.nnz(), 9);
    EXPECT_FALSE(mutable_dataset.is_empty());

    // Test empty datasets
    SparseDatasetMut<float> empty_mutable;
    SparseDataset<float> empty_immutable;
    
    EXPECT_EQ(empty_mutable.len(), 0);
    EXPECT_EQ(empty_mutable.dim(), 0);
    EXPECT_EQ(empty_mutable.nnz(), 0);
    EXPECT_TRUE(empty_mutable.is_empty());
    
    EXPECT_EQ(empty_immutable.len(), 0);
    EXPECT_EQ(empty_immutable.dim(), 0);
    EXPECT_EQ(empty_immutable.nnz(), 0);
    EXPECT_TRUE(empty_immutable.is_empty());
}

TEST_F(SparseDatasetTest, GetVector) {
    // Test getting vectors from immutable dataset
    for (size_t i = 0; i < data.size(); ++i) {
        auto [components, values] = immutable_dataset.get(i);
        EXPECT_EQ(components, data[i].first);
        EXPECT_EQ(values, data[i].second);
    }

    // Test getting vectors from mutable dataset
    for (size_t i = 0; i < data.size(); ++i) {
        auto [components, values] = mutable_dataset.get(i);
        EXPECT_EQ(components, data[i].first);
        EXPECT_EQ(values, data[i].second);
    }

    // Test out of range access
    EXPECT_THROW(immutable_dataset.get(3), std::out_of_range);
    EXPECT_THROW(mutable_dataset.get(3), std::out_of_range);
}

TEST_F(SparseDatasetTest, VectorProperties) {
    // Test vector lengths in immutable dataset
    EXPECT_EQ(immutable_dataset.vector_len(0), 3);
    EXPECT_EQ(immutable_dataset.vector_len(1), 2);
    EXPECT_EQ(immutable_dataset.vector_len(2), 4);

    // Test vector lengths in mutable dataset
    EXPECT_EQ(mutable_dataset.vector_len(0), 3);
    EXPECT_EQ(mutable_dataset.vector_len(1), 2);
    EXPECT_EQ(mutable_dataset.vector_len(2), 4);

    // Test vector offsets in immutable dataset
    EXPECT_EQ(immutable_dataset.vector_offset(0), 0);
    EXPECT_EQ(immutable_dataset.vector_offset(1), 3);
    EXPECT_EQ(immutable_dataset.vector_offset(2), 5);

    // Test offset to id conversion
    EXPECT_EQ(immutable_dataset.offset_to_id(0), 0);
    EXPECT_EQ(immutable_dataset.offset_to_id(3), 1);
    EXPECT_EQ(immutable_dataset.offset_to_id(5), 2);
    EXPECT_THROW(immutable_dataset.offset_to_id(1), std::out_of_range);

    // Test id to offset conversion
    EXPECT_EQ(immutable_dataset.id_to_offset(0), 0);
    EXPECT_EQ(immutable_dataset.id_to_offset(1), 3);
    EXPECT_EQ(immutable_dataset.id_to_offset(2), 5);
    EXPECT_THROW(immutable_dataset.id_to_offset(3), std::out_of_range);

    // Test id to offset and length conversion
    auto [offset0, len0] = immutable_dataset.id_to_offset_len(0);
    EXPECT_EQ(offset0, 0);
    EXPECT_EQ(len0, 3);

    auto [offset1, len1] = immutable_dataset.id_to_offset_len(1);
    EXPECT_EQ(offset1, 3);
    EXPECT_EQ(len1, 2);

    auto [offset2, len2] = immutable_dataset.id_to_offset_len(2);
    EXPECT_EQ(offset2, 5);
    EXPECT_EQ(len2, 4);

    EXPECT_THROW(immutable_dataset.id_to_offset_len(3), std::out_of_range);
}

TEST_F(SparseDatasetTest, GetWithOffset) {
    // Test getting vectors with offset from immutable dataset
    auto [components0, values0] = immutable_dataset.get_with_offset(0, 3);
    EXPECT_EQ(components0, data[0].first);
    EXPECT_EQ(values0, data[0].second);

    auto [components1, values1] = immutable_dataset.get_with_offset(3, 2);
    EXPECT_EQ(components1, data[1].first);
    EXPECT_EQ(values1, data[1].second);

    auto [components2, values2] = immutable_dataset.get_with_offset(5, 4);
    EXPECT_EQ(components2, data[2].first);
    EXPECT_EQ(values2, data[2].second);

    // Test out of range access
    EXPECT_THROW(immutable_dataset.get_with_offset(9, 1), std::out_of_range);
    EXPECT_THROW(immutable_dataset.get_with_offset(8, 2), std::out_of_range);
}

TEST_F(SparseDatasetTest, Iterator) {
    // Test iterator for immutable dataset
    size_t i = 0;
    for (const auto& [components, values] : immutable_dataset) {
        EXPECT_EQ(components, data[i].first);
        EXPECT_EQ(values, data[i].second);
        ++i;
    }
    EXPECT_EQ(i, data.size());
}

TEST_F(SparseDatasetTest, PushPairs) {
    SparseDatasetMut<float> dataset;
    
    // Test pushing pairs
    std::vector<std::pair<uint16_t, float>> pairs = {
        {0, 1.0f}, {2, 2.0f}, {4, 3.0f}
    };
    dataset.push_pairs(pairs);
    
    EXPECT_EQ(dataset.len(), 1);
    EXPECT_EQ(dataset.dim(), 5);
    EXPECT_EQ(dataset.nnz(), 3);
    
    auto [components, values] = dataset.get(0);
    std::vector<uint16_t> expected_components = {0, 2, 4};
    std::vector<float> expected_values = {1.0f, 2.0f, 3.0f};
    
    EXPECT_EQ(components, expected_components);
    EXPECT_EQ(values, expected_values);
    
    // Test pushing unsorted pairs
    std::vector<std::pair<uint16_t, float>> unsorted_pairs = {
        {3, 1.0f}, {1, 2.0f}, {5, 3.0f}
    };
    EXPECT_THROW(dataset.push_pairs(unsorted_pairs), std::invalid_argument);
    
    // Test pushing empty pairs
    std::vector<std::pair<uint16_t, float>> empty_pairs;
    EXPECT_THROW(dataset.push_pairs(empty_pairs), std::invalid_argument);
}

TEST_F(SparseDatasetTest, Push) {
    SparseDatasetMut<float> dataset;
    
    // Test pushing vectors
    std::vector<uint16_t> components = {0, 2, 4};
    std::vector<float> values = {1.0f, 2.0f, 3.0f};
    dataset.push(components, values);
    
    EXPECT_EQ(dataset.len(), 1);
    EXPECT_EQ(dataset.dim(), 5);
    EXPECT_EQ(dataset.nnz(), 3);
    
    auto [got_components, got_values] = dataset.get(0);
    EXPECT_EQ(got_components, components);
    EXPECT_EQ(got_values, values);
    
    // Test pushing vectors with different sizes
    std::vector<uint16_t> components2 = {1, 3};
    std::vector<float> values2 = {4.0f, 5.0f, 6.0f};
    EXPECT_THROW(dataset.push(components2, values2), std::invalid_argument);
    
    // Test pushing empty vectors
    std::vector<uint16_t> empty_components;
    std::vector<float> empty_values;
    EXPECT_THROW(dataset.push(empty_components, empty_values), std::invalid_argument);
    
    // Test pushing unsorted vectors
    std::vector<uint16_t> unsorted_components = {4, 2, 0};
    std::vector<float> unsorted_values = {3.0f, 2.0f, 1.0f};
    EXPECT_THROW(dataset.push(unsorted_components, unsorted_values), std::invalid_argument);
}

TEST_F(SparseDatasetTest, SpaceUsage) {
    // Test space usage for immutable dataset
    size_t expected_immutable_space = 
        sizeof(size_t) * 2 +                      // n_vecs and d
        immutable_dataset.len() * sizeof(size_t) + sizeof(size_t) +  // offsets
        immutable_dataset.nnz() * sizeof(uint16_t) +  // components
        immutable_dataset.nnz() * sizeof(float);      // values
    
    EXPECT_EQ(immutable_dataset.space_usage_byte(), expected_immutable_space);
    
    // Test space usage for mutable dataset
    size_t expected_mutable_space = 
        sizeof(size_t) +                          // d
        (mutable_dataset.len() + 1) * sizeof(size_t) +  // offsets
        mutable_dataset.nnz() * sizeof(uint16_t) +  // components
        mutable_dataset.nnz() * sizeof(float);      // values
    
    EXPECT_EQ(mutable_dataset.space_usage_byte(), expected_mutable_space);
}

// Main function is provided by main_test.cpp
