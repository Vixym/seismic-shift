// top_k_selector_test.cpp
#include <gtest/gtest.h>
#include "../src/top_k_selector.h"
#include <vector>
#include <random>
#include <algorithm>

using namespace utils;

// Test basic functionality of HeapFaiss
TEST(TopKSelectorTest, HeapFaissBasic) {
    HeapFaiss selector(3);
    
    // Push some values
    selector.push(5.0f);
    selector.push(3.0f);
    selector.push(7.0f);
    
    // Get top-k
    auto result = selector.topk();
    
    // Check size
    ASSERT_EQ(result.size(), 3);
    
    // Check order (decreasing by distance)
    EXPECT_EQ(result[0].first, 7.0f);
    EXPECT_EQ(result[1].first, 5.0f);
    EXPECT_EQ(result[2].first, 3.0f);
    
    // Check IDs (should be 2, 0, 1 based on insertion order)
    EXPECT_EQ(result[0].second, 2);
    EXPECT_EQ(result[1].second, 0);
    EXPECT_EQ(result[2].second, 1);
}

// Test HeapFaiss with more elements than k
TEST(TopKSelectorTest, HeapFaissMoreThanK) {
    HeapFaiss selector(3);
    
    // Push more values than k
    selector.push(5.0f);
    selector.push(3.0f);
    selector.push(7.0f);
    selector.push(1.0f);
    selector.push(9.0f);
    
    // Get top-k
    auto result = selector.topk();
    
    // Check size (should still be k)
    ASSERT_EQ(result.size(), 3);
    
    // Check that we have the 3 smallest values
    EXPECT_EQ(result[0].first, 5.0f);
    EXPECT_EQ(result[1].first, 3.0f);
    EXPECT_EQ(result[2].first, 1.0f);
}

// Test push_with_id functionality
TEST(TopKSelectorTest, HeapFaissPushWithId) {
    HeapFaiss selector(3);
    
    // Push with custom IDs
    selector.push_with_id(5.0f, 100);
    selector.push_with_id(3.0f, 200);
    selector.push_with_id(7.0f, 300);
    
    // Get top-k
    auto result = selector.topk();
    
    // Check size
    ASSERT_EQ(result.size(), 3);
    
    // Check order (decreasing by distance)
    EXPECT_EQ(result[0].first, 7.0f);
    EXPECT_EQ(result[1].first, 5.0f);
    EXPECT_EQ(result[2].first, 3.0f);
    
    // Check custom IDs
    EXPECT_EQ(result[0].second, 300);
    EXPECT_EQ(result[1].second, 100);
    EXPECT_EQ(result[2].second, 200);
}

// Test extend functionality
TEST(TopKSelectorTest, HeapFaissExtend) {
    HeapFaiss selector(3);
    
    // Extend with a vector of distances
    std::vector<float> distances = {5.0f, 3.0f, 7.0f, 1.0f, 9.0f};
    selector.extend(distances);
    
    // Get top-k
    auto result = selector.topk();
    
    // Check size
    ASSERT_EQ(result.size(), 3);
    
    // Check that we have the 3 smallest values
    EXPECT_EQ(result[0].first, 5.0f);
    EXPECT_EQ(result[1].first, 3.0f);
    EXPECT_EQ(result[2].first, 1.0f);
}

// Test with random data
TEST(TopKSelectorTest, HeapFaissRandom) {
    const size_t k = 10;
    const size_t n = 1000;
    
    HeapFaiss selector(k);
    
    // Generate random distances
    std::vector<float> distances;
    distances.reserve(n);
    
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    
    for (size_t i = 0; i < n; ++i) {
        float value = dist(rng);
        distances.push_back(value);
        selector.push(value);
    }
    
    // Get top-k from selector
    auto result = selector.topk();
    
    // Check size
    ASSERT_EQ(result.size(), k);
    
    // Sort the original distances
    std::sort(distances.begin(), distances.end());
    
    // Check that we have the k smallest values (in reverse order)
    for (size_t i = 0; i < k; ++i) {
        EXPECT_FLOAT_EQ(result[k - i - 1].first, distances[i]);
    }
}

// Test with negative distances (useful for dot product similarity)
TEST(TopKSelectorTest, HeapFaissNegativeDistances) {
    HeapFaiss selector(3);
    
    // Push negative values (for dot product, smaller is better)
    selector.push(-5.0f);
    selector.push(-3.0f);
    selector.push(-7.0f);
    selector.push(-1.0f);
    selector.push(-9.0f);
    
    // Get top-k
    auto result = selector.topk();
    
    // Check size
    ASSERT_EQ(result.size(), 3);
    
    // Check that we have the 3 smallest values (most negative)
    // Note: The sort order is by decreasing value, so for negative numbers
    // this means -5 > -7 > -9 (i.e., -5 is "larger" than -9)
    EXPECT_EQ(result[0].first, -5.0f);
    EXPECT_EQ(result[1].first, -7.0f);
    EXPECT_EQ(result[2].first, -9.0f);
}

// Test with empty selector
TEST(TopKSelectorTest, HeapFaissEmpty) {
    HeapFaiss selector(3);
    
    // Get top-k without pushing any values
    auto result = selector.topk();
    
    // Check size (should be 0)
    EXPECT_EQ(result.size(), 0);
}

// Test with k=1
TEST(TopKSelectorTest, HeapFaissSingleElement) {
    HeapFaiss selector(1);
    
    // Push some values
    selector.push(5.0f);
    selector.push(3.0f);
    selector.push(7.0f);
    
    // Get top-k
    auto result = selector.topk();
    
    // Check size
    ASSERT_EQ(result.size(), 1);
    
    // Check that we have the smallest value
    EXPECT_EQ(result[0].first, 3.0f);
    EXPECT_EQ(result[0].second, 1);
}
