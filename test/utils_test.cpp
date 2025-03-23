// utils_test.cpp
#include <gtest/gtest.h>
#include "../src/utils.h"
#include <vector>
#include <unordered_set>
#include <string>
#include <random>

// Test for intersection function
TEST(UtilsTest, Intersection)
{
    std::vector<int> set1 = {1, 3, 5, 7, 9};
    std::vector<int> set2 = {2, 3, 6, 7, 10};

    // Expected intersection size: 2 (elements 3 and 7)
    EXPECT_EQ(utils::intersection(set1, set2), 2);

    // Empty intersection
    std::vector<int> set3 = {2, 4, 6, 8, 10};
    EXPECT_EQ(utils::intersection(set1, set3), 0);

    // One is subset of another
    std::vector<int> set4 = {3, 5, 7};
    EXPECT_EQ(utils::intersection(set1, set4), 3);

    // Same sets
    EXPECT_EQ(utils::intersection(set1, set1), set1.size());

    // Empty sets
    std::vector<int> empty;
    EXPECT_EQ(utils::intersection(empty, set1), 0);
    EXPECT_EQ(utils::intersection(set1, empty), 0);
    EXPECT_EQ(utils::intersection(empty, empty), 0);

    // String type test
    std::vector<std::string> strSet1 = {"apple", "banana", "cherry"};
    std::vector<std::string> strSet2 = {"banana", "date", "fig"};
    EXPECT_EQ(utils::intersection(strSet1, strSet2), 1);
}

// Prefetch is hardware-dependent and hard to test functionally
// We can just verify it compiles and doesn't crash
TEST(UtilsTest, PrefetchReadNTA)
{
    std::vector<int> data = {1, 2, 3, 4, 5};

    // This is mainly a compilation test
    utils::prefetch_read_NTA(data.data(), 2);

    // No assertions since the function's effects are hardware-dependent
    SUCCEED();
}

// Test for type_of function
TEST(UtilsTest, TypeOf)
{
    // Test with various types
    int i = 42;
    double d = 3.14;
    std::string s = "test";
    std::vector<int> v = {1, 2, 3};

    // The exact string representation is compiler-dependent,
    // but we can verify it contains the core type name
    std::string int_type = std::string(utils::type_of(i));
    std::string double_type = std::string(utils::type_of(d));
    std::string string_type = std::string(utils::type_of(s));
    std::string vector_type = std::string(utils::type_of(v));

    // Check that the type names are not empty
    EXPECT_FALSE(int_type.empty());
    EXPECT_FALSE(double_type.empty());
    EXPECT_FALSE(string_type.empty());
    EXPECT_FALSE(vector_type.empty());

    // Check that different types have different names
    EXPECT_NE(int_type, double_type);
    EXPECT_NE(int_type, string_type);
    EXPECT_NE(double_type, string_type);
}

// Test for binary_search_branchless function
TEST(UtilsTest, BinarySearchBranchless)
{
    // Test with sorted array
    std::vector<uint16_t> data = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};

    // Test finding exact values
    for (size_t i = 0; i < data.size(); i++)
    {
        EXPECT_EQ(utils::binary_search_branchless(data, data[i]), i);
    }

    // Test values not in the array
    EXPECT_EQ(utils::binary_search_branchless(data, 0), 0);
    EXPECT_EQ(utils::binary_search_branchless(data, 2), 1);
    EXPECT_EQ(utils::binary_search_branchless(data, 4), 2);
    EXPECT_EQ(utils::binary_search_branchless(data, 6), 3);
    EXPECT_EQ(utils::binary_search_branchless(data, 8), 4);

    // Test value larger than any in the array
    EXPECT_EQ(utils::binary_search_branchless(data, 20), data.size() - 1);
    EXPECT_EQ(utils::binary_search_branchless(data, 100), data.size() - 1);

    // Test with singleton array
    std::vector<uint16_t> single = {5};
    EXPECT_EQ(utils::binary_search_branchless(single, 3), 0);
    EXPECT_EQ(utils::binary_search_branchless(single, 5), 0);
    EXPECT_EQ(utils::binary_search_branchless(single, 7), 0);

    // Test with empty array
    std::vector<uint16_t> empty;
    EXPECT_EQ(utils::binary_search_branchless(empty, 5), 0);

    // Test with large array
    std::vector<uint16_t> large(1000);
    for (size_t i = 0; i < large.size(); i++)
    {
        large[i] = static_cast<uint16_t>(i * 2); // 0, 2, 4, ...
    }

    for (size_t i = 0; i < large.size(); i++)
    {
        EXPECT_EQ(utils::binary_search_branchless(large, large[i]), i);
    }

    // Random testing
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::vector<uint16_t> random_data(100);
    for (size_t i = 0; i < random_data.size(); i++)
    {
        random_data[i] = static_cast<uint16_t>(i * 10 + (rng() % 5)); // Some randomness but still sorted
    }

    std::sort(random_data.begin(), random_data.end());

    for (size_t i = 0; i < random_data.size(); i++)
    {
        size_t pos = utils::binary_search_branchless(random_data, random_data[i]);
        EXPECT_LE(pos, i);                           // Should find position <= actual index (for duplicates)
        EXPECT_EQ(random_data[pos], random_data[i]); // The value at the found position should match
    }
}

// Test for Float wrapper class
TEST(UtilsTest, FloatWrapper)
{
    utils::Float a(1.5f);
    utils::Float b(2.5f);
    utils::Float zero = utils::Float::zero();

    // Test basic operations
    EXPECT_EQ(static_cast<float>(a + b), 4.0f);
    EXPECT_EQ(static_cast<float>(a * b), 3.75f);
    EXPECT_EQ(static_cast<float>(zero), 0.0f);

    // Test to_f32
    EXPECT_EQ(a.to_f32().value(), 1.5f);
    EXPECT_EQ(b.to_f32().value(), 2.5f);
    EXPECT_EQ(zero.to_f32().value(), 0.0f);

    // Test partial_cmp
    EXPECT_EQ(a.partial_cmp(b), std::partial_ordering::less);
    EXPECT_EQ(b.partial_cmp(a), std::partial_ordering::greater);
    EXPECT_EQ(a.partial_cmp(a), std::partial_ordering::equivalent);

    // Test equality operator
    EXPECT_TRUE(a == utils::Float(1.5f));
    EXPECT_FALSE(a == b);
}

// Test for SparseDataset class
TEST(UtilsTest, SparseDataset)
{
    utils::SparseDataset<utils::Float> dataset(10); // 10 dimensions

    // Create sparse vectors
    std::vector<std::pair<uint32_t, utils::Float>> vec1 = {
        {0, utils::Float(1.0f)},
        {2, utils::Float(3.0f)},
        {5, utils::Float(2.0f)}};

    std::vector<std::pair<uint32_t, utils::Float>> vec2 = {
        {1, utils::Float(4.0f)},
        {5, utils::Float(1.0f)},
        {7, utils::Float(2.0f)}};

    // Test adding vectors
    dataset.add(vec1);
    dataset.add(vec2);

    // Test size and dimensions
    EXPECT_EQ(dataset.size(), 2);
    EXPECT_EQ(dataset.dim(), 10);

    // Test retrieving vectors
    const auto &retrieved_vec1 = dataset.iter_vector(0);
    const auto &retrieved_vec2 = dataset.iter_vector(1);

    // Check vector contents
    EXPECT_EQ(retrieved_vec1.size(), 3);
    EXPECT_EQ(retrieved_vec2.size(), 3);

    EXPECT_EQ(retrieved_vec1[0].first, 0);
    EXPECT_EQ(static_cast<float>(retrieved_vec1[0].second), 1.0f);
    EXPECT_EQ(retrieved_vec1[1].first, 2);
    EXPECT_EQ(static_cast<float>(retrieved_vec1[1].second), 3.0f);
    EXPECT_EQ(retrieved_vec1[2].first, 5);
    EXPECT_EQ(static_cast<float>(retrieved_vec1[2].second), 2.0f);

    // Test get method
    auto [components, values] = dataset.get(0);

    // Note: get() constructs a static vector internally, so we can't directly test
    // the contents. But we can verify it doesn't crash.
    EXPECT_NE(components, nullptr);
    EXPECT_NE(values, nullptr);
}

// Test for dot_product_dense_sparse function
TEST(UtilsTest, DotProductDenseSparse)
{
    // Create dense vector
    std::vector<utils::Float> dense = {
        utils::Float(1.0f), utils::Float(2.0f), utils::Float(3.0f),
        utils::Float(4.0f), utils::Float(5.0f)};

    // Create sparse vector components and values
    std::vector<uint32_t> sparse_components = {0, 2, 4, 0}; // Last 0 is a terminator
    std::vector<utils::Float> sparse_values = {
        utils::Float(2.0f), utils::Float(1.0f), utils::Float(3.0f), utils::Float::zero()};

    // Calculate expected result: 1.0*2.0 + 3.0*1.0 + 5.0*3.0 = 2.0 + 3.0 + 15.0 = 20.0
    float expected = 20.0f;

    // Test dot product
    float result = utils::dot_product_dense_sparse(dense, sparse_components.data(), sparse_values.data());
    EXPECT_FLOAT_EQ(result, expected);

    // Test with empty sparse vector
    std::vector<uint32_t> empty_components = {0};
    std::vector<utils::Float> empty_values = {utils::Float::zero()};
    result = utils::dot_product_dense_sparse(dense, empty_components.data(), empty_values.data());
    EXPECT_FLOAT_EQ(result, 0.0f);
}

// Test for compute_centroid_assignments function
TEST(UtilsTest, ComputeCentroidAssignments)
{
    // Create a dataset
    utils::SparseDataset<utils::Float> dataset(5); // 5 dimensions

    // Create documents
    std::vector<std::pair<uint32_t, utils::Float>> doc0 = {
        {0, utils::Float(1.0f)}, {2, utils::Float(2.0f)}};
    std::vector<std::pair<uint32_t, utils::Float>> doc1 = {
        {1, utils::Float(3.0f)}, {3, utils::Float(4.0f)}};
    std::vector<std::pair<uint32_t, utils::Float>> doc2 = {
        {0, utils::Float(2.0f)}, {2, utils::Float(3.0f)}};
    std::vector<std::pair<uint32_t, utils::Float>> doc3 = {
        {1, utils::Float(1.0f)}, {3, utils::Float(2.0f)}};

    dataset.add(doc0);
    dataset.add(doc1);
    dataset.add(doc2);
    dataset.add(doc3);

    // Doc IDs to cluster
    std::vector<size_t> doc_ids = {0, 1, 2, 3};

    // Choose centroids
    std::vector<size_t> centroids = {0, 1};

    // Compute assignments
    auto assignments = utils::compute_centroid_assignments(
        doc_ids, dataset, centroids, std::unordered_set<size_t>());

    // Verify basic properties
    EXPECT_EQ(assignments.size(), 4);

    // Sort assignments by document ID for easy verification
    std::sort(assignments.begin(), assignments.end(),
              [](const auto &a, const auto &b)
              { return a.second < b.second; });

    // Doc 0 should be assigned to centroid 0 (itself)
    EXPECT_EQ(assignments[0].second, 0);
    EXPECT_EQ(assignments[0].first, 0);

    // Doc 1 should be assigned to centroid 1 (itself)
    EXPECT_EQ(assignments[1].second, 1);
    EXPECT_EQ(assignments[1].first, 1);

    // Doc 2 should be assigned to centroid 0 (more similar pattern)
    EXPECT_EQ(assignments[2].second, 2);
    EXPECT_EQ(assignments[2].first, 0);

    // Doc 3 should be assigned to centroid 1 (more similar pattern)
    EXPECT_EQ(assignments[3].second, 3);
    EXPECT_EQ(assignments[3].first, 1);

    // Test with excluded centroid
    std::unordered_set<size_t> to_avoid = {0};
    auto assignments_with_avoid = utils::compute_centroid_assignments(
        doc_ids, dataset, centroids, to_avoid);

    // All documents should be assigned to centroid 1 now
    for (const auto &[centroid_id, doc_id] : assignments_with_avoid)
    {
        EXPECT_EQ(centroid_id, 1);
    }
}

// Test for do_random_kmeans_on_docids function (basic version)
TEST(UtilsTest, RandomKmeansOnDocIds)
{
    // Create a dataset
    utils::SparseDataset<utils::Float> dataset(5); // 5 dimensions

    // Create 10 documents with two clear clusters
    for (int i = 0; i < 5; i++)
    {
        // Cluster 1: Strong in dimensions 0, 2
        std::vector<std::pair<uint32_t, utils::Float>> doc = {
            {0, utils::Float(1.0f + 0.1f * i)},
            {2, utils::Float(2.0f + 0.1f * i)}};
        dataset.add(doc);
    }

    for (int i = 0; i < 5; i++)
    {
        // Cluster 2: Strong in dimensions 1, 3
        std::vector<std::pair<uint32_t, utils::Float>> doc = {
            {1, utils::Float(3.0f + 0.1f * i)},
            {3, utils::Float(2.0f + 0.1f * i)}};
        dataset.add(doc);
    }

    // Doc IDs to cluster
    std::vector<size_t> doc_ids(10);
    std::iota(doc_ids.begin(), doc_ids.end(), 0);

    // Perform k-means
    auto result = utils::do_random_kmeans_on_docids(doc_ids, 2, dataset, 1);

    // Verify result size
    EXPECT_EQ(result.size(), doc_ids.size());

    // Group documents by assigned centroid
    std::vector<size_t> centroid1_docs;
    std::vector<size_t> centroid2_docs;

    for (const auto &[centroid_id, doc_id] : result)
    {
        if (centroid_id == result[0].first)
        {
            centroid1_docs.push_back(doc_id);
        }
        else
        {
            centroid2_docs.push_back(doc_id);
        }
    }

    // Verify we have two non-empty clusters
    EXPECT_FALSE(centroid1_docs.empty());
    EXPECT_FALSE(centroid2_docs.empty());

    // Note: we can't verify exact clustering due to random seed,
    // but we can check if same-type documents tend to cluster together

    // Count documents from our logical "first cluster" in each k-means cluster
    int first_cluster_in_centroid1 = 0;
    for (size_t doc_id : centroid1_docs)
    {
        if (doc_id < 5)
        {
            first_cluster_in_centroid1++;
        }
    }

    int first_cluster_in_centroid2 = 0;
    for (size_t doc_id : centroid2_docs)
    {
        if (doc_id < 5)
        {
            first_cluster_in_centroid2++;
        }
    }

    // Either centroid1 should mostly contain docs 0-4 or centroid2 should
    bool good_clustering =
        (first_cluster_in_centroid1 >= 3 && centroid1_docs.size() <= 7) ||
        (first_cluster_in_centroid2 >= 3 && centroid2_docs.size() <= 7);

    EXPECT_TRUE(good_clustering);
}

// Test for do_random_kmeans_on_docids_ii_dot_product function
TEST(UtilsTest, RandomKmeansWithDotProduct)
{
    // Create a dataset
    utils::SparseDataset<utils::Float> dataset(5); // 5 dimensions

    // Create 10 documents with two clear clusters
    for (int i = 0; i < 5; i++)
    {
        // Cluster 1: Strong in dimensions 0, 2
        std::vector<std::pair<uint32_t, utils::Float>> doc = {
            {0, utils::Float(1.0f + 0.1f * i)},
            {2, utils::Float(2.0f + 0.1f * i)}};
        dataset.add(doc);
    }

    for (int i = 0; i < 5; i++)
    {
        // Cluster 2: Strong in dimensions 1, 3
        std::vector<std::pair<uint32_t, utils::Float>> doc = {
            {1, utils::Float(3.0f + 0.1f * i)},
            {3, utils::Float(2.0f + 0.1f * i)}};
        dataset.add(doc);
    }

    // Doc IDs to cluster
    std::vector<size_t> doc_ids(10);
    std::iota(doc_ids.begin(), doc_ids.end(), 0);

    // Perform k-means with dot product
    auto result = utils::do_random_kmeans_on_docids_ii_dot_product(
        doc_ids, 2, dataset, 1, 0.5f, 5);

    // Verify result size
    EXPECT_EQ(result.size(), doc_ids.size());

    // Group documents by assigned centroid
    std::vector<size_t> centroid1_docs;
    std::vector<size_t> centroid2_docs;

    for (const auto &[centroid_id, doc_id] : result)
    {
        if (centroid_id == result[0].first)
        {
            centroid1_docs.push_back(doc_id);
        }
        else
        {
            centroid2_docs.push_back(doc_id);
        }
    }

    // Verify we have two non-empty clusters
    EXPECT_FALSE(centroid1_docs.empty());
    EXPECT_FALSE(centroid2_docs.empty());

    // Similar check as above
    int first_cluster_in_centroid1 = 0;
    for (size_t doc_id : centroid1_docs)
    {
        if (doc_id < 5)
        {
            first_cluster_in_centroid1++;
        }
    }

    int first_cluster_in_centroid2 = 0;
    for (size_t doc_id : centroid2_docs)
    {
        if (doc_id < 5)
        {
            first_cluster_in_centroid2++;
        }
    }

    bool good_clustering =
        (first_cluster_in_centroid1 >= 3 && centroid1_docs.size() <= 7) ||
        (first_cluster_in_centroid2 >= 3 && centroid2_docs.size() <= 7);

    EXPECT_TRUE(good_clustering);
}

// Test for do_random_kmeans_on_docids_ii_approx_dot_product function
TEST(UtilsTest, RandomKmeansWithApproxDotProduct)
{
    // Create a dataset
    utils::SparseDataset<utils::Float> dataset(5); // 5 dimensions

    // Create 10 documents with two clear clusters
    for (int i = 0; i < 5; i++)
    {
        // Cluster 1: Strong in dimensions 0, 2
        std::vector<std::pair<uint32_t, utils::Float>> doc = {
            {0, utils::Float(1.0f + 0.1f * i)},
            {2, utils::Float(2.0f + 0.1f * i)}};
        dataset.add(doc);
    }

    for (int i = 0; i < 5; i++)
    {
        // Cluster 2: Strong in dimensions 1, 3
        std::vector<std::pair<uint32_t, utils::Float>> doc = {
            {1, utils::Float(3.0f + 0.1f * i)},
            {3, utils::Float(2.0f + 0.1f * i)}};
        dataset.add(doc);
    }

    // Doc IDs to cluster
    std::vector<size_t> doc_ids(10);
    std::iota(doc_ids.begin(), doc_ids.end(), 0);

    // Perform k-means with approximate dot product
    auto result = utils::do_random_kmeans_on_docids_ii_approx_dot_product(
        doc_ids, 2, dataset, 1, 5);

    // Verify result size
    EXPECT_EQ(result.size(), doc_ids.size());

    // Group documents by assigned centroid
    std::vector<size_t> centroid1_docs;
    std::vector<size_t> centroid2_docs;

    for (const auto &[centroid_id, doc_id] : result)
    {
        if (centroid_id == result[0].first)
        {
            centroid1_docs.push_back(doc_id);
        }
        else
        {
            centroid2_docs.push_back(doc_id);
        }
    }

    // Verify we have two non-empty clusters
    EXPECT_FALSE(centroid1_docs.empty());
    EXPECT_FALSE(centroid2_docs.empty());

    // Similar check as above
    int first_cluster_in_centroid1 = 0;
    for (size_t doc_id : centroid1_docs)
    {
        if (doc_id < 5)
        {
            first_cluster_in_centroid1++;
        }
    }

    int first_cluster_in_centroid2 = 0;
    for (size_t doc_id : centroid2_docs)
    {
        if (doc_id < 5)
        {
            first_cluster_in_centroid2++;
        }
    }

    bool good_clustering =
        (first_cluster_in_centroid1 >= 3 && centroid1_docs.size() <= 7) ||
        (first_cluster_in_centroid2 >= 3 && centroid2_docs.size() <= 7);

    EXPECT_TRUE(good_clustering);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}