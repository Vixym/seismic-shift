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

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}