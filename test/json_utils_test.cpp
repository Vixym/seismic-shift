// json_utils_test.cpp
#include <gtest/gtest.h>
#include "../src/json_utils.h"
#include <fstream>
#include <sstream>

using namespace utils;

// Test for DocIdType class
TEST(JsonUtilsTest, DocIdType) {
    // Test string ID
    DocIdType string_id("test123");
    EXPECT_EQ(string_id.as_string(), "test123");
    
    // Test numeric ID
    DocIdType numeric_id(42);
    EXPECT_EQ(numeric_id.as_string(), "42");
    
    // Test JSON serialization/deserialization for string ID
    nlohmann::json j_string = string_id;
    EXPECT_EQ(j_string.get<std::string>(), "test123");
    
    DocIdType deserialized_string;
    from_json(j_string, deserialized_string);
    EXPECT_EQ(deserialized_string.as_string(), "test123");
    
    // Test JSON serialization/deserialization for numeric ID
    nlohmann::json j_numeric = numeric_id;
    EXPECT_EQ(j_numeric.get<size_t>(), 42);
    
    DocIdType deserialized_numeric;
    from_json(j_numeric, deserialized_numeric);
    EXPECT_EQ(deserialized_numeric.as_string(), "42");
}

// Test for JsonFormat class
TEST(JsonUtilsTest, JsonFormat) {
    // Create a test vector
    std::unordered_map<std::string, float> vec = {
        {"dim1", 1.0f},
        {"dim2", 2.5f},
        {"dim3", -3.7f}
    };
    
    // Test with string ID
    JsonFormat format_string(DocIdType("doc1"), vec);
    EXPECT_EQ(format_string.get_id_as_string(), "doc1");
    EXPECT_EQ(format_string.get_vector().size(), 3);
    EXPECT_EQ(format_string.get_vector().at("dim1"), 1.0f);
    EXPECT_EQ(format_string.get_vector().at("dim2"), 2.5f);
    EXPECT_EQ(format_string.get_vector().at("dim3"), -3.7f);
    
    // Test with numeric ID
    JsonFormat format_numeric(DocIdType(123), vec);
    EXPECT_EQ(format_numeric.get_id_as_string(), "123");
    
    // Test JSON serialization/deserialization
    nlohmann::json j = format_string;
    EXPECT_EQ(j["id"], "doc1");
    EXPECT_EQ(j["vector"]["dim1"], 1.0f);
    EXPECT_EQ(j["vector"]["dim2"], 2.5f);
    EXPECT_EQ(j["vector"]["dim3"], -3.7f);
    
    JsonFormat deserialized = j.get<JsonFormat>();
    EXPECT_EQ(deserialized.get_id_as_string(), "doc1");
    EXPECT_EQ(deserialized.get_vector().at("dim1"), 1.0f);
    EXPECT_EQ(deserialized.get_vector().at("dim2"), 2.5f);
    EXPECT_EQ(deserialized.get_vector().at("dim3"), -3.7f);
}

// Test for extract_jsonl function
TEST(JsonUtilsTest, ExtractJsonl) {
    // Create a test vector
    std::unordered_map<std::string, float> vec = {
        {"dim1", 1.0f},
        {"dim2", 2.5f},
        {"dim3", -3.7f}
    };
    
    JsonFormat format(DocIdType("doc1"), vec);
    
    // Test with Float type
    auto [id, coords, values] = extract_jsonl<Float>(format);
    
    EXPECT_EQ(id, "doc1");
    EXPECT_EQ(coords.size(), 3);
    EXPECT_EQ(values.size(), 3);
    
    // Since the order of elements in an unordered_map is not guaranteed,
    // we need to check each value individually
    for (size_t i = 0; i < coords.size(); i++) {
        const std::string& coord = coords[i];
        const Float& value = values[i];
        
        EXPECT_EQ(value.as_(), vec[coord]);
    }
    
    // Test with Float16 type
    auto [id16, coords16, values16] = extract_jsonl<Float16>(format);
    
    EXPECT_EQ(id16, "doc1");
    EXPECT_EQ(coords16.size(), 3);
    EXPECT_EQ(values16.size(), 3);
    
    for (size_t i = 0; i < coords16.size(); i++) {
        const std::string& coord = coords16[i];
        const Float16& value = values16[i];
        
        EXPECT_FLOAT_EQ(value.as_(), vec[coord]);
    }
}

// Test for read_queries function
TEST(JsonUtilsTest, ReadQueries) {
    // Create a temporary JSONL file
    std::string temp_file = "test_queries.jsonl";
    std::ofstream file(temp_file);
    
    file << R"({"id": "doc1", "vector": {"dim1": 1.0, "dim2": 2.5, "dim3": -3.7}})" << std::endl;
    file << R"({"id": 42, "vector": {"x": 0.5, "y": -1.2, "z": 3.0}})" << std::endl;
    file.close();
    
    // Read the queries
    auto queries = read_queries(temp_file);
    
    // Verify results
    ASSERT_EQ(queries.size(), 2);
    
    // First query
    auto [id1, coords1, values1] = queries[0];
    EXPECT_EQ(id1, "doc1");
    EXPECT_EQ(coords1.size(), 3);
    EXPECT_EQ(values1.size(), 3);
    
    // Create a map for easier lookup
    std::unordered_map<std::string, float> vec1;
    for (size_t i = 0; i < coords1.size(); i++) {
        vec1[coords1[i]] = values1[i];
    }
    
    EXPECT_FLOAT_EQ(vec1["dim1"], 1.0f);
    EXPECT_FLOAT_EQ(vec1["dim2"], 2.5f);
    EXPECT_FLOAT_EQ(vec1["dim3"], -3.7f);
    
    // Second query
    auto [id2, coords2, values2] = queries[1];
    EXPECT_EQ(id2, "42");
    EXPECT_EQ(coords2.size(), 3);
    EXPECT_EQ(values2.size(), 3);
    
    // Create a map for easier lookup
    std::unordered_map<std::string, float> vec2;
    for (size_t i = 0; i < coords2.size(); i++) {
        vec2[coords2[i]] = values2[i];
    }
    
    EXPECT_FLOAT_EQ(vec2["x"], 0.5f);
    EXPECT_FLOAT_EQ(vec2["y"], -1.2f);
    EXPECT_FLOAT_EQ(vec2["z"], 3.0f);
    
    // Clean up
    std::remove(temp_file.c_str());
    
    // Test with non-existent file
    EXPECT_THROW(read_queries("non_existent_file.jsonl"), std::runtime_error);
}

// Test error handling in read_queries
TEST(JsonUtilsTest, ReadQueriesErrorHandling) {
    // Create a temporary JSONL file with invalid JSON
    std::string temp_file = "test_invalid_queries.jsonl";
    std::ofstream file(temp_file);
    
    file << R"({"id": "doc1", "vector": {"dim1": 1.0, "dim2": 2.5, "dim3": -3.7}})" << std::endl;
    file << R"(invalid json)" << std::endl;
    file << R"({"id": 42, "vector": {"x": 0.5, "y": -1.2, "z": 3.0}})" << std::endl;
    file.close();
    
    // Redirect cerr to capture error messages
    std::stringstream cerr_buffer;
    std::streambuf* old_cerr = std::cerr.rdbuf(cerr_buffer.rdbuf());
    
    // Read the queries - should not throw despite invalid line
    auto queries = read_queries(temp_file);
    
    // Restore cerr
    std::cerr.rdbuf(old_cerr);
    
    // Verify results - should have 2 valid entries
    ASSERT_EQ(queries.size(), 2);
    
    // Check that error was logged
    std::string error_output = cerr_buffer.str();
    EXPECT_TRUE(error_output.find("Error parsing JSON") != std::string::npos);
    
    // Clean up
    std::remove(temp_file.c_str());
}
