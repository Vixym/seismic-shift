#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>
#include <memory>

#include "../src/elias_fano.h"
#include "../src/inverted_index.h"
#include "../src/seismic_index.h"
#include "../src/sparse_dataset.h"

using namespace seismic;

class SeismicIndexTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test data in JSONL format
        std::stringstream json_data;
        json_data << R"({"id": "doc1", "tokens": ["word1", "word2", "word3"], "values": [1.0, 2.0, 3.0]})" << std::endl;
        json_data << R"({"id": "doc2", "tokens": ["word2", "word4"], "values": [4.0, 5.0]})" << std::endl;
        json_data << R"({"id": "doc3", "tokens": ["word1", "word2", "word3", "word4"], "values": [1.0, 2.0, 3.0, 4.0]})" << std::endl;
        
        // Process the test data
        size_t row_count = 3;
        auto [dataset, doc_mapping, token_map] = SeismicIndex<float>::process_data(json_data, row_count);
        
        // Create configuration
        config = Configuration()
            .pruning_strategy(PruningStrategy::fixed_size(10))
            .blocking_strategy(BlockingStrategy::fixed_size(2))
            .summarization_strategy(SummarizationStrategy::fixed_size(3));

        // Create the index
        index = std::make_unique<SeismicIndex<float>>(SeismicIndex<float>::new_from_dataset(
            dataset,
            config,
            std::move(doc_mapping),
            std::move(token_map)
        ));
    }

    Configuration config;
    std::unique_ptr<SeismicIndex<float>> index;
};

TEST_F(SeismicIndexTest, SearchRaw) {
    // Test raw search functionality
    std::vector<std::string> query_tokens = {"word1", "word2"};
    std::vector<float> query_values = {1.0f, 2.0f};
    
    auto results = index->search_raw(query_tokens, query_values, 2, 2, 0.7f, 0, false);
    
    // Should get 2 results
    EXPECT_EQ(results.size(), 2);
    
    // Results should be sorted by score
    EXPECT_GT(std::abs(results[0].first), std::abs(results[1].first));
}

TEST_F(SeismicIndexTest, SearchMapped) {
    // Test search with document mapping
    std::vector<std::string> query_tokens = {"word1", "word2"};
    std::vector<float> query_values = {1.0f, 2.0f};
    
    auto results = index->search("query1", query_tokens, query_values, 2, 2, 0.7f, 0, false);
    
    // Should get 2 results
    EXPECT_EQ(results.size(), 2);
    
    // Check result format
    for (const auto& [query_id, score, doc_id] : results) {
        EXPECT_EQ(query_id, "query1");
        EXPECT_TRUE(doc_id.find("doc") != std::string::npos);
        EXPECT_GT(std::abs(score), 0.0f);
    }
}

TEST_F(SeismicIndexTest, TokenMapping) {
    // Test with unknown tokens
    std::vector<std::string> query_tokens = {"unknown_word", "word1"};
    std::vector<float> query_values = {1.0f, 2.0f};
    
    auto results = index->search_raw(query_tokens, query_values, 2, 2, 0.7f, 0, false);
    
    // Should still get results, but only based on "word1"
    EXPECT_GT(results.size(), 0);
}

TEST_F(SeismicIndexTest, ProcessDataWithExistingMapping) {
    // Create a predefined token mapping
    std::unordered_map<std::string, size_t> existing_map = {
        {"word1", 0},
        {"word2", 1}
    };
    
    std::stringstream json_data;
    json_data << R"({"id": "doc1", "tokens": ["word1", "word2", "word3"], "values": [1.0, 2.0, 3.0]})" << std::endl;
    
    // Process data with existing mapping
    auto [dataset, doc_mapping, token_map] = SeismicIndex<float>::process_data(
        json_data, 
        1, 
        std::optional<std::unordered_map<std::string, size_t>>(existing_map)
    );
    
    // Should only have mapped the tokens that were in the existing mapping
    EXPECT_EQ(dataset.len(), 1);
    auto [components, values] = dataset.get(0);
    EXPECT_EQ(components.size(), 2); // Only word1 and word2 should be mapped
}

TEST_F(SeismicIndexTest, SpaceUsage) {
    size_t space = index->space_usage_byte();
    EXPECT_GT(space, 0);
}

// Main function is provided by main_test.cpp
