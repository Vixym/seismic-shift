#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <memory>
#include <unordered_set>

#include "../src/distances.h"
#include "../src/elias_fano.h"
#include "../src/inverted_index.h"
#include "../src/sparse_dataset.h"
#include "../src/quantized_summary.h"

using namespace seismic;

class InvertedIndexTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test data
        data = {
            {{0, 2, 4}, {1.0f, 2.0f, 3.0f}},
            {{1, 3}, {4.0f, 5.0f}},
            {{0, 1, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f}},
            {{2, 4}, {2.5f, 3.5f}},
            {{0, 1, 3, 4}, {1.5f, 2.5f, 3.5f, 4.5f}}
        };

        // Create mutable dataset
        for (const auto& [components, values] : data) {
            mutable_dataset.push(components, values);
        }

        // Convert to immutable dataset
        dataset = mutable_dataset.to_immutable();

        // Create configuration for the inverted index
        config = Configuration()
            .pruning_strategy(PruningStrategy::fixed_size(10))
            .blocking_strategy(BlockingStrategy::fixed_size(2))
            .summarization_strategy(SummarizationStrategy::fixed_size(3));

        // Build the inverted index
        index = InvertedIndex<float>::build(dataset, config);
    }

    std::vector<std::pair<std::vector<uint16_t>, std::vector<float>>> data;
    SparseDatasetMut<float> mutable_dataset;
    SparseDataset<float> dataset;
    Configuration config;
    InvertedIndex<float> index;
};

TEST_F(InvertedIndexTest, BasicProperties) {
    // Test index properties
    EXPECT_EQ(index.len(), 5);
    EXPECT_EQ(index.dim(), 5);
    EXPECT_FALSE(index.is_empty());
    EXPECT_EQ(index.knn_len(), 0); // No KNN graph yet

    // Test space usage
    EXPECT_GT(index.space_usage_byte(), 0);
}

TEST_F(InvertedIndexTest, Search) {
    // Create a query
    std::vector<uint16_t> query_components = {0, 2, 4};
    std::vector<float> query_values = {1.0f, 2.0f, 3.0f};

    // Search with the query
    auto results = index.search(query_components, query_values, 2, 3, 0.7f, 0, false);

    // We should get 2 results
    EXPECT_EQ(results.size(), 2);

    // The first document should be the most similar (which is document 0 in this case)
    EXPECT_EQ(results[0].second, 0);
}

TEST_F(InvertedIndexTest, ConfigurationBuilders) {
    // Test configuration builders
    Configuration config;

    // Test pruning strategy
    config.pruning_strategy(PruningStrategy::fixed_size(100));
    EXPECT_EQ(config.get_pruning().get_type(), PruningStrategy::Type::FixedSize);
    EXPECT_EQ(config.get_pruning().get_n_postings(), 100);

    config.pruning_strategy(PruningStrategy::global_threshold(200, 1.5f));
    EXPECT_EQ(config.get_pruning().get_type(), PruningStrategy::Type::GlobalThreshold);
    EXPECT_EQ(config.get_pruning().get_n_postings(), 200);
    EXPECT_FLOAT_EQ(config.get_pruning().get_max_fraction(), 1.5f);

    // Test blocking strategy
    config.blocking_strategy(BlockingStrategy::fixed_size(50));
    EXPECT_EQ(config.get_blocking().get_type(), BlockingStrategy::Type::FixedSize);
    EXPECT_EQ(config.get_blocking().get_block_size(), 50);

    config.blocking_strategy(BlockingStrategy::random_kmeans(0.2f, 3, 
        ClusteringAlgorithm::random_kmeans()));
    EXPECT_EQ(config.get_blocking().get_type(), BlockingStrategy::Type::RandomKmeans);
    EXPECT_FLOAT_EQ(config.get_blocking().get_centroid_fraction(), 0.2f);
    EXPECT_EQ(config.get_blocking().get_min_cluster_size(), 3);
    EXPECT_EQ(config.get_blocking().get_clustering_algorithm().get_type(), 
              ClusteringAlgorithm::Type::RandomKmeans);

    // Test summarization strategy
    config.summarization_strategy(SummarizationStrategy::fixed_size(30));
    EXPECT_EQ(config.get_summarization().get_type(), SummarizationStrategy::Type::FixedSize);
    EXPECT_EQ(config.get_summarization().get_n_components(), 30);

    config.summarization_strategy(SummarizationStrategy::energy_preserving(0.75f));
    EXPECT_EQ(config.get_summarization().get_type(), SummarizationStrategy::Type::EnergyPreserving);
    EXPECT_FLOAT_EQ(config.get_summarization().get_summary_energy(), 0.75f);

    // Test KNN configuration
    config.knn(KnnConfiguration(10));
    EXPECT_EQ(config.get_knn_config().get_nknn(), 10);
    EXPECT_FALSE(config.get_knn_config().get_knn_path().has_value());

    // Test batched indexing
    config.batched_indexing(std::optional<size_t>(1000));
    EXPECT_TRUE(config.get_batched_indexing().has_value());
    EXPECT_EQ(config.get_batched_indexing().value(), 1000);
}

TEST_F(InvertedIndexTest, KnnGraph) {
    // Create a small KNN graph with 2 neighbors per vector
    Knn knn = Knn::new_from_index(index, 2);

    // Add the KNN graph to the index
    index.add_knn(knn);

    // Check that the KNN graph was added
    EXPECT_EQ(index.knn_len(), 2);
    EXPECT_TRUE(index.knn_graph().has_value());

    // Search with the KNN graph
    std::vector<uint16_t> query_components = {0, 2, 4};
    std::vector<float> query_values = {1.0f, 2.0f, 3.0f};

    // Search with the query and use KNN refinement
    auto results = index.search(query_components, query_values, 3, 3, 0.7f, 2, false);

    // We should get 3 results
    EXPECT_EQ(results.size(), 3);
}

TEST_F(InvertedIndexTest, PostingListPacking) {
    // Test packing and unpacking of offset and length
    size_t offset = 12345;
    size_t len = 67;
    uint64_t packed = PostingList::pack_offset_len(offset, len);

    // Unpack and verify
    auto [unpacked_offset, unpacked_len] = PostingList::unpack_offset_len(packed);
    EXPECT_EQ(unpacked_offset, offset);
    EXPECT_EQ(unpacked_len, len);

    // Test with maximum values
    size_t max_offset = (1ULL << 48) - 1;
    size_t max_len = 65535; // 2^16 - 1
    uint64_t max_packed = PostingList::pack_offset_len(max_offset, max_len);

    auto [max_unpacked_offset, max_unpacked_len] = PostingList::unpack_offset_len(max_packed);
    EXPECT_EQ(max_unpacked_offset, max_offset);
    EXPECT_EQ(max_unpacked_len, max_len);
}

TEST_F(InvertedIndexTest, PostingListBuild) {
    // Create a simple set of postings for component 0
    std::vector<std::pair<float, size_t>> postings = {
        {1.0f, 0}, {1.5f, 4}, {1.0f, 2}
    };
    
    // Build a posting list using fixed size blocking and summarization
    Configuration config;
    config.blocking_strategy(BlockingStrategy::fixed_size(2))
          .summarization_strategy(SummarizationStrategy::fixed_size(3));
    
    PostingList posting_list = PostingList::build(dataset, postings, config);
    
    // Check space usage
    EXPECT_GT(posting_list.space_usage_byte(), 0);
}

TEST_F(InvertedIndexTest, PostingListSearch) {
    // Create a simple set of postings
    std::vector<std::pair<float, size_t>> postings = {
        {1.0f, 0}, {1.5f, 4}, {1.0f, 2}, {4.0f, 1}, {2.5f, 3}
    };
    
    // Build a posting list using fixed size blocking and summarization
    Configuration config;
    config.blocking_strategy(BlockingStrategy::fixed_size(2))
          .summarization_strategy(SummarizationStrategy::fixed_size(3));
    
    PostingList posting_list = PostingList::build(dataset, postings, config);
    
    // Create a query
    std::vector<float> query(5, 0.0f);
    query[0] = 1.0f; query[2] = 2.0f; query[4] = 3.0f;
    
    std::vector<uint16_t> query_components = {0, 2, 4};
    std::vector<float> query_values = {1.0f, 2.0f, 3.0f};
    
    // Create a heap and visited set
    utils::HeapFaiss heap(3);
    std::unordered_set<size_t> visited;
    
    // Search the posting list
    posting_list.search(query, query_components, query_values, 3, 0.7f, heap, visited, dataset, true);
    
    // Check that we got results
    EXPECT_GT(heap.len(), 0);
    
    // Get the top results
    auto results = heap.topk();
    EXPECT_GT(results.size(), 0);
}

TEST_F(InvertedIndexTest, PostingListFixedSizeBlocking) {
    // Create a vector of document IDs
    std::vector<size_t> posting_list = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    
    // Apply fixed size blocking with block size 3
    std::vector<size_t> block_offsets = PostingList::fixed_size_blocking(posting_list, 3);
    
    // Check the block offsets
    EXPECT_EQ(block_offsets.size(), 5); // 4 blocks + 1 final offset
    EXPECT_EQ(block_offsets[0], 0);
    EXPECT_EQ(block_offsets[1], 3);
    EXPECT_EQ(block_offsets[2], 6);
    EXPECT_EQ(block_offsets[3], 9);
    EXPECT_EQ(block_offsets[4], 10);
}

TEST_F(InvertedIndexTest, PostingListFixedSizeSummary) {
    // Create a block of document IDs
    std::vector<size_t> block = {0, 2, 4};
    
    // Apply fixed size summary
    auto [components, values] = PostingList::fixed_size_summary(dataset, block, 2);
    
    // Check the summary
    EXPECT_EQ(components.size(), 2);
    EXPECT_EQ(values.size(), 2);
}

TEST_F(InvertedIndexTest, PostingListEnergyPreservingSummary) {
    // Create a block of document IDs
    std::vector<size_t> block = {0, 2, 4};
    
    // Apply energy preserving summary with 80% energy preservation
    auto [components, values] = PostingList::energy_preserving_summary(dataset, block, 0.8f);
    
    // Check the summary
    EXPECT_GT(components.size(), 0);
    EXPECT_EQ(components.size(), values.size());
}

TEST_F(InvertedIndexTest, RandomKmeansBlocking) {
    // Create a vector of document IDs
    std::vector<size_t> posting_list = {0, 1, 2, 3, 4};
    std::vector<size_t> original_posting_list = posting_list;
    
    // Apply random kmeans blocking
    ClusteringAlgorithm clustering_algorithm = ClusteringAlgorithm::random_kmeans();
    std::vector<size_t> block_offsets = PostingList::blocking_with_random_kmeans(
        posting_list, 0.4f, 1, dataset, clustering_algorithm);
    
    // Check the block offsets
    EXPECT_GT(block_offsets.size(), 1); // At least one block + final offset
    EXPECT_EQ(block_offsets[0], 0);
    EXPECT_EQ(block_offsets.back(), 5);
    
    // Check that the posting list was reordered but contains the same elements
    EXPECT_EQ(posting_list.size(), original_posting_list.size());
    std::sort(posting_list.begin(), posting_list.end());
    std::sort(original_posting_list.begin(), original_posting_list.end());
    EXPECT_EQ(posting_list, original_posting_list);
}

TEST_F(InvertedIndexTest, InvertedIndexBuildWithDifferentConfig) {
    // Create a configuration with different strategies
    Configuration config;
    config.pruning_strategy(PruningStrategy::global_threshold(10, 1.5f))
          .blocking_strategy(BlockingStrategy::random_kmeans(0.3f, 1, ClusteringAlgorithm::random_kmeans()))
          .summarization_strategy(SummarizationStrategy::energy_preserving(0.8f));
    
    // Build the inverted index
    InvertedIndex<float> custom_index = InvertedIndex<float>::build(dataset, config);
    
    // Check basic properties
    EXPECT_EQ(custom_index.len(), 5);
    EXPECT_EQ(custom_index.dim(), 5);
    EXPECT_FALSE(custom_index.is_empty());
    
    // Search with a query
    std::vector<uint16_t> query_components = {0, 2, 4};
    std::vector<float> query_values = {1.0f, 2.0f, 3.0f};
    
    auto results = custom_index.search(query_components, query_values, 2, 3, 0.7f, 0, false);
    
    // We should get 2 results
    EXPECT_EQ(results.size(), 2);
}

// Main function is provided by main_test.cpp
