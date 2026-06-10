#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <optional>
#include <cstring>

#include "../inverted_index.h"
#include "../sparse_dataset.h"

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/optional.hpp>

// CEREAL_REGISTER_TYPE(InvertedIndex);
// CEREAL_REGISTER_TYPE(seismic::ClusteringAlgorithm);
// CEREAL_REGISTER_TYPE(seismic::PruningStrategy);
// CEREAL_REGISTER_TYPE(seismic::BlockingStrategy);
// CEREAL_REGISTER_TYPE(seismic::SummarizationStrategy);
// CEREAL_REGISTER_TYPE(seismic::KnnConfiguration);
// CEREAL_REGISTER_TYPE(seismic::Configuration);
// CEREAL_REGISTER_TYPE(seismic::SpaceUsage);
// CEREAL_REGISTER_TYPE(seismic::PostingList);
// CEREAL_REGISTER_TYPE(seismic::Knn);
// CEREAL_REGISTER_TYPE(seismic::InvertedIndex);

#include "../my_inverted_index.h"

using namespace seismic;

SparseDatasetMut<float> make_test_dataset_10() {
    SparseDatasetMut<float> ds;   // default ctor

    auto add_doc = [&](std::initializer_list<std::pair<uint16_t, float>> nz) {
        std::vector<uint16_t> comps;
        std::vector<float>    vals;
        comps.reserve(nz.size());
        vals.reserve(nz.size());
        for (auto [idx, val] : nz) {
            comps.push_back(idx);
            vals.push_back(val);
        }
        ds.push(comps, vals);
    };

    // Cluster A: around [1, 0]
    add_doc({{0, 1.0f}});              // doc 0
    add_doc({{0, 0.95f}, {1, 0.05f}}); // doc 1
    add_doc({{0, 0.9f},  {1, 0.1f}});  // doc 2
    add_doc({{0, 1.1f}});              // doc 3
    add_doc({{0, 0.85f}, {1, 0.15f}}); // doc 4

    // Cluster B: around [0, 1]
    add_doc({{1, 1.0f}});              // doc 5
    add_doc({{1, 0.95f}, {0, 0.05f}}); // doc 6
    add_doc({{1, 0.9f},  {0, 0.1f}});  // doc 7
    add_doc({{1, 1.1f}});              // doc 8
    add_doc({{1, 0.85f}, {0, 0.15f}}); // doc 9

    return ds;
}

// --- Helper: build a tiny toy dataset ---
// 3 docs in a low-dimensional space.
// doc 0 ~ [1, 0, 0, 0]
// doc 1 ~ [0, 1, 0, 0]
// doc 2 ~ [0.9, 0.1, 0, 0] (should be closest to doc 0)
SparseDatasetMut<float> make_tiny_dataset() {
    SparseDatasetMut<float> ds;   // default ctor

    // doc 0:  [1, 0, 0, 0]
    {
        std::vector<uint16_t> comps = {0};
        std::vector<float>    vals  = {1.0f};
        ds.push(comps, vals);
    }

    // doc 1:  [0, 1, 0, 0]
    {
        std::vector<uint16_t> comps = {1};
        std::vector<float>    vals  = {1.0f};
        ds.push(comps, vals);
    }

    // doc 2:  [0.9, 0.1, 0, 0]
    {
        std::vector<uint16_t> comps = {0, 1};
        std::vector<float>    vals  = {0.9f, 0.1f};
        ds.push(comps, vals);
    }

    return ds;
}

// Optional: tiny helper to pretty-print a result list
template <typename Result>
void print_results(const std::vector<Result>& results) {
    std::cout << "Top-k results:\n";
    for (size_t i = 0; i < results.size(); ++i) {
        // <<< ADJUST: field names of your result type
        // Example 1: if Result = std::pair<size_t, float> (doc_id, score)
        size_t doc_id = results[i].first;
        float score   = results[i].second;

        // Example 2: if Result is a struct:
        // size_t doc_id = results[i].doc_id;
        // float score   = results[i].score;

        std::cout << "  rank " << i
                  << " -> doc_id=" << doc_id
                  << " score=" << score << "\n";
    }
}

int main() {
    // 1. Build tiny dataset
    auto dataset = make_tiny_dataset();
    const size_t n_docs = dataset.len();   // <<< ADJUST if method is size() instead of len()
    const size_t dim    = dataset.dim();   // <<< ADJUST if method differs

    std::cout << "Tiny dataset built: n_docs=" << n_docs
              << " dim=" << dim << "\n";

    // 2. Build a minimal Configuration
    // Create configuration using builder pattern
    Configuration config;
    
    // Set pruning strategy
    PruningStrategy pruning = PruningStrategy::fixed_size(4000);
    config.pruning_strategy(pruning);
    
    // Set blocking strategy
    BlockingStrategy blocking = BlockingStrategy::random_kmeans(
        .1, 1000, ClusteringAlgorithm::random_kmeans());
    config.blocking_strategy(blocking);
    
    // Set summarization strategy
    SummarizationStrategy summarization = SummarizationStrategy::energy_preserving(.1);
    config.summarization_strategy(summarization);

    // 3. Build JLT index on tiny dataset
    std::cout << "Building JLT inverted index on tiny dataset...\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    // Static build interface (this is from your error logs):
    //   static JltInvertedIndex<T> build(const SparseDatasetMut<T>&, const Configuration&)
    auto index = MyInvertedIndex<float>::build(dataset, config);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> build_time = t1 - t0;
    std::cout << "Index build finished in " << build_time.count() << " seconds.\n";

    // 4. Create a tiny sparse query vector
    // Query ~ [0.95, 0.05, 0, 0], so it should retrieve doc 2 or doc 0 as closest
    std::vector<uint16_t> q_components = {0, 1};
    std::vector<float>    q_values     = {0.95f, 0.05f};

    const size_t k = 3;

    // 5. Run query against your JLT index
    std::cout << "Running top-" << k << " query...\n";

    auto tq0 = std::chrono::high_resolution_clock::now();

    // <<< ADJUST: query API
    // Examples:
    //   auto results = index.search_top_k(q_components, q_values, k);
    //   auto results = index.knn_query(q_components, q_values, k);
    //   auto results = index.query(q_components, q_values, k);
    auto results = index.search(
        q_components,
        q_values,
        1,
        .3,
        .3,
        5,
        false); // change name if needed

    auto tq1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> query_time = tq1 - tq0;

    std::cout << "Query finished in " << query_time.count() << " seconds.\n";

    // 6. Print results
    print_results(results);

    // 7. Optional: simple correctness sanity checks (replace with actual expectations)
    // For example, assert we got at least one result and doc 0 or 2 is on top.
    if (!results.empty()) {
        size_t best_doc_id = results[0].first; // <<< ADJUST field if needed
        std::cout << "Best doc id: " << best_doc_id << "\n";
    }

    return 0;
}