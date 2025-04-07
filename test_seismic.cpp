#include <iostream>
#include <vector>
#include <string>
#include "src/seismic_index.h"

int main() {
    std::cout << "Testing Seismic C++ Library" << std::endl;
    
    // Create a simple dataset
    seismic::SparseDataset<float> dataset;
    
    // Add a few vectors to the dataset
    std::vector<uint16_t> components1 = {1, 2, 3};
    std::vector<float> values1 = {0.1f, 0.2f, 0.3f};
    dataset.add(components1, values1);
    
    std::vector<uint16_t> components2 = {2, 3, 4};
    std::vector<float> values2 = {0.2f, 0.3f, 0.4f};
    dataset.add(components2, values2);
    
    // Print dataset information
    std::cout << "Dataset size: " << dataset.len() << std::endl;
    std::cout << "Dataset dimension: " << dataset.dim() << std::endl;
    
    // Create a configuration
    seismic::Configuration config;
    config.n_postings = 10;
    config.centroid_fraction = 0.1;
    config.min_cluster_size = 2;
    config.summary_energy = 0.4;
    
    // Create an index
    std::cout << "Creating index..." << std::endl;
    std::optional<std::vector<std::string>> doc_mapping = std::nullopt;
    std::unordered_map<std::string, size_t> token_map;
    
    auto index = seismic::SeismicIndex<float>::new_from_dataset(
        dataset, config, doc_mapping, token_map
    );
    
    // Print index information
    std::cout << "Index size: " << index.len() << std::endl;
    std::cout << "Index dimension: " << index.dim() << std::endl;
    
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}
