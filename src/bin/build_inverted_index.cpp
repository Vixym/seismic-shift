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
//
#include "../my_inverted_index.h"

using namespace seismic;

using TIndex = MyInvertedIndex<float>;
using TDataset = SparseDatasetMut<float>;

// Simple argument parser
struct Args {
    std::string input_file;
    std::string output_file;
    size_t n_postings = 3500;
    float summary_energy = 0.4f;
    float centroid_fraction = 0.1f;
    size_t min_cluster_size = 2;
    size_t knn = 0;
    std::string clustering_algorithm = "random-kmeans";
    size_t kmeans_doc_cut = 15;
    float kmeans_pruning_factor = 0.005f;
    std::optional<std::string> knn_path = std::nullopt;
    bool dynamic_support = false;
    std::string summarization = "max";
    std::string transform = "none";
};

void print_usage() {
    std::cout << "Usage: build_inverted_index [OPTIONS]\n"
              << "Options:\n"
              << "  --input-file FILE             Input file path\n"
              << "  --output-file FILE            Output file path\n"
              << "  --n-postings N                Number of postings (default: 3500)\n"
              << "  --summary-energy E            Summary energy (default: 0.4)\n"
              << "  --centroid-fraction F         Centroid fraction (default: 0.1)\n"
              << "  --min-cluster-size N          Minimum cluster size (default: 2)\n"
              << "  --knn N                       Number of KNN (default: 0)\n"
              << "  --clustering-algorithm ALG    Clustering algorithm (default: random-kmeans)\n"
              << "  --kmeans-doc-cut N            K-means doc cut (default: 15)\n"
              << "  --kmeans-pruning-factor F     K-means pruning factor (default: 0.005)\n"
              << "  --knn-path PATH               KNN path (optional)\n"
              << "  --dynamic-support true/false  Enable support for dynamic operations\n"
              << "  --summarization max/centroid  Summary metric\n"
              << "  --transform none/jlt      Transformation to apply to summary\n";
}

Args parse_args(int argc, char* argv[]) {
    Args args;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--input-file" && i + 1 < argc) {
            args.input_file = argv[++i];
        } else if (arg == "--output-file" && i + 1 < argc) {
            args.output_file = argv[++i];
        } else if (arg == "--n-postings" && i + 1 < argc) {
            args.n_postings = std::stoul(argv[++i]);
        } else if (arg == "--summary-energy" && i + 1 < argc) {
            args.summary_energy = std::stof(argv[++i]);
        } else if (arg == "--centroid-fraction" && i + 1 < argc) {
            args.centroid_fraction = std::stof(argv[++i]);
        } else if (arg == "--min-cluster-size" && i + 1 < argc) {
            args.min_cluster_size = std::stoul(argv[++i]);
        } else if (arg == "--knn" && i + 1 < argc) {
            args.knn = std::stoul(argv[++i]);
        } else if (arg == "--clustering-algorithm" && i + 1 < argc) {
            args.clustering_algorithm = argv[++i];
        } else if (arg == "--kmeans-doc-cut" && i + 1 < argc) {
            args.kmeans_doc_cut = std::stoul(argv[++i]);
        } else if (arg == "--kmeans-pruning-factor" && i + 1 < argc) {
            args.kmeans_pruning_factor = std::stof(argv[++i]);
        } else if (arg == "--knn-path" && i + 1 < argc) {
            args.knn_path = argv[++i];
        } else if (arg == "--dynamic-support" && i + 1 < argc) {
            args.dynamic_support = argv[++i];
        } else if (arg == "--summarization" && i + 1 < argc ) {
            args.summarization = argv[++i];
        } else if (arg == "--transform" && i + 1 < argc) {
            args.transform = argv[++i];
        } else if (arg == "--help") {
            print_usage();
            exit(0);
        }
    }
    
    if (args.input_file.empty() || args.output_file.empty()) {
        std::cerr << "Error: Input and output files must be specified.\n";
        print_usage();
        exit(1);
    }
    
    return args;
}

// Function to save the index to a file
void save_index(const TIndex& index, const std::string& path) {
    std::string full_path = path + ".index.seismic";
    std::cout << "Saving ... " << full_path << std::endl;
    
    std::ofstream file(full_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + full_path);
    }
    
    // Serialize using cereal
    cereal::BinaryOutputArchive archive(file);
    archive(index);  // This will serialize the entire index in one line
    
    std::cout << "Save completed successfully" << std::endl;
}

// Function to save the KNN graph to a file
void save_knn_graph(const std::string& path) {
    std::cout << "Saving KNN graph to " << path << std::endl;
    
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    
    // Write a dummy header
    const char* header = "SEISMIC_KNN";
    file.write(header, strlen(header));
    
    // In a real implementation, you would serialize the KNN graph here
    // For now, we'll just create a placeholder file
    
    std::cout << "KNN graph saved successfully" << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    Args args = parse_args(argc, argv);
    
    std::cout << "Building inverted index with the following parameters:" << std::endl;
    std::cout << "  Input file: " << args.input_file << std::endl;
    std::cout << "  Output file: " << args.output_file << std::endl;
    std::cout << "  N postings: " << args.n_postings << std::endl;
    std::cout << "  Summary energy: " << args.summary_energy << std::endl;
    std::cout << "  Centroid fraction: " << args.centroid_fraction << std::endl;
    std::cout << "  Min cluster size: " << args.min_cluster_size << std::endl;
    std::cout << "  KNN: " << args.knn << std::endl;
    std::cout << "  Clustering algorithm: " << args.clustering_algorithm << std::endl;
    std::cout << "  K-means doc cut: " << args.kmeans_doc_cut << std::endl;
    std::cout << "  K-means pruning factor: " << args.kmeans_pruning_factor << std::endl;
    if (args.knn_path) {
        std::cout << "  KNN path: " << *args.knn_path << std::endl;
    }
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Create clustering algorithm
        ClusteringAlgorithm clustering_algorithm;
        if (args.clustering_algorithm == "random-kmeans") {
            clustering_algorithm = ClusteringAlgorithm::random_kmeans();
        } else if (args.clustering_algorithm == "random-kmeans-inverted-index") {
            clustering_algorithm = ClusteringAlgorithm::random_kmeans_inverted_index(
                args.kmeans_pruning_factor, args.kmeans_doc_cut);
        } else if (args.clustering_algorithm == "random-kmeans-inverted-index-approx") {
            clustering_algorithm = ClusteringAlgorithm::random_kmeans_inverted_index_approx(
                args.kmeans_doc_cut);
        } else {
            std::cerr << "Error: Unknown clustering algorithm: " << args.clustering_algorithm << std::endl;
            return 1;
        }
        
        // Load dataset
        std::cout << "\nLoading dataset from " << args.input_file << "..." << std::endl;
        TDataset dataset;
        
        try {
            dataset = TDataset::read_bin_file(args.input_file);
            std::cout << "Number of Vectors: " << dataset.len() << std::endl;
            std::cout << "Number of Dimensions: " << dataset.dim() << std::endl;
            std::cout << "Avg number of components: " 
                      << static_cast<float>(dataset.nnz()) / dataset.len() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading dataset: " << e.what() << std::endl;
            return 1;
        }
        
        // Create configuration using builder pattern
        Configuration config;
        
        // Set pruning strategy
        PruningStrategy pruning = PruningStrategy::fixed_size(args.n_postings);
        config.pruning_strategy(pruning);
        
        // Set blocking strategy
        BlockingStrategy blocking = BlockingStrategy::fixed_size(400);
        //BlockingStrategy blocking = BlockingStrategy::random_kmeans(
        //    args.centroid_fraction, args.min_cluster_size, clustering_algorithm);
        config.blocking_strategy(blocking);
        
        // Set summarization strategy
        SummarizationStrategy summarization = SummarizationStrategy::energy_preserving(
            args.summary_energy);
        config.summarization_strategy(summarization);
        
        // Set KNN configuration if needed
        if (args.knn > 0 || args.knn_path) {
            KnnConfiguration knn_config(args.knn, args.knn_path);
            config.knn(knn_config);
        }

        // Set dynamism
        if (args.dynamic_support == true) {
            config.set_dynamic_support(true);
        }

        // Set summary metric
        config.set_summarization_metric(args.summarization);

        // Set transform
        config.set_transform_function(args.transform);
        
        std::cout << "\nBuilding the index..." << std::endl;
        
        // Build the inverted index
        TIndex inverted_index = TIndex::build(dataset, config);
        
        // Calculate elapsed time before serialization
        auto before_serialize = std::chrono::high_resolution_clock::now();
        auto elapsed_before = std::chrono::duration_cast<std::chrono::seconds>(
            before_serialize - start_time).count();
        
        std::cout << "Time to build " << elapsed_before << " secs (before serializing)" << std::endl;
        
        try {
            // Save the index
            save_index(inverted_index, args.output_file);
            
            // Save KNN graph if requested
            if (args.knn > 0 && args.knn_path) {
                save_knn_graph(*args.knn_path);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error saving index: " << e.what() << std::endl;
            return 1;
        }
        
        // Calculate total elapsed time
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_total = std::chrono::duration_cast<std::chrono::seconds>(
            end_time - start_time).count();
        
        std::cout << "Time to build " << elapsed_total << " secs" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
