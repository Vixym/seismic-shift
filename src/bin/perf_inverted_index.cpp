#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <algorithm>
#include <optional>
#include <cstring>
#include <filesystem>

#include "../inverted_index.h"
#include "../sparse_dataset.h"

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/optional.hpp>

#include "../my_inverted_index.h"

using namespace seismic;
using TIndex = MyInvertedIndex<float>;
using TDataset = SparseDatasetMut<float>;

// Simple argument parser
struct Args
{
    std::optional<std::string> index_file;
    std::optional<std::string> query_file;
    std::optional<std::string> output_path;
    size_t n_queries = 10000;
    size_t k = 10;
    size_t n_runs = 1;
    size_t query_cut = 10;
    float heap_factor = 0.7f;
    size_t n_knn = 0;
    bool first_sorted = false;
    float alpha = 0.0f;  // centroid+radius pruning strength (0 = legacy count-based skip)
};

void print_usage()
{
    std::cout << "Usage: perf_inverted_index [OPTIONS]\n"
              << "Options:\n"
              << "  --index-file FILE             Path of the index\n"
              << "  --query-file FILE             Query file\n"
              << "  --output-path FILE            Output file to write the results\n"
              << "  --n-queries N                 Number of queries to evaluate (default: 10000)\n"
              << "  --k N                         Number of top-k results to retrieve (default: 10)\n"
              << "  --n-runs N                    Number of runs to perform (default: 1)\n"
              << "  --query-cut N                 Query cut parameter (default: 10)\n"
              << "  --heap-factor F               Heap factor (default: 0.7)\n"
              << "  --n-knn N                     Number of KNN (default: 0)\n"
              << "  --first-sorted                Whether to sort by estimated dot products (default: false)\n";
}

Args parse_args(int argc, char *argv[])
{
    Args args;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "--index-file" && i + 1 < argc)
        {
            args.index_file = argv[++i];
        }
        else if (arg == "--query-file" && i + 1 < argc)
        {
            args.query_file = argv[++i];
        }
        else if (arg == "--output-path" && i + 1 < argc)
        {
            args.output_path = argv[++i];
        }
        else if (arg == "--n-queries" && i + 1 < argc)
        {
            args.n_queries = std::stoul(argv[++i]);
        }
        else if (arg == "--k" && i + 1 < argc)
        {
            args.k = std::stoul(argv[++i]);
        }
        else if (arg == "--n-runs" && i + 1 < argc)
        {
            args.n_runs = std::stoul(argv[++i]);
        }
        else if (arg == "--query-cut" && i + 1 < argc)
        {
            args.query_cut = std::stoul(argv[++i]);
        }
        else if (arg == "--heap-factor" && i + 1 < argc)
        {
            args.heap_factor = std::stof(argv[++i]);
        }
        else if (arg == "--n-knn" && i + 1 < argc)
        {
            args.n_knn = std::stoul(argv[++i]);
        }
        else if (arg == "--first-sorted")
        {
            args.first_sorted = true;
        }
        else if (arg == "--alpha" && i + 1 < argc)
        {
            args.alpha = std::stof(argv[++i]);
        }
        else if (arg == "--help")
        {
            print_usage();
            exit(0);
        }
    }

    if (!args.index_file || !args.query_file || !args.output_path)
    {
        std::cerr << "Error: Index file, query file, and output path must be specified.\n";
        print_usage();
        exit(1);
    }

    return args;
}

// Function to load the index from a file
TIndex load_index(const std::string &path)
{
    std::string full_path = path + ".index.seismic";
    std::cout << "Loading index from " << full_path << "..." << std::endl;

    std::ifstream file(full_path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open file for reading: " + full_path);
    }

    // Create an empty index
    TIndex index;

    try
    {
        // Deserialize using cereal
        cereal::BinaryInputArchive archive(file);
        archive(index); // This will deserialize the entire index in one line

        std::cout << "Index loaded successfully" << std::endl;
        std::cout << "Number of documents: " << index.len() << std::endl;
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("Failed to deserialize index: " + std::string(e.what()));
    }

    return index;
}

// Function to write results to a file
void write_results(const std::string &path, const std::vector<std::vector<std::pair<float, size_t>>> &results)
{
    std::cout << "Writing results to " << path << "..." << std::endl;

    std::ofstream file(path);
    if (!file)
    {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    for (size_t query_id = 0; query_id < results.size(); ++query_id)
    {
        const auto &result = results[query_id];
        for (size_t idx = 0; idx < result.size(); ++idx)
        {
            const auto &[score, doc_id] = result[idx];
            file << query_id << "\t" << doc_id << "\t" << (idx + 1) << "\t" << score << std::endl;
        }
    }

    std::cout << "Results written successfully" << std::endl;
}

int main(int argc, char *argv[])
{
    // Parse command-line arguments
    Args args = parse_args(argc, argv);

    std::cout << "Performance testing with the following parameters:" << std::endl;
    std::cout << "  Index file: " << *args.index_file << std::endl;
    std::cout << "  Query file: " << *args.query_file << std::endl;
    std::cout << "  Output path: " << *args.output_path << std::endl;
    std::cout << "  N queries: " << args.n_queries << std::endl;
    std::cout << "  K: " << args.k << std::endl;
    std::cout << "  N runs: " << args.n_runs << std::endl;
    std::cout << "  Query cut: " << args.query_cut << std::endl;
    std::cout << "  Heap factor: " << args.heap_factor << std::endl;
    std::cout << "  N KNN: " << args.n_knn << std::endl;
    std::cout << "  First sorted: " << (args.first_sorted ? "true" : "false") << std::endl;

    try
    {
        // Load the index
        TIndex inverted_index = load_index(*args.index_file);

        // Load queries
        std::cout << "\nLoading queries from " << *args.query_file << "..." << std::endl;
        TDataset queries;

        try
        {
            queries = TDataset::read_bin_file(*args.query_file);
            std::cout << "Number of Queries: " << queries.len() << std::endl;
            std::cout << "Number of Dimensions: " << queries.dim() << std::endl;
            std::cout << "Avg number of components: "
                      << static_cast<float>(queries.nnz()) / queries.len() << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error loading queries: " << e.what() << std::endl;
            return 1;
        }

        // Limit the number of queries to evaluate
        size_t n_queries = std::min(args.n_queries, queries.len());

        std::cout << "Searching for top-" << args.k << " results" << std::endl;
        std::cout << "Number of evaluated queries: " << n_queries << std::endl;
        std::cout << "Number of documents: " << inverted_index.len() << std::endl;
        std::cout << "Avg number of non-zero components: "
                  << static_cast<float>(inverted_index.nnz()) / inverted_index.len() << std::endl;

        // Run the search
        std::vector<std::vector<std::pair<float, size_t>>> results;
        results.reserve(n_queries);

        auto query_start = std::chrono::high_resolution_clock::now();
        for (size_t run = 0; run < args.n_runs; ++run)
        {
            results.clear();
            for (size_t query_id = 0; query_id < n_queries; ++query_id)
            {
                const auto &query = queries.get(query_id);

                const auto &q_components = query.first;
                const auto &q_values = query.second;

                auto cur_results = inverted_index.search(
                    q_components,
                    q_values,
                    args.k,
                    args.query_cut,
                    args.heap_factor,
                    args.n_knn,
                    args.first_sorted,
                    args.alpha,
                    query_id % 1000 == 1);

                if (cur_results.size() < args.k)
                {   
                    std::cout << "FAIL! The query " << query_id
                              << " has only " << cur_results.size() << " results." << std::endl;
                    return 0;
                }

                results.push_back(cur_results);
            }
        }

        auto query_end = std::chrono::high_resolution_clock::now();
        auto query_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                           query_end - query_start)
                           .count();

        std::cout << "Time " << query_elapsed / (args.n_runs * n_queries) << " microsecs per query" << std::endl;
        std::cerr << query_elapsed << std::endl;

        // Write results to file
        write_results(*args.output_path, results);

        // auto insert_start = std::chrono::high_resolution_clock::now();
        // for (size_t run = 0; run < 100; ++run)
        // {
        //     const auto &query = queries.get(run);
        //     const auto &q_components = query.first;
        //     const auto &q_values = query.second;

        //     inverted_index.insert_doc(
        //         q_components,
        //         q_values);
        // }

        // auto insert_end = std::chrono::high_resolution_clock::now();
        // auto insert_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        //                    insert_end - insert_start)
        //                    .count();
        // std::cout << "Time " << insert_elapsed / 1000 << " microsecs per insert_document" << std::endl;
        // std::cerr << insert_elapsed << std::endl;
        
        // auto start_time = std::chrono::high_resolution_clock::now();

        // for (size_t run = 0; run < 1000; ++run)
        // {
        //     inverted_index.delete_doc(run);
        // }

        // inverted_index.resize();

        // auto end_time = std::chrono::high_resolution_clock::now();
        // auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        //                    end_time - start_time)
        //                    .count();

        // std::cout << "Time " << elapsed / 1000 << " microsecs per delete_document" << std::endl;
        // std::cerr << elapsed << std::endl;

        // Print space usage
        inverted_index.print_space_usage_byte();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
