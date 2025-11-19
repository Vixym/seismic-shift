#ifndef SEISMIC_INDEX_H
#define SEISMIC_INDEX_H

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <optional>
#include <fstream>
#include <iostream>
#include <chrono>
#include <sstream>

// Remove libarchive includes
// #include <archive.h>
// #include <archive_entry.h>

#include "inverted_index.h"
#include "sparse_dataset.h"
#include "space_usage.h"
#include "data_type.h"

#include "dynamic_inverted_index.h"

namespace seismic
{
    // TODO: return to space usage calculation here

    /**
     * A wrapper class for InvertedIndex that adds document mapping functionality
     * and token-to-id mapping.
     */
    template <typename T>
    class SeismicIndex : public SpaceUsage
    {

    private:
        InvertedIndex<T> inverted_index_;
        std::optional<std::vector<std::string>> document_mapping_;
        std::unordered_map<std::string, size_t> token_to_id_map_;

    public:
        // Default constructor
        SeismicIndex() = default;

        // Constructor with all components
        SeismicIndex(
            InvertedIndex<T> inverted_index,
            std::optional<std::vector<std::string>> document_mapping,
            std::unordered_map<std::string, size_t> token_to_id_map) : inverted_index_(std::move(inverted_index)),
                                                                       document_mapping_(std::move(document_mapping)),
                                                                       token_to_id_map_(std::move(token_to_id_map)) {}

        // Create a new index from a dataset and configuration
        static SeismicIndex<T> new_from_dataset(
            const SparseDataset<T> &dataset,
            const Configuration &config,
            std::optional<std::vector<std::string>> document_mapping,
            std::unordered_map<std::string, size_t> token_to_id_map)
        {
            auto inverted_index = InvertedIndex<T>::build(dataset, config);
            return SeismicIndex<T>(
                std::move(inverted_index),
                std::move(document_mapping),
                std::move(token_to_id_map));
        }

        // Helper function to remap document IDs to their string representations
        std::vector<std::tuple<std::string, float, std::string>> remap_doc_ids(
            const std::vector<std::pair<float, size_t>> &plain_results,
            const std::string &query_id) const
        {
            std::vector<std::tuple<std::string, float, std::string>> remapped_results;
            remapped_results.reserve(plain_results.size());

            for (const auto &[distance, doc_id] : plain_results)
            {
                std::string doc_name;
                if (document_mapping_)
                {
                    doc_name = (*document_mapping_)[doc_id];
                }
                else
                {
                    doc_name = std::to_string(doc_id);
                }
                remapped_results.emplace_back(query_id, distance, std::move(doc_name));
            }

            return remapped_results;
        }

        // Search functionality that returns raw results (document IDs and scores)
        std::vector<std::pair<float, size_t>> search_raw(
            const std::vector<std::string> &query_components_original,
            const std::vector<float> &query_values,
            size_t k,
            size_t query_cut,
            float heap_factor,
            size_t n_knn,
            bool first_sorted) const
        {
            // Convert string components to IDs
            std::vector<uint16_t> filtered_components;
            std::vector<float> filtered_values;
            filtered_components.reserve(query_components_original.size());
            filtered_values.reserve(query_components_original.size());

            for (size_t i = 0; i < query_components_original.size(); ++i)
            {
                auto it = token_to_id_map_.find(query_components_original[i]);
                if (it != token_to_id_map_.end())
                {
                    filtered_components.push_back(static_cast<uint16_t>(it->second));
                    filtered_values.push_back(query_values[i]);
                }
            }

            return inverted_index_.search(
                filtered_components,
                filtered_values,
                k,
                query_cut,
                heap_factor,
                n_knn,
                first_sorted);
        }

        // Search functionality that returns mapped results (with document names)
        std::vector<std::tuple<std::string, float, std::string>> search(
            const std::string &query_id,
            const std::vector<std::string> &query_components_original,
            const std::vector<float> &query_values,
            size_t k,
            size_t query_cut,
            float heap_factor,
            size_t n_knn,
            bool first_sorted) const
        {
            auto results = search_raw(
                query_components_original,
                query_values,
                k,
                query_cut,
                heap_factor,
                n_knn,
                first_sorted);

            return remap_doc_ids(results, query_id); // return the documents remapped
        }

        // Helper method to process data from a stream
        static std::tuple<SparseDataset<T>, std::optional<std::vector<std::string>>, std::unordered_map<std::string, size_t>>
        process_data(
            std::istream &input_stream,
            size_t row_count,
            const std::optional<std::unordered_map<std::string, size_t>> &input_token_to_id_map = std::nullopt)
        {
            // Create mappings and dataset
            std::vector<std::string> doc_id_mapping;
            doc_id_mapping.reserve(row_count);

            std::unordered_map<std::string, size_t> token_to_id_map;
            token_to_id_map.reserve(30000); // Reserve space for a reasonable number of tokens

            // Create a mutable dataset to build
            SparseDatasetMut<T> dataset_mut;

            std::string line;
            size_t line_count = 0;

            // Process each line as a JSON document
            while (std::getline(input_stream, line) && line_count < row_count)
            {
                // Simple JSON parsing (in a real implementation, use a proper JSON parser)
                // Format expected: {"id": "doc_id", "tokens": ["token1", "token2", ...], "values": [1.0, 2.0, ...]}

                // Extract document ID
                size_t id_pos = line.find("\"id\"");
                if (id_pos == std::string::npos)
                    continue;

                size_t id_start = line.find("\"", id_pos + 4) + 1;
                size_t id_end = line.find("\"", id_start);
                std::string doc_id = line.substr(id_start, id_end - id_start);
                doc_id_mapping.push_back(doc_id);

                // Extract tokens
                size_t tokens_pos = line.find("\"tokens\"");
                if (tokens_pos == std::string::npos)
                    continue;

                size_t tokens_start = line.find("[", tokens_pos);
                size_t tokens_end = line.find("]", tokens_start);
                std::string tokens_str = line.substr(tokens_start + 1, tokens_end - tokens_start - 1);

                // Extract values
                size_t values_pos = line.find("\"values\"");
                if (values_pos == std::string::npos)
                    continue;

                size_t values_start = line.find("[", values_pos);
                size_t values_end = line.find("]", values_start);
                std::string values_str = line.substr(values_start + 1, values_end - values_start - 1);

                // Parse tokens and values
                std::vector<std::string> tokens;
                std::vector<T> values;

                // Parse tokens
                size_t pos = 0;
                while (pos < tokens_str.size())
                {
                    size_t token_start = tokens_str.find("\"", pos);
                    if (token_start == std::string::npos)
                        break;

                    size_t token_end = tokens_str.find("\"", token_start + 1);
                    if (token_end == std::string::npos)
                        break;

                    std::string token = tokens_str.substr(token_start + 1, token_end - token_start - 1);
                    tokens.push_back(token);

                    pos = token_end + 1;
                }

                // Parse values
                pos = 0;
                while (pos < values_str.size())
                {
                    size_t value_end = values_str.find(",", pos);
                    if (value_end == std::string::npos)
                    {
                        value_end = values_str.size();
                    }

                    std::string value_str = values_str.substr(pos, value_end - pos);
                    // Trim whitespace
                    value_str.erase(0, value_str.find_first_not_of(" \t\n\r\f\v"));
                    value_str.erase(value_str.find_last_not_of(" \t\n\r\f\v") + 1);

                    if (!value_str.empty())
                    {
                        T value = static_cast<T>(std::stof(value_str));
                        values.push_back(value);
                    }

                    if (value_end == values_str.size())
                        break;
                    pos = value_end + 1;
                }

                // Map tokens to IDs
                std::vector<uint16_t> ids;

                if (!input_token_to_id_map)
                {
                    // If no input mapping, create mapping as we go
                    for (const auto &token : tokens)
                    {
                        if (token_to_id_map.find(token) == token_to_id_map.end())
                        {
                            token_to_id_map[token] = token_to_id_map.size();
                        }
                    }

                    // Map tokens to IDs
                    for (const auto &token : tokens)
                    {
                        ids.push_back(static_cast<uint16_t>(token_to_id_map[token]));
                    }
                }
                else
                {
                    // Use provided mapping
                    const auto &valid_mapping = *input_token_to_id_map;
                    for (const auto &token : tokens)
                    {
                        auto it = valid_mapping.find(token);
                        if (it != valid_mapping.end())
                        {
                            ids.push_back(static_cast<uint16_t>(it->second));
                        }
                    }
                }

                // Create pairs and sort by ID
                std::vector<std::pair<uint16_t, T>> converted_vector;
                converted_vector.reserve(std::min(ids.size(), values.size()));

                for (size_t i = 0; i < std::min(ids.size(), values.size()); ++i)
                {
                    converted_vector.emplace_back(ids[i], values[i]);
                }

                // Sort by ID (like the Rust code does)
                std::sort(converted_vector.begin(), converted_vector.end(),
                          [](const auto &a, const auto &b)
                          { return a.first < b.first; });

                // Add to dataset
                if (!converted_vector.empty())
                {
                    dataset_mut.push_pairs(converted_vector);
                }

                line_count++;

                // Print progress every 1000 lines
                if (line_count % 1000 == 0)
                {
                    std::cout << "Processed " << line_count << " / " << row_count << " lines" << std::endl;
                }
            }

            // Convert mutable dataset to immutable
            SparseDataset<T> final_dataset = dataset_mut.to_immutable();

            return {final_dataset, doc_id_mapping, token_to_id_map};
        }

        static SeismicIndex<T> from_file(
            const std::string &file_path,
            const Configuration &config,
            const std::optional<std::unordered_map<std::string, size_t>> &input_token_to_id_map = std::nullopt)
        {
            // Determine file type based on extension
            std::string extension = file_path.substr(file_path.find_last_of(".") + 1);

            if (extension == "jsonl" || extension == "json")
            {
                return from_json(file_path, config, input_token_to_id_map);
            }
            else if (extension == "gz" && file_path.find(".tar.gz") != std::string::npos)
            {
                return from_tar(file_path, config, input_token_to_id_map);
            }
            else
            {
                throw std::runtime_error("Unsupported file type. Supported files: .jsonl, .tar.gz");
            }
        }

        static SeismicIndex<T> from_json(
            const std::string &json_path,
            const Configuration &config,
            const std::optional<std::unordered_map<std::string, size_t>> &input_token_to_id_map = std::nullopt)
        {
            std::cout << "Reading the collection from JSON file: " << json_path << std::endl;
            auto start = std::chrono::high_resolution_clock::now();

            // Open the file
            std::ifstream file(json_path);
            if (!file.is_open())
            {
                throw std::runtime_error("Failed to open file: " + json_path);
            }

            // Count the number of lines
            size_t row_count = 0;
            std::string line;
            while (std::getline(file, line))
            {
                row_count++;
            }

            std::cout << "Number of rows: " << row_count << std::endl;
            std::cout << "Elapsed time to read the number of rows "
                      << std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::high_resolution_clock::now() - start)
                             .count()
                      << " s" << std::endl;

            // Reset file pointer to beginning
            file.clear();
            file.seekg(0, std::ios::beg);

            // Process the data
            auto [final_data, doc_id_mapping, token_to_id_mapping] =
                process_data(file, row_count, input_token_to_id_map);

            std::cout << "Elapsed time to read the collection "
                      << std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::high_resolution_clock::now() - start)
                             .count()
                      << " s" << std::endl;

            return SeismicIndex<T>::new_from_dataset(
                final_data,
                config,
                doc_id_mapping,
                token_to_id_mapping);
        }

        static SeismicIndex<T> from_tar(
            const std::string &tar_path,
            const Configuration &config
            // std::optional<std::unordered_map<std::string, size_t>> input_token_to_id_map = std::nullopt
        )
        {
            // TODO: return to this, use `input_token_to_id_map` in process_data
            // Simplified implementation that doesn't rely on libarchive
            // This is a temporary solution until the proper dependencies are available
            std::cout << "Reading the collection from tar.gz file: " << tar_path << std::endl;
            std::cout << "WARNING: tar.gz support is temporarily disabled." << std::endl;
            std::cout << "Please extract the file manually and use the .jsonl file directly." << std::endl;

            // Create a minimal empty dataset to return
            SparseDataset<T> empty_dataset(0);
            std::optional<std::vector<std::string>> empty_doc_mapping;
            std::unordered_map<std::string, size_t> empty_token_map;

            // Return a minimal index
            return SeismicIndex<T>::new_from_dataset(
                empty_dataset,
                config,
                empty_doc_mapping,
                empty_token_map);
        }

        void print_space_usage_byte()
        {
            inverted_index_.print_space_usage_byte();
        }

        // Implement the pure virtual method from SpaceUsage
        std::size_t space_usage_byte() const override
        {
            size_t total = 0;

            // Add size of inverted index
            total += inverted_index_.space_usage_byte();

            // Add size of document mapping if present
            if (document_mapping_)
            {
                for (const auto &doc : *document_mapping_)
                {
                    total += doc.size() * sizeof(char);
                    total += sizeof(std::string); // overhead for each string
                }
                total += sizeof(std::vector<std::string>); // vector overhead
            }

            // Add size of token to id map
            for (const auto &[token, id] : token_to_id_map_)
            {
                total += token.size() * sizeof(char);
                total += sizeof(std::string);                    // overhead for each string
                total += sizeof(size_t);                         // size of id
                total += sizeof(std::pair<std::string, size_t>); // pair overhead
            }
            total += sizeof(std::unordered_map<std::string, size_t>); // map overhead

            return total;
        }

        size_t dim()
        {
            return inverted_index_.dim();
        }

        size_t nnz()
        {
            return inverted_index_.nnz();
        }

        size_t len()
        {
            return inverted_index_.len();
        }

        InvertedIndex<T> &inverted_index()
        {
            return inverted_index_;
        }

        SparseDataset<T> &dataset()
        {
            return inverted_index_.dataset();
        }

        void add_knn(const Knn &knn)
        {
            inverted_index_.add_knn(knn);
        }

        size_t knn_len()
        {
            return inverted_index_.knn_len();
        }
    };

} // namespace seismic

#endif // SEISMIC_INDEX_H
