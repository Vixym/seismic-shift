#ifndef SEISMIC_INDEX_H
#define SEISMIC_INDEX_H

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <optional>
#include <fstream>
#include <iostream>

#include "inverted_index.h"
#include "sparse_dataset.h"
#include "space_usage.h"
#include "data_type.h"

namespace seismic {
    // TODO: return to space usage calculation here

    /**
    * A wrapper class for InvertedIndex that adds document mapping functionality
    * and token-to-id mapping.
    */
    template <typename T>
    class SeismicIndex : public SpaceUsage {
    
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
        std::unordered_map<std::string, size_t> token_to_id_map
    ) : inverted_index_(std::move(inverted_index)),
        document_mapping_(std::move(document_mapping)),
        token_to_id_map_(std::move(token_to_id_map)) {}

    // Create a new index from a dataset and configuration
    static SeismicIndex<T> new_from_dataset(
        const SparseDataset<T>& dataset,
        const Configuration& config,
        std::optional<std::vector<std::string>> document_mapping,
        std::unordered_map<std::string, size_t> token_to_id_map
    ) {
        auto inverted_index = InvertedIndex<T>::build(dataset, config);
        return SeismicIndex<T>(
            std::move(inverted_index),
            std::move(document_mapping),
            std::move(token_to_id_map)
        );
    }

    // Helper function to remap document IDs to their string representations
    std::vector<std::tuple<std::string, float, std::string>> remap_doc_ids(
        const std::vector<std::pair<float, size_t>>& plain_results,
        const std::string& query_id
    ) const {
        std::vector<std::tuple<std::string, float, std::string>> remapped_results;
        remapped_results.reserve(plain_results.size());

        for (const auto& [distance, doc_id] : plain_results) {
            std::string doc_name;
            if (document_mapping_) {
                doc_name = (*document_mapping_)[doc_id];
            } else {
                doc_name = std::to_string(doc_id);
            }
            remapped_results.emplace_back(query_id, distance, std::move(doc_name));
        }

        return remapped_results;
    }

    // Search functionality that returns raw results (document IDs and scores)
    std::vector<std::pair<float, size_t>> search_raw(
        const std::vector<std::string>& query_components_original,
        const std::vector<float>& query_values,
        size_t k,
        size_t query_cut,
        float heap_factor,
        size_t n_knn,
        bool first_sorted
    ) const {
        // Convert string components to IDs
        std::vector<uint16_t> filtered_components;
        std::vector<float> filtered_values;
        filtered_components.reserve(query_components_original.size());
        filtered_values.reserve(query_components_original.size());

        for (size_t i = 0; i < query_components_original.size(); ++i) {
            auto it = token_to_id_map_.find(query_components_original[i]);
            if (it != token_to_id_map_.end()) {
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
            first_sorted
        );
    }

    // Search functionality that returns mapped results (with document names)
    std::vector<std::tuple<std::string, float, std::string>> search(
        const std::string& query_id,
        const std::vector<std::string>& query_components_original,
        const std::vector<float>& query_values,
        size_t k,
        size_t query_cut,
        float heap_factor,
        size_t n_knn,
        bool first_sorted
    ) const {
        auto results = search_raw(
            query_components_original,
            query_values,
            k,
            query_cut,
            heap_factor,
            n_knn,
            first_sorted
        );

        return remap_doc_ids(results, query_id); // return the documents remapped
    }

    // Process data from a stream and create necessary mappings
    static std::tuple<SparseDataset<T>, std::vector<std::string>, std::unordered_map<std::string, size_t>>
    process_data(
        std::istream& reader,
        size_t row_count,
        const std::optional<std::unordered_map<std::string, size_t>>& input_token_to_id_map = std::nullopt
    );

    static SeismicIndex<T> from_file(
        const std::string& file_path,
        const Configuration& config,
        const std::optional<std::unordered_map<std::string, size_t>& input_token_to_id_map = std::nullopt
    ) {
        if (file_path.ends_with(".jsonl")) {
            return SeismicIndex::from_json(file_path, config, input_token_to_id_map);
        } else if (file_path.ends_with(".tar.gz")) {
            return SeismicIndex::from_tar(file_path, config, input_token_to_id_map);
        } else {
            throw std::runtime_error("Unsupported file type. Supported files: .jsonl, .tar.gz");
        }
    }

    static SeismicIndex<T> from_json(
        const std::string& json_path,
        const Configuration& config,
        const std::optional<std::unordered_map<std::string, size_t>>& input_token_to_id_map = std::nullopt
    ) {
        std::cout << "Reading the collection.." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        // read the file and count rows
        std::ifstream f(json_path);
        std::istream reader = std::istream(f);
        std::size_t row_count = std::distance(std::istream_iterator<std::string>(reader), std::istream_iterator<std::string>());

        std::cout << "Number of rows: " << row_count << std::endl;
        std::cout << "Elapsed time to read the number of rows " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() << " s" << std::endl;

        std::ifstream f(json_path);
        std::istream reader = std::istream(f);

        // deserialize json
        auto [final_data, doc_id_mapping, token_to_id_mapping] = process_data(reader, row_count);
        std::cout << "Elapsed time to read the collection " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() << " s" << std::endl;

        return SeismicIndex<T>::new_from_dataset(
            final_data,
            config,
            doc_id_mapping,
            token_to_id_mapping
        );
    }

    static SeismicIndex<T> from_tar(
        const std::string& tar_path,
        const Configuration& config,
        std::optional<std::unordered_map<std::string, size_t>> input_token_to_id_map = std::nullopt
    ) {
        std::cout << "Reading the collection.." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        // Decompress gz, extract first file (json) of the archive
        // Read the file and count rows
        std::cout << "Opening tar archive: " << tar_path << std::endl;
        
        // Use libarchive to open and read the tar.gz file
        struct archive* a = archive_read_new();
        archive_read_support_filter_gzip(a);
        archive_read_support_format_tar(a);
        
        if (archive_read_open_filename(a, tar_path.c_str(), 10240) != ARCHIVE_OK) {
            throw std::runtime_error("Failed to open archive: " + tar_path);
        }
        
        struct archive_entry* entry;
        if (archive_read_next_header(a, &entry) != ARCHIVE_OK) {
            throw std::runtime_error("Failed to read first entry in archive");
        }
        
        // Count rows
        size_t row_count = 0;
        std::stringstream buffer;
        
        const void* buff;
        size_t size;
        la_int64_t offset;
        
        while (archive_read_data_block(a, &buff, &size, &offset) == ARCHIVE_OK) {
            buffer.write(static_cast<const char*>(buff), size);
        }
        
        std::string line;
        std::istringstream content(buffer.str());
        while (std::getline(content, line)) {
            row_count++;
        }
        
        std::cout << "Number of rows: " << row_count << std::endl;
        std::cout << "Elapsed time to read the number of rows " 
                  << std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::high_resolution_clock::now() - start).count() 
                  << " s" << std::endl;
        
        // Reopen the archive for processing
        archive_read_free(a);
        a = archive_read_new();
        archive_read_support_filter_gzip(a);
        archive_read_support_format_tar(a);
        
        if (archive_read_open_filename(a, tar_path.c_str(), 10240) != ARCHIVE_OK) {
            throw std::runtime_error("Failed to reopen archive: " + tar_path);
        }
        
        if (archive_read_next_header(a, &entry) != ARCHIVE_OK) {
            throw std::runtime_error("Failed to read first entry in archive");
        }
        
        // Read content into a stream for processing
        buffer.str("");
        buffer.clear();
        
        while (archive_read_data_block(a, &buff, &size, &offset) == ARCHIVE_OK) {
            buffer.write(static_cast<const char*>(buff), size);
        }
        
        std::istringstream reader(buffer.str());
        
        // Deserialize json
        auto [final_data, doc_id_mapping, token_to_id_mapping] = 
            process_data(reader, row_count, input_token_to_id_map);
            
        std::cout << "Elapsed time to read the collection " 
                  << std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::high_resolution_clock::now() - start).count() 
                  << " s" << std::endl;
        
        archive_read_free(a);
        
        return SeismicIndex<T>::new_from_dataset(
            final_data,
            config,
            doc_id_mapping,
            token_to_id_mapping
        );
    }

    static void print_space_usage_byte() {
        this.inverted_index_.print_space_usage_byte();
    }

    static size_t dim() {
        this.inverted_index_.dim();
    }

    static size_t nnz() {
        this.inverted_index_.nnz();
    }

    static size_t len() {
        this.inverted_index_.len();
    }

    static InvertedIndex<T>& inverted_index() {
        return this.inverted_index_;
    }

    static SparseDataset<T>& dataset() {
        return this.inverted_index_.dataset();
    }

    static void add_knn(const Knn& knn) {
        this.inverted_index_.add_knn(knn);
    }

    static size_t knn_len() {
        return this.inverted_index_.knn_len();
    }
};

} // namespace seismic

// Include the implementation
#include "seismic_index_impl.tpp"

#endif // SEISMIC_INDEX_H
