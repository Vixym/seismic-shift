#ifndef CONFIGURATION_STRATEGIES_H
#define CONFIGURATION_STRATEGIES_H

namespace seismic {

const size_t THRESHOLD_BINARY_SEARCH = 10;

/**
 * Clustering algorithm types for the inverted index.
 */
 class ClusteringAlgorithm {
    public:
        enum class Type {
            RandomKmeans,
            RandomKmeansInvertedIndex,
            RandomKmeansInvertedIndexApprox
        };
        
        // Default constructor - RandomKmeansInvertedIndexApprox with doc_cut = 15
        ClusteringAlgorithm() 
            : type(Type::RandomKmeansInvertedIndexApprox), 
              pruning_factor(0.0f), 
              doc_cut(15) {}
        
        // RandomKmeans constructor
        static ClusteringAlgorithm random_kmeans() {
            ClusteringAlgorithm algorithm;
            algorithm.type = Type::RandomKmeans;
            return algorithm;
        }
        
        // RandomKmeansInvertedIndex constructor
        static ClusteringAlgorithm random_kmeans_inverted_index(float pruning_factor, size_t doc_cut) {
            ClusteringAlgorithm algorithm;
            algorithm.type = Type::RandomKmeansInvertedIndex;
            algorithm.pruning_factor = pruning_factor;
            algorithm.doc_cut = doc_cut;
            return algorithm;
        }
        
        // RandomKmeansInvertedIndexApprox constructor
        static ClusteringAlgorithm random_kmeans_inverted_index_approx(size_t doc_cut) {
            ClusteringAlgorithm algorithm;
            algorithm.type = Type::RandomKmeansInvertedIndexApprox;
            algorithm.doc_cut = doc_cut;
            return algorithm;
        }
        
        // Getters
        Type get_type() const { return type; }
        float get_pruning_factor() const { return pruning_factor; }
        size_t get_doc_cut() const { return doc_cut; }

        // Serialization
        template <class Archive>
        void serialize(Archive& archive) {
            archive(type, pruning_factor, doc_cut);
        }
        
    private:
        Type type;
        float pruning_factor;
        size_t doc_cut;
    };

/**
 * Represents the possible choices for the strategy used to prune the posting
 * lists at building time.
 * There are the following possible strategies:
 * - `Fixed  { n_postings: usize }`: Every posting list is pruned by taking its top-`n_postings`
 * - `GlobalThreshold { n_postings: usize, max_fraction: f32 }`: We globally select a threshold and 
* we prune all the postings with smaller score. The threshold is chosen so that every posting list has 
* `n_postings` on average. We limit the number of postings per list to `max_fraction*n_postings`.
 */
 class PruningStrategy {
    public:
        enum class Type {
            FixedSize,
            GlobalThreshold
        };
    
        // Default constructor - FixedSize with n_postings = 3500
        PruningStrategy() : type(Type::FixedSize), n_postings(3500), max_fraction(0.0f) {}
    
        // FixedSize constructor
        static PruningStrategy fixed_size(size_t n_postings) {
            PruningStrategy strategy;
            strategy.type = Type::FixedSize;
            strategy.n_postings = n_postings;
            return strategy;
        }
    
        // GlobalThreshold constructor
        static PruningStrategy global_threshold(size_t n_postings, float max_fraction) {
            PruningStrategy strategy;
            strategy.type = Type::GlobalThreshold;
            strategy.n_postings = n_postings;
            strategy.max_fraction = max_fraction;
            return strategy;
        }

        // Getters
        Type get_type() const { return type; }
        size_t get_n_postings() const { return n_postings; }
        float get_max_fraction() const { return max_fraction; }

        // Serialization
        template <class Archive>
        void serialize(Archive& archive) {
            archive(type, n_postings, max_fraction);
        }
    
    private:
        Type type;
        size_t n_postings;
        float max_fraction;
    };

    /**
 * Represents the possible choices for the strategy used to block the posting lists.
 */
class BlockingStrategy {
    public:
        enum class Type {
            FixedSize,
            RandomKmeans
        };
    
        // Default constructor - RandomKmeans with default parameters
        BlockingStrategy() 
            : type(Type::RandomKmeans), 
              block_size(0), 
              centroid_fraction(0.1f), 
              min_cluster_size(2), 
              clustering_algorithm(ClusteringAlgorithm()) {}
    
        // FixedSize constructor
        static BlockingStrategy fixed_size(size_t block_size) {
            BlockingStrategy strategy;
            strategy.type = Type::FixedSize;
            strategy.block_size = block_size;
            return strategy;
        }
    
        // RandomKmeans constructor
        static BlockingStrategy random_kmeans(float centroid_fraction, size_t min_cluster_size, 
                                             const ClusteringAlgorithm& algorithm) {
            BlockingStrategy strategy;
            strategy.type = Type::RandomKmeans;
            strategy.centroid_fraction = centroid_fraction;
            strategy.min_cluster_size = min_cluster_size;
            strategy.clustering_algorithm = algorithm;
            return strategy;
        }
    
        // Getters
        Type get_type() const { return type; }
        size_t get_block_size() const { return block_size; }
        float get_centroid_fraction() const { return centroid_fraction; }
        size_t get_min_cluster_size() const { return min_cluster_size; }
        const ClusteringAlgorithm& get_clustering_algorithm() const { return clustering_algorithm; }

        // Serialization
        template <class Archive>
        void serialize(Archive& archive) {
            archive(type, block_size, centroid_fraction, min_cluster_size, clustering_algorithm);
        }
    
    private:
        Type type;
        size_t block_size;
        float centroid_fraction;
        size_t min_cluster_size;
        ClusteringAlgorithm clustering_algorithm;
    };

/**
 * Represents the possible choices for the strategy used to summarize blocks.
 */
 class SummarizationStrategy {
    public:
        enum class Type {
            FixedSize,
            EnergyPreserving
        };
    
        // Default constructor - EnergyPreserving with summary_energy = 0.4
        SummarizationStrategy() : type(Type::EnergyPreserving), n_components(0), summary_energy(0.4f) {}
    
        // FixedSize constructor
        static SummarizationStrategy fixed_size(size_t n_components) {
            SummarizationStrategy strategy;
            strategy.type = Type::FixedSize;
            strategy.n_components = n_components;
            return strategy;
        }
    
        // EnergyPreserving constructor
        static SummarizationStrategy energy_preserving(float summary_energy) {
            SummarizationStrategy strategy;
            strategy.type = Type::EnergyPreserving;
            strategy.summary_energy = summary_energy;
            return strategy;
        }
    
        // Getters
        Type get_type() const { return type; }
        size_t get_n_components() const { return n_components; }
        float get_summary_energy() const { return summary_energy; }

        // Serialization
        template <class Archive>
        void serialize(Archive& archive) {
            archive(type, n_components, summary_energy);
        }
    
    private:
        Type type;
        size_t n_components;
        float summary_energy;
};    

/**
 * KNN configuration class for the inverted index.
 */
 class KnnConfiguration {
    private:
        size_t nknn;
        std::optional<std::string> knn_path;
    public:
        // Default constructor
        KnnConfiguration() : nknn(0), knn_path(std::nullopt) {}
        
        // Constructor with parameters
        KnnConfiguration(size_t nknn, std::optional<std::string> knn_path = std::nullopt)
            : nknn(nknn), knn_path(std::move(knn_path)) {}
        
        // Getters
        size_t get_nknn() const { return nknn; }
        const std::optional<std::string>& get_knn_path() const { return knn_path; }

        // Serialization
        template <class Archive>
        void serialize(Archive& archive) {
            archive(nknn, knn_path);
        }
    };

/**
 * Configuration for the inverted index.
 * Contains parameters for pruning, blocking, summarization, and kNN.
 */
 class Configuration {
    public:
        // Default constructor
        Configuration() : pruning(PruningStrategy()), blocking(BlockingStrategy()),
                         summarization(SummarizationStrategy()), knn_config(KnnConfiguration()) {}
    
        // Builder pattern methods
        Configuration& pruning_strategy(const PruningStrategy& pruning);
        Configuration& blocking_strategy(const BlockingStrategy& blocking);
        Configuration& summarization_strategy(const SummarizationStrategy& summarization);
        Configuration& knn(const KnnConfiguration& knn_config);
        Configuration& batched_indexing(const std::optional<size_t>& batched_indexing);
    
        // Getters
        const PruningStrategy& get_pruning() const { return pruning; }
        const BlockingStrategy& get_blocking() const { return blocking; }
        const SummarizationStrategy& get_summarization() const { return summarization; }
        const KnnConfiguration& get_knn_config() const { return knn_config; }
        const std::optional<size_t>& get_batched_indexing() const { return batched_indexing_size; }

        // Serialization
        template <class Archive>
        void serialize(Archive& archive) {
            archive(pruning, blocking, summarization, knn_config, batched_indexing_size);
        }
    
    private:
        PruningStrategy pruning;
        BlockingStrategy blocking;
        SummarizationStrategy summarization;
        KnnConfiguration knn_config;
        std::optional<size_t> batched_indexing_size;
    };

    /**
 * KNN graph implementation for the inverted index.
 */
 class Knn {
    // TODO implement space usage
    private:
        size_t n_vecs;
        size_t d;
        BitVector neighbours;
        size_t nbits;

    public:
        static constexpr size_t KNN_QUERY_CUT = 10;
        static constexpr float KNN_HEAP_FACTOR = 0.7f;

        // Default constructor
        Knn() : n_vecs(0), d(0), nbits(0) {}
    
        // Constructor with parameters
        Knn(size_t n_vecs, size_t d, BitVector neighbours, size_t nbits)
            : n_vecs(n_vecs), d(d), neighbours(std::move(neighbours)), nbits(nbits) {}
    
        // Build a KNN graph from an inverted index
        template <class Index>
        static Knn new_from_index(const Index& index, size_t d) {
            const size_t n_vecs = index.len();
            std::cout << "Computing kNN: ";
            
            // Store search results for each document
            std::vector<std::vector<std::pair<float, size_t>>> docs_search_results(n_vecs);
            
            // Parallel processing of each document
            #pragma omp parallel for
            for (size_t my_doc_id = 0; my_doc_id < index.dataset().len(); ++my_doc_id) {
                auto [components, values] = index.dataset().get(my_doc_id);
                
                // Convert values to float
                std::vector<float> f32_values(values.size());
                for (size_t i = 0; i < values.size(); ++i) {
                    f32_values[i] = to_f32(values[i]);
                }
                
                // Search for similar documents
                auto results = index.search(
                    components,
                    f32_values,
                    d + 1, // +1 to filter the document itself if present
                    KNN_QUERY_CUT,
                    KNN_HEAP_FACTOR,
                    0,
                    false
                );
                
                // Filter out the document itself and take only the top d
                std::vector<std::pair<float, size_t>> filtered_results;
                filtered_results.reserve(d);
                
                for (const auto& [distance, doc_id] : results) {
                    if (doc_id != my_doc_id) {
                        filtered_results.emplace_back(distance, doc_id);
                        if (filtered_results.size() >= d) {
                            break;
                        }
                    }
                }
                
                docs_search_results[my_doc_id] = std::move(filtered_results);
            }
            
            // Flatten the results into a vector of document IDs
            std::vector<uint64_t> doc_ids;
            doc_ids.reserve(n_vecs * d);
            
            for (const auto& results : docs_search_results) {
                for (const auto& [_, doc_id] : results) {
                    doc_ids.push_back(static_cast<uint64_t>(doc_id));
                }
            }
            
            // Compress into a bit vector
            auto [bv, nbits] = compress_into_bitvector(doc_ids, n_vecs, d);
            
            return Knn(n_vecs, d, std::move(bv), nbits);
        }
    
        // Load a KNN graph from a serialized file
        static Knn new_from_serialized(const std::string& path, std::optional<size_t> limit = std::nullopt) {
            std::cout << "Reading KNN from file: " << path << std::endl;
    
            // Read the serialized data
            std::ifstream file(path, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Failed to open KNN file: " + path);
            }
    
            // Deserialize the Knn object
            size_t n_vecs, d, nbits;
            file.read(reinterpret_cast<char*>(&n_vecs), sizeof(n_vecs));
            file.read(reinterpret_cast<char*>(&d), sizeof(d));
            file.read(reinterpret_cast<char*>(&nbits), sizeof(nbits));
    
            // Read the bit vector
            BitVectorMut neighbours_mut;
            size_t bit_vector_size;
            file.read(reinterpret_cast<char*>(&bit_vector_size), sizeof(bit_vector_size));
            
            std::vector<uint64_t> data(bit_vector_size);
            file.read(reinterpret_cast<char*>(data.data()), bit_vector_size * sizeof(uint64_t));
            
            for (size_t i = 0; i < bit_vector_size; ++i) {
                neighbours_mut.append_bits(data[i], 64);
            }
            
            BitVector neighbours = neighbours_mut.into();
    
            std::cout << "Number of vectors: " << n_vecs << std::endl;
            std::cout << "Number of neighbors: " << d << std::endl;
            std::cout << "Number of bits: " << nbits << std::endl;
            
            if (limit.has_value()) {
                size_t nknn = limit.value();
                if (nknn < d) {
                    std::cout << "We only take " << nknn << " neighbors per element!" << std::endl;
                    
                    // Create a new bit vector with fewer neighbors per vector
                    BitVectorMut new_neighbours(n_vecs * nbits * nknn);
                    
                    for (size_t id = 0; id < n_vecs; ++id) {
                        for (size_t i = 0; i < nknn; ++i) {
                            size_t bit_offset = id * d * nbits + i * nbits;
                            uint64_t neighbor = neighbours.get_bits_unchecked(bit_offset, nbits);
                            new_neighbours.append_bits(neighbor, nbits);
                        }
                    }
                    
                    return Knn(n_vecs, nknn, new_neighbours.into(), nbits);
                } else {
                    std::cout << "We only take " << d << " neighbors per element!" << std::endl;
                    
                    // Create a new bit vector with fewer neighbors per vector
                    BitVectorMut new_neighbours(n_vecs * nbits * d);
                    
                    for (size_t id = 0; id < n_vecs; ++id) {
                        for (size_t i = 0; i < d; ++i) {
                            size_t bit_offset = id * d * nbits + i * nbits;
                            uint64_t neighbor = neighbours.get_bits_unchecked(bit_offset, nbits);
                            new_neighbours.append_bits(neighbor, nbits);
                        }
                    }
                    
                    return Knn(n_vecs, d, new_neighbours.into(), nbits);
                }
            } else {
                std::cout << "We only take " << d << " neighbors per element!" << std::endl;
                
                // Create a new bit vector with fewer neighbors per vector
                BitVectorMut new_neighbours(n_vecs * nbits * d);
                
                for (size_t id = 0; id < n_vecs; ++id) {
                    for (size_t i = 0; i < d; ++i) {
                        size_t bit_offset = id * d * nbits + i * nbits;
                        uint64_t neighbor = neighbours.get_bits_unchecked(bit_offset, nbits);
                        new_neighbours.append_bits(neighbor, nbits);
                    }
                }
                
                return Knn(n_vecs, d, new_neighbours.into(), nbits);
            }
        }
    
        // Serialize the KNN graph to a file
        void serialize(const std::string& output_file) const {
            std::ofstream file(output_file, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Failed to open KNN file: " + output_file);
            }
            
            // Write the Knn object
            file.write(reinterpret_cast<const char*>(&n_vecs), sizeof(n_vecs));
            file.write(reinterpret_cast<const char*>(&d), sizeof(d));
            file.write(reinterpret_cast<const char*>(&nbits), sizeof(nbits));
            
            // Write the bit vector
            const auto& neighbours = get_neighbours();
            size_t bit_vector_size = neighbours.size();
            file.write(reinterpret_cast<const char*>(&bit_vector_size), sizeof(bit_vector_size));
            
            // Since we can't directly access the data, we'll write the bits in chunks
            for (size_t i = 0; i < bit_vector_size; i += 64) {
                size_t bits_to_read = std::min(size_t(64), bit_vector_size - i);
                uint64_t bits = neighbours.get_bits_unchecked(i, bits_to_read);
                file.write(reinterpret_cast<const char*>(&bits), sizeof(bits));
            }
        }

        // Compress a sequence of document IDs into a bit vector
        static std::pair<BitVector, size_t> compress_into_bitvector(
            const std::vector<uint64_t>& data, size_t n_vecs, [[maybe_unused]] size_t d) {

            // Calculate number of bits needed to represent n_vecs
            size_t nbits = static_cast<size_t>(std::ceil(std::log2(static_cast<double>(n_vecs))));
            
            // Create a bit vector to store the compressed data
            BitVectorMut neighbours(data.size() * nbits);
                    
            // Append each document ID to the bit vector
            for (uint64_t x : data) {
                neighbours.append_bits(x, nbits);
            }
                    
            return {neighbours.into(), nbits};
        }
    
        // Refine the search results using the KNN graph
        template <typename T>
        void refine(const std::vector<float>& query,
                   utils::HeapFaiss& heap,
                   std::unordered_set<size_t>& visited,
                   const SparseDataset<T>& forward_index,
                   size_t in_n_knn) const {
            // Use the minimum of the requested number of neighbors and the available number
            size_t n_knn = std::min(d, in_n_knn);
    
            // Get the current top-k results
            auto neighbours_results = heap.topk();
    
            // For each result, explore its neighbors
            for (const auto& [_distance, offset] : neighbours_results) {
                size_t id = forward_index.offset_to_id(offset);
                
                // For each neighbor of the current result
                for (size_t i = 0; i < n_knn; ++i) {
                    size_t bit_offset = id * d * nbits + i * nbits;
                    uint64_t neighbour = neighbours.get_bits_unchecked(bit_offset, nbits);
                    
                    // Get the offset and length of the neighbor's vector
                    size_t offset = forward_index.vector_offset(static_cast<size_t>(neighbour));
                    size_t len = forward_index.vector_len(static_cast<size_t>(neighbour));
                    
                    // If not already visited, compute the distance and add to the heap
                    if (visited.find(offset) == visited.end()) {
                        auto [v_components, v_values] = forward_index.get_with_offset(offset, len);
                        
                        float distance = distances::dot_product_dense_sparse(query, v_components, v_values);
                        
                        visited.insert(offset);
                        heap.push_with_id(-1.0f * distance, offset);
                    }
                }
            }
        }

        // Refine the search results using the KNN graph
        template <typename T>
        void refine(const std::vector<float>& query,
                   utils::HeapFaiss& heap,
                   std::unordered_set<size_t>& visited,
                   const SparseDatasetMut<T>& forward_index,
                   size_t in_n_knn) const {
            // Use the minimum of the requested number of neighbors and the available number
            size_t n_knn = std::min(d, in_n_knn);
    
            // Get the current top-k results
            auto neighbours_results = heap.topk();
    
            // For each result, explore its neighbors
            for (const auto& [_distance, offset] : neighbours_results) {
                size_t id = forward_index.offset_to_id(offset);
                
                // For each neighbor of the current result
                for (size_t i = 0; i < n_knn; ++i) {
                    size_t bit_offset = id * d * nbits + i * nbits;
                    uint64_t neighbour = neighbours.get_bits_unchecked(bit_offset, nbits);
                    
                    // Get the offset and length of the neighbor's vector
                    size_t offset = forward_index.vector_offset(static_cast<size_t>(neighbour));
                    size_t len = forward_index.vector_len(static_cast<size_t>(neighbour));
                    
                    // If not already visited, compute the distance and add to the heap
                    if (visited.find(offset) == visited.end()) {
                        auto [v_components, v_values] = forward_index.get_with_offset(offset, len);
                        
                        float distance = distances::dot_product_dense_sparse(query, v_components, v_values);
                        
                        visited.insert(offset);
                        heap.push_with_id(-1.0f * distance, offset);
                    }
                }
            }
        }
    
        // Getters
        size_t get_n_vecs() const { return n_vecs; }
        size_t get_d() const { return d; }
        const BitVector& get_neighbours() const { return neighbours; }
        size_t get_nbits() const { return nbits; }
        
        // Space usage calculation
        size_t space_usage_byte() const;

        template <class Archive>
        void serialize(Archive& archive) {
            archive(n_vecs, d, neighbours, nbits); \
        }
    };

} // namespace seismic

#endif // INVERTED_INDEX_H