#ifndef SPARSE_DATASET_H
#define SPARSE_DATASET_H

#include <vector>
#include <memory>
#include <optional>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <string>
#include <utility>
#include <iterator>

#include "data_type.h"
#include "space_usage.h"

namespace seismic {

// Forward declaration
template <typename T>
class SparseDatasetMut;

/**
 * Implementation of an immutable sparse dataset.
 * 
 * A sparse vector in a `dim`-dimensional space consists of two sequences: components,
 * which are distinct values in the range [0, `dim`), and their corresponding values of type `T`.
 * The type `T` is typically expected to be a float type such as `Float16`, `float`, or `double`.
 * The components are of type `uint16_t`.
 */
template <typename T>
class SparseDataset {
private:
    size_t n_vecs;
    size_t d;
    std::vector<size_t> offsets;
    std::vector<uint16_t> components;
    std::vector<T> values;

    // Make SparseDatasetMut a friend class to access private members
    friend class SparseDatasetMut<T>;

    // Returns the range of positions of the slice with the given `id`.
    static std::pair<size_t, size_t> vector_range(const std::vector<size_t>& offsets, size_t id) {
        assert(id < offsets.size() - 1 && "vector_range::The id is out of range");
        return {offsets[id], offsets[id + 1]};
    }

public:
    // Default constructor
    SparseDataset() : n_vecs(0), d(0) {
        offsets.push_back(0);
    }

    // Constructor with parameters
    SparseDataset(size_t n_vecs, size_t d, 
                 std::vector<size_t> offsets,
                 std::vector<uint16_t> components,
                 std::vector<T> values)
        : n_vecs(n_vecs), d(d), offsets(std::move(offsets)), 
          components(std::move(components)), values(std::move(values)) {}

    /**
     * Retrieves the components and values of the sparse vector at the specified index.
     * 
     * @param id The index of the sparse vector to retrieve.
     * @return A pair containing views of components and values of the sparse vector.
     * @throws std::out_of_range if the specified index is out of range.
     */
    std::pair<std::vector<uint16_t>, std::vector<T>> get(size_t id) const {
        if (id >= n_vecs) {
            throw std::out_of_range("get::The id is out of range");
        }

        auto [start, end] = vector_range(offsets, id);
        
        std::vector<uint16_t> v_components(components.begin() + start, components.begin() + end);
        std::vector<T> v_values(values.begin() + start, values.begin() + end);

        return {v_components, v_values};
    }

    /**
     * Returns a vector of the dataset at the specified `offset` and `len`.
     * 
     * @param offset The starting index of the vector to retrieve.
     * @param len The length of the vector to retrieve.
     * @return A pair containing views of components and values of the vector.
     * @throws std::out_of_range if the offset + len is out of range.
     */
    std::pair<std::vector<uint16_t>, std::vector<T>> get_with_offset(size_t offset, size_t len) const {
        if (offset + len > components.size()) {
            throw std::out_of_range("The offset + len is out of range");
        }

        std::vector<uint16_t> v_components(components.begin() + offset, components.begin() + offset + len);
        std::vector<T> v_values(values.begin() + offset, values.begin() + offset + len);

        return {v_components, v_values};
    }

    /**
     * Returns the offset of the vector with the specified index.
     * 
     * @param id The index of the vector.
     * @return The offset of the vector.
     * @throws std::out_of_range if the specified index is out of range.
     */
    size_t vector_offset(size_t id) const {
        if (id >= n_vecs) {
            throw std::out_of_range("vector_offset::The id is out of range");
        }

        return offsets[id];
    }

    /**
     * Returns the length of the vector with the specified index.
     * 
     * @param id The index of the vector.
     * @return The length of the vector.
     * @throws std::out_of_range if the specified index is out of range.
     */
    size_t vector_len(size_t id) const {
        if (id >= n_vecs) {
            throw std::out_of_range("vector_len::The id is out of range");
        }

        return offsets[id + 1] - offsets[id];
    }

    /**
     * Returns the number of vectors in the dataset.
     * 
     * @return The number of vectors in the dataset.
     */
    size_t len() const {
        return n_vecs;
    }

    /**
     * Checks if the dataset is empty.
     * 
     * @return True if the dataset is empty, false otherwise.
     */
    bool is_empty() const {
        return n_vecs == 0;
    }

    /**
     * Returns the number of components of the dataset.
     * 
     * @return The number of components of the dataset.
     */
    size_t dim() const {
        return d;
    }

    /**
     * Returns the number of non-zero components in the dataset.
     * 
     * @return The number of non-zero components in the dataset.
     */
    size_t nnz() const {
        return components.size();
    }

    /**
     * Converts the `offset` of a vector within the dataset to its id.
     * 
     * @param offset The offset of the vector.
     * @return The id of the vector.
     * @throws std::out_of_range if the offset is not the first position of a vector in the dataset.
     */
    size_t offset_to_id(size_t offset) const {
        
        auto it = std::lower_bound(offsets.begin(), offsets.end(), offset);
        if (it == offsets.end() || *it != offset) {
            throw std::out_of_range("The offset is not the first position of a vector in the dataset");
        }
        return std::distance(offsets.begin(), it);
    }

    /**
     * Converts the id of a vector to its offset.
     * 
     * @param id The id of the vector.
     * @return The offset of the vector.
     * @throws std::out_of_range if the specified index is out of range.
     */
    size_t id_to_offset(size_t id) const {
        if (id >= n_vecs) {
            throw std::out_of_range("id_to_offset::The id is out of range");
        }
        return offsets[id];
    }

    /**
     * Converts the id of a vector to its offset and length.
     * 
     * @param id The id of the vector.
     * @return A pair containing the offset and length of the vector.
     * @throws std::out_of_range if the specified index is out of range.
     */
    std::pair<size_t, size_t> id_to_offset_len(size_t id) const {
        if (id >= n_vecs) {
            throw std::out_of_range("id_to_offset_len::The id is out of range");
        }
        return {offsets[id], offsets[id + 1] - offsets[id]};
    }

    /**
     * Reads a sparse dataset from a binary file.
     * 
     * @param fname The name of the file to read from.
     * @return A SparseDataset object.
     * @throws std::runtime_error if the file cannot be opened or read.
     */
    static SparseDataset<T> read_bin_file(const std::string& fname) {
        return read_bin_file_limit(fname, std::nullopt);
    }

    /**
     * Reads a sparse dataset from a binary file with a limit on the number of vectors.
     * 
     * @param fname The name of the file to read from.
     * @param limit The maximum number of vectors to read.
     * @return A SparseDataset object.
     * @throws std::runtime_error if the file cannot be opened or read.
     */
    static SparseDataset<T> read_bin_file_limit(const std::string& fname, std::optional<size_t> limit) {
        std::ifstream file(fname, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + fname);
        }

        uint32_t n_vecs;
        file.read(reinterpret_cast<char*>(&n_vecs), sizeof(n_vecs));

        if (limit) {
            n_vecs = std::min(n_vecs, static_cast<uint32_t>(*limit));
        }

        SparseDatasetMut<T> dataset;
        
        for (uint32_t i = 0; i < n_vecs; ++i) {
            uint32_t n;
            file.read(reinterpret_cast<char*>(&n), sizeof(n));

            std::vector<uint16_t> components(n);
            std::vector<T> values(n);

            for (uint32_t j = 0; j < n; ++j) {
                uint32_t c;
                file.read(reinterpret_cast<char*>(&c), sizeof(c));
                components[j] = static_cast<uint16_t>(c);
            }

            for (uint32_t j = 0; j < n; ++j) {
                float v;
                file.read(reinterpret_cast<char*>(&v), sizeof(v));
                values[j] = static_cast<T>(v);
            }

            dataset.push(components, values);
        }

        return dataset.to_immutable();
    }

    // Iterator class for SparseDataset
    class Iterator {
    private:
        const SparseDataset<T>* dataset;
        size_t current_id;

    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = std::pair<std::vector<uint16_t>, std::vector<T>>;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type*;
        using reference = const value_type&;

        Iterator(const SparseDataset<T>* dataset, size_t id) : dataset(dataset), current_id(id) {}

        value_type operator*() const {
            return dataset->get(current_id);
        }

        Iterator& operator++() {
            ++current_id;
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const Iterator& other) const {
            return dataset == other.dataset && current_id == other.current_id;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }
    };

    /**
     * Returns an iterator to the beginning of the dataset.
     * 
     * @return An iterator to the beginning of the dataset.
     */
    Iterator begin() const {
        return Iterator(this, 0);
    }

    /**
     * Returns an iterator to the end of the dataset.
     * 
     * @return An iterator to the end of the dataset.
     */
    Iterator end() const {
        return Iterator(this, n_vecs);
    }

    /**
     * Returns the space usage of the dataset in bytes.
     * 
     * @return The space usage of the dataset in bytes.
     */
    size_t space_usage_byte() const {
        return sizeof(n_vecs) + sizeof(d) + 
               offsets.size() * sizeof(size_t) +
               components.size() * sizeof(uint16_t) +
               values.size() * sizeof(T);
    }

    template <class Archive> \
    void serialize(Archive& archive) { \
        archive(n_vecs, d, offsets, components, values); \
    }
};

/**
 * Implementation of a mutable sparse dataset.
 * 
 * A sparse vector in a `dim`-dimensional space consists of two sequences: components,
 * which are distinct values in the range [0, `dim`), and their corresponding values of type `T`.
 * The type `T` is typically expected to be a float type such as `Float16`, `float`, or `double`.
 * The components are of type `uint16_t`.
 */
template <typename T>
class SparseDatasetMut {
private:
    size_t d;
    size_t n_vecs;
    std::vector<size_t> offsets;
    std::vector<uint16_t> components;
    std::vector<T> values;
    std::vector<bool> alive;

public:
    // Default constructor
    SparseDatasetMut() : d(0), n_vecs(0) {
        offsets.push_back(0);
    }

    size_t add_document(const std::vector<uint16_t>& components, const std::vector<T>& values) {
        if (components.size() != values.size()) {
            throw std::invalid_argument("Vectors have different sizes");
        }
        if (components.empty()) {
            throw std::invalid_argument("Components cannot be empty");
        }
        if (!std::is_sorted(components.begin(), components.end())) {
            throw std::invalid_argument("Components must be given in sorted order");
        }

        this->n_vecs += 1;
        if (components.back() >= this->d) {
            this->d = components.back() + 1;
        }

        this->components.insert(this->components.end(), components.begin(), components.end());
        this->values.insert(this->values.end(), values.begin(), values.end());
        this->offsets.push_back(this->components.size());
        this->alive.push_back(true);

        return this->offsets.size()-2;
    }

    std::pair<size_t, size_t> id_to_offset_len(size_t id) const {
        if (id >= n_vecs) {
            throw std::out_of_range("id_to_offset_len::The id is out of range");
        }
        return {offsets[id], offsets[id + 1] - offsets[id]};
    }

    static SparseDatasetMut<T> read_bin_file_limit(const std::string& fname, std::optional<size_t> limit) {
        std::ifstream file(fname, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + fname);
        }

        uint32_t n_vecs;
        file.read(reinterpret_cast<char*>(&n_vecs), sizeof(n_vecs));

        if (limit) {
            n_vecs = std::min(n_vecs, static_cast<uint32_t>(*limit));
        }

        SparseDatasetMut<T> dataset;
        
        for (uint32_t i = 0; i < n_vecs; ++i) {
            uint32_t n;
            file.read(reinterpret_cast<char*>(&n), sizeof(n));

            std::vector<uint16_t> components(n);
            std::vector<T> values(n);

            for (uint32_t j = 0; j < n; ++j) {
                uint32_t c;
                file.read(reinterpret_cast<char*>(&c), sizeof(c));
                components[j] = static_cast<uint16_t>(c);
            }

            for (uint32_t j = 0; j < n; ++j) {
                float v;
                file.read(reinterpret_cast<char*>(&v), sizeof(v));
                values[j] = static_cast<T>(v);
            }

            dataset.push(components, values);
        }

        return dataset;
    }

    size_t offset_to_id(size_t offset) const {
        auto it = std::lower_bound(offsets.begin(), offsets.end(), offset);
        if (it == offsets.end() || *it != offset) {
            throw std::out_of_range("The offset is not the first position of a vector in the dataset");
        }
        return std::distance(offsets.begin(), it);
    }

    static SparseDatasetMut<T> read_bin_file(const std::string& fname) {
        return read_bin_file_limit(fname, std::nullopt);
    }

    void push(const std::vector<std::pair<uint16_t, T>> pairs) {
        if (pairs.empty()) {
            //throw std::invalid_argument("Pairs cannot be empty");
            std::cout << "Pairs are empty" << std::endl;
            return;
        }

        for (size_t i = 1; i < pairs.size(); ++i) {
            if (pairs[i-1].first >= pairs[i].first) {
                throw std::invalid_argument("Components must be given in sorted order");
            }
        }

        this->n_vecs += 1;
        if (pairs.back().first >= this->d) {
            this->d = pairs.back().first + 1;
        }

        for (const auto& [c, v] : pairs) {
            this->components.push_back(c);
            this->values.push_back(v);
        }
        this->offsets.push_back(this->components.size());
        this->alive.push_back(true);
    }

    void push(const std::vector<std::pair<size_t, T>> pairs) {
        if (pairs.empty()) {
            //throw std::invalid_argument("Pairs cannot be empty");
            std::cout << "Pairs are empty" << std::endl;
            return;
        }

        for (size_t i = 1; i < pairs.size(); ++i) {
            if (pairs[i-1].first >= pairs[i].first) {
                throw std::invalid_argument("Components must be given in sorted order");
            }
        }

        this->n_vecs += 1;
        if (pairs.back().first >= this->d) {
            this->d = pairs.back().first + 1;
        }

        for (const auto& [c, v] : pairs) {
            this->components.push_back(c);
            this->values.push_back(v);
        }
        this->offsets.push_back(this->components.size());
        this->alive.push_back(true);
    }

    /**
     * Adds a new sparse vector to the dataset.
     * 
     * @param components The components of the sparse vector.
     * @param values The values of the sparse vector.
     * @throws std::invalid_argument if the sizes of components and values are different,
     *         or if components is empty, or if components is not sorted.
     */
    void push(const std::vector<uint16_t>& components, const std::vector<T>& values, bool debug=false) {
        if (components.size() != values.size()) {
            throw std::invalid_argument("Vectors have different sizes");
        }
        if (components.empty()) {
            throw std::invalid_argument("Components cannot be empty");
        }
        if (!std::is_sorted(components.begin(), components.end())) {
            throw std::invalid_argument("Components must be given in sorted order");
        }
        if (debug) {
        std::cout << "components: size=" << components.size()
          << " cap=" << components.capacity() << std::endl;
        std::cout << "values: size=" << values.size()
          << " cap=" << values.capacity() << std::endl;
                  std::cout << "offsets: size=" << offsets.size()
          << " cap=" << offsets.capacity() << std::endl;
                            std::cout << "alive: size=" << alive.size()
          << " cap=" << alive.capacity() << std::endl;
        }
        this->n_vecs += 1;
        if (components.back() >= this->d) {
            this->d = components.back() + 1;
        }

        this->components.insert(this->components.end(), components.begin(), components.end());
        this->values.insert(this->values.end(), values.begin(), values.end());
        this->offsets.push_back(this->components.size());
        this->alive.push_back(true);
        if (debug) {
                    std::cout << "components: size=" << components.size()
          << " cap=" << components.capacity() << std::endl;
        std::cout << "values: size=" << values.size()
          << " cap=" << values.capacity() << std::endl;
                  std::cout << "offsets: size=" << offsets.size()
          << " cap=" << offsets.capacity() << std::endl;
                            std::cout << "alive: size=" << alive.size()
          << " cap=" << alive.capacity() << std::endl;
        }
    }

    void set_dead(size_t id) {
        if (id >= len()) {
            throw std::out_of_range("set_dead: The id is out of range");
        }

        this->alive[id] = false;
    }

    bool is_alive(size_t id) const {
        if (id >= len()) {
            throw std::out_of_range("is_alive: The id is out of range");
        }

        return this->alive[id];
    }

    /**
     * Adds a new sparse vector to the dataset using pairs of components and values.
     * 
     * @param pairs The pairs of components and values.
     * @throws std::invalid_argument if pairs is empty or if components is not sorted.
     */
    void push_pairs(const std::vector<std::pair<uint16_t, T>>& pairs) {
        if (pairs.empty()) {
            throw std::invalid_argument("Pairs cannot be empty");
        }

        for (size_t i = 1; i < pairs.size(); ++i) {
            if (pairs[i-1].first >= pairs[i].first) {
                throw std::invalid_argument("Components must be given in sorted order");
            }
        }

        this->n_vecs += 1;
        if (pairs.back().first >= this->d) {
            this->d = pairs.back().first + 1;
        }

        for (const auto& [c, v] : pairs) {
            this->components.push_back(c);
            this->values.push_back(v);
        }
        this->offsets.push_back(this->components.size());
        this->alive.push_back(true);
    }

    size_t id_to_offset(size_t id) const {
        if (id >= n_vecs) {
            throw std::out_of_range("id_to_offset::The id is out of range");
        }
        return offsets[id];
    }

    /**
     * Returns the length of the vector with the specified index.
     * 
     * @param id The index of the vector.
     * @return The length of the vector.
     * @throws std::out_of_range if the specified index is out of range.
     */
    size_t vector_len(size_t id) const {
        if (id >= len()) {
            throw std::out_of_range("vector_len::The id is out of range");
        }

        return offsets[id + 1] - offsets[id];
    }

    std::pair<std::vector<uint16_t>, std::vector<T>> get_with_offset(size_t offset, size_t len) const {
        if (offset + len > components.size()) {
            throw std::out_of_range("The offset + len is out of range");
        }

        std::vector<uint16_t> v_components(components.begin() + offset, components.begin() + offset + len);
        std::vector<T> v_values(values.begin() + offset, values.begin() + offset + len);

        return {v_components, v_values};
    }

    size_t vector_offset(size_t id) const {
        if (id >= len()) {
            throw std::out_of_range("vector_offset::The id is out of range");
        }

        return offsets[id];
    }

    /**
     * Returns the number of vectors in the dataset.
     * 
     * @return The number of vectors in the dataset.
     */
    size_t len() const {
        return offsets.size() - 1;
    }

    /**
     * Checks if the dataset is empty.
     * 
     * @return True if the dataset is empty, false otherwise.
     */
    bool is_empty() const {
        return len() == 0;
    }

    /**
     * Returns the number of components of the dataset.
     * 
     * @return The number of components of the dataset.
     */
    size_t dim() const {
        return d;
    }

    /**
     * Returns the number of non-zero components in the dataset.
     * 
     * @return The number of non-zero components in the dataset.
     */
    size_t nnz() const {
        return components.size();
    }

    /**
     * Retrieves the components and values of the sparse vector at the specified index.
     * 
     * @param id The index of the sparse vector to retrieve.
     * @return A pair containing views of components and values of the sparse vector.
     * @throws std::out_of_range if the specified index is out of range.
     */
    std::pair<std::vector<uint16_t>, std::vector<T>> get(size_t id) const {
        if (id >= len()) {
            throw std::out_of_range("get::The id is out of range");
        }

        auto [start, end] = SparseDataset<T>::vector_range(offsets, id);

        std::vector<uint16_t> v_components(components.begin() + start, components.begin() + end);
        std::vector<T> v_values(values.begin() + start, values.begin() + end);

        return {v_components, v_values};
    }

    /**
     * Zero-copy view of a document's (components, values) directly into the
     * dataset's contiguous storage. Unlike get(), this allocates nothing, which
     * matters in hot build loops (blocking/summarization) that touch every doc
     * against every centroid. The view is valid only while the dataset is not
     * mutated (push/set_dead do not move existing entries, but a reallocation
     * from push() would invalidate outstanding views).
     */
    struct VectorView {
        const uint16_t* components;
        const T* values;
        size_t len;
    };

    VectorView get_view(size_t id) const {
        if (id >= len()) {
            throw std::out_of_range("get_view::The id is out of range");
        }
        size_t start = offsets[id];
        size_t end = offsets[id + 1];
        return VectorView{components.data() + start, values.data() + start, end - start};
    }

    /**
     * Converts the mutable dataset to an immutable one.
     * 
     * @return An immutable SparseDataset.
     */
    SparseDataset<T> to_immutable() const {
        return SparseDataset<T>(len(), d, offsets, components, values);
    }

    /**
     * Returns the space usage of the dataset in bytes.
     * 
     * @return The space usage of the dataset in bytes.
     */
    size_t space_usage_byte() const {
        return sizeof(d) + 
               offsets.size() * sizeof(size_t) +
               components.size() * sizeof(uint16_t) +
               values.size() * sizeof(T) + 
               alive.size() * sizeof(bool);
    }
    
    template <class Archive>
    void serialize(Archive& archive) {
        archive(d, n_vecs, offsets, components, values, alive);
    }
};

} // namespace seismic

#endif // SPARSE_DATASET_H
