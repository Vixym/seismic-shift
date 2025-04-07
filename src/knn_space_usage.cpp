#include "inverted_index.h"

namespace seismic {

size_t Knn::space_usage_byte() const {
    size_t total_size = 0;
    
    // Size of member variables
    total_size += sizeof(n_vecs);
    total_size += sizeof(d);
    total_size += sizeof(nbits);
    
    // Size of BitVector
    // Assuming BitVector is a vector-like structure with a size() method
    total_size += neighbours.size() * sizeof(uint64_t) / 8; // Convert bits to bytes
    
    return total_size;
}

} // namespace seismic
