#include "inverted_index.h"

namespace seismic {

template <typename T>
size_t InvertedIndex<T>::space_usage_byte() const {
    size_t total_size = 0;
    
    // Add size of forward index
    total_size += forward_index_.space_usage_byte();
    
    // Add size of posting lists
    for (const auto& posting_list : posting_lists_) {
        total_size += posting_list.space_usage_byte();
    }
    
    // Add size of KNN graph if present
    if (knn_) {
        // Assuming a simple calculation for KNN graph size
        // In a real implementation, you would call knn_->space_usage_byte()
        total_size += knn_->space_usage_byte();
    }
    
    return total_size;
}

// Explicit template instantiations for the types we need
template size_t InvertedIndex<float>::space_usage_byte() const;
template size_t InvertedIndex<double>::space_usage_byte() const;
template size_t InvertedIndex<Float16>::space_usage_byte() const;

} // namespace seismic
