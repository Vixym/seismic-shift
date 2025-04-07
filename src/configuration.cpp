#include "inverted_index.h"

namespace seismic {

// Builder pattern methods implementation for Configuration class
Configuration& Configuration::pruning_strategy(const PruningStrategy& pruning) {
    this->pruning = pruning;
    return *this;
}

Configuration& Configuration::blocking_strategy(const BlockingStrategy& blocking) {
    this->blocking = blocking;
    return *this;
}

Configuration& Configuration::summarization_strategy(const SummarizationStrategy& summarization) {
    this->summarization = summarization;
    return *this;
}

Configuration& Configuration::knn(const KnnConfiguration& knn_config) {
    this->knn_config = knn_config;
    return *this;
}

Configuration& Configuration::batched_indexing(const std::optional<size_t>& batch_size) {
    this->batched_indexing_size = batch_size;
    return *this;
}

} // namespace seismic
