#include "inverted_index.h"

namespace seismic {

// Builder pattern methods implementation for Configuration class
Configuration& Configuration::set_dynamic_support(const bool dynamic_support) {
    this->dynamic_support = dynamic_support;
    return *this;
}

Configuration& Configuration::set_transform_function(const std::string& transform_function) {
    this->transform_function = transform_function;
    return *this;
}

Configuration& Configuration::set_summarization_metric(const std::string& summarization_metric) {
    this->summarization_metric = summarization_metric;
    return *this;
}

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
