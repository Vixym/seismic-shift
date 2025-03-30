#ifndef QUANTIZED_SUMMARY_H
#define QUANTIZED_SUMMARY_H

#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>

#include "data_type.h"
#include "space_usage.h"
#include "sparse_dataset.h"
#include "elias_fano.h"

namespace seismic {

/**
 * Implementation of a quantized summary for efficient storage and retrieval of sparse datasets.
 * 
 * This class quantizes the values of a sparse dataset to reduce memory usage while maintaining
 * reasonable accuracy for distance calculations.
 */
class QuantizedSummary {
private:
    size_t n_summaries;
    size_t d;
    EliasFano offsets;
    std::vector<uint16_t> summaries_ids; // There cannot be more than 2^16 summaries
    std::vector<uint8_t> values;
    std::vector<float> minimums;
    std::vector<float> quants;

    static constexpr size_t N_CLASSES = 256; // We store quantized values in a u8. Number of classes cannot be more than 256

public:
    // Default constructor
    QuantizedSummary() : n_summaries(0), d(0) {}

    // Constructor with parameters
    QuantizedSummary(size_t n_summaries, size_t d,
                     EliasFano offsets,
                     std::vector<uint16_t> summaries_ids,
                     std::vector<uint8_t> values,
                     std::vector<float> minimums,
                     std::vector<float> quants)
        : n_summaries(n_summaries), d(d), offsets(std::move(offsets)),
          summaries_ids(std::move(summaries_ids)), values(std::move(values)),
          minimums(std::move(minimums)), quants(std::move(quants)) {}

    /**
     * Creates an iterator to calculate distances between the query vector and all vectors in the dataset.
     * 
     * @param query_components The components of the query vector.
     * @param query_values The values of the query vector.
     * @return A DistancesIter object.
     */
    class DistancesIter {
    private:
        size_t current;
        std::vector<float> distances;

    public:
        DistancesIter(const QuantizedSummary* summaries, const std::vector<uint16_t>& query_components, const std::vector<float>& query_values) 
            : current(0) {
            
            distances.resize(summaries->n_summaries, 0.0f);

            for (size_t i = 0; i < query_components.size(); ++i) {
                uint16_t qc = query_components[i];
                float qv = query_values[i];

                if (qc >= summaries->d) {
                    break;
                }

                auto current_offset = summaries->offsets.select(qc);
                auto next_offset = summaries->offsets.select(qc + 1);

                if (!current_offset || !next_offset || *next_offset - *current_offset == 0) {
                    continue;
                }

                for (size_t j = *current_offset; j < *next_offset; ++j) {
                    uint16_t s_id = summaries->summaries_ids[j];
                    uint8_t v = summaries->values[j];
                    float val = v * summaries->quants[s_id] + summaries->minimums[s_id];
                    distances[s_id] += val * qv;
                }
            }
        }

        std::optional<float> next() {
            if (current < distances.size()) {
                return distances[current++];
            }
            return std::nullopt;
        }

        bool has_next() const {
            return current < distances.size();
        }
    };

    /**
     * Creates an iterator to calculate distances between the query vector and all vectors in the dataset.
     * 
     * @param query_components The components of the query vector.
     * @param query_values The values of the query vector.
     * @return A DistancesIter object.
     */
    DistancesIter distances_iter(const std::vector<uint16_t>& query_components, const std::vector<float>& query_values) const {
        return DistancesIter(this, query_components, query_values);
    }

    /**
     * Quantizes a vector of values to reduce memory usage.
     * 
     * @param values The vector of values to quantize.
     * @return A tuple containing the minimum value, the quantization factor, and the quantized values.
     */
    template <typename T>
    static std::tuple<float, float, std::vector<uint8_t>> quantize(const std::vector<T>& values) {
        assert(!values.empty() && "Values cannot be empty");

        // Compute min and max values in the vector
        T min_val = values[0];
        T max_val = values[0];
        
        for (const auto& v : values) {
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
        }

        float min_f = seismic::to_f32(min_val);
        float max_f = seismic::to_f32(max_val);

        // Quantization splits the range [min, max] into N_CLASSES blocks of equal size (max-min)/N_CLASSES
        std::vector<uint8_t> quantized_values;
        quantized_values.reserve(values.size());
        
        float quant = (max_f - min_f) / static_cast<float>(N_CLASSES);
        if (quant == 0) {  // All values are the same
            quant = 1.0f;  // Avoid division by zero
            quantized_values.resize(values.size(), 0);
        } else {
            for (const auto& v : values) {
                float v_f = seismic::to_f32(v);
                // Clamp to [0, N_CLASSES-1] to avoid overflow
                uint8_t q = static_cast<uint8_t>(std::min(static_cast<float>(N_CLASSES - 1), std::max(0.0f, (v_f - min_f) / quant)));
                quantized_values.push_back(q);
            }
        }

        return {min_f, quant, quantized_values};
    }

    /**
     * Returns the space usage of the quantized summary in bytes.
     * 
     * @return The space usage in bytes.
     */
    size_t space_usage_byte() const {
        return sizeof(n_summaries) +
               sizeof(d) +
               offsets.space_usage_byte() +
               summaries_ids.size() * sizeof(uint16_t) +
               values.size() * sizeof(uint8_t) +
               minimums.size() * sizeof(float) +
               quants.size() * sizeof(float);
    }

    /**
     * Creates a QuantizedSummary from a SparseDataset.
     * 
     * @param dataset The sparse dataset to create the quantized summary from.
     * @return A QuantizedSummary object.
     */
    template <typename T>
    static QuantizedSummary from_sparse_dataset(const SparseDataset<T>& dataset) {
        assert(dataset.len() <= std::numeric_limits<uint16_t>::max() && 
               "Number of summaries cannot be more than 2^16");

        // Invert the dataset to map components to summaries
        std::vector<std::vector<std::pair<uint8_t, size_t>>> inverted_pairs(dataset.dim());
        std::vector<float> minimums;
        std::vector<float> quants;

        minimums.reserve(dataset.len());
        quants.reserve(dataset.len());

        for (size_t doc_id = 0; doc_id < dataset.len(); ++doc_id) {
            auto [components, values] = dataset.get(doc_id);
            
            auto [minimum, quant, current_codes] = quantize(values);
            
            minimums.push_back(minimum);
            quants.push_back(quant);
            
            for (size_t i = 0; i < components.size(); ++i) {
                uint16_t c = components[i];
                uint8_t score = current_codes[i];
                inverted_pairs[c].push_back({score, doc_id});
            }
        }

        // Create the quantized summary
        std::vector<size_t> offsets;
        std::vector<uint16_t> summaries_ids;
        std::vector<uint8_t> codes;

        offsets.push_back(0);
        for (const auto& ip : inverted_pairs) {
            for (const auto& [score, id] : ip) {
                codes.push_back(score);
                summaries_ids.push_back(static_cast<uint16_t>(id));
            }
            offsets.push_back(summaries_ids.size());
        }

        return QuantizedSummary(
            dataset.len(),
            dataset.dim(),
            EliasFano::from(offsets),
            std::move(summaries_ids),
            std::move(codes),
            std::move(minimums),
            std::move(quants)
        );
    }
};

} // namespace seismic

#endif // QUANTIZED_SUMMARY_H
