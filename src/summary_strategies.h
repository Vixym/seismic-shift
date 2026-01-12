#ifndef SUMMARY_STRATEGIES_H
#define SUMMARY_STRATEGIES_H

namespace seismic {
    
using SparseVector = std::vector<std::pair<uint16_t, float>>;

// Required interface for summarization strategy
// class SummaryBase
// {
// public:
//     virtual SparseVector summary_init() const = 0;
//     virtual SparseVector summary_insert(const SparseVector& summary, const SparseVector& vec) const = 0;
//     virtual SparseVector summary_delete(const SparseVector& summary, const SparseVector& vec) const = 0;
//     virtual ~SummaryBase() = default;
// };

class MaxSummary
{
public:
    MaxSummary() = default;

    static std::pair<size_t, size_t> unpack_offset_len(uint64_t pack) { return {static_cast<size_t>(pack >> 16), static_cast<size_t>(pack & 0xFFFF)}; }
    
    static SparseVector summary_init_packed(
        const SparseDatasetMut<float>& dataset,
        const std::vector<size_t>& block,
        const size_t n_components)
    {
        // For each component in the document, store the largest value seen so far
        std::unordered_map<uint16_t, float> hash;
            
        // For each document in the block
        for (size_t packed_posting : block) {
            // Skip doc is doc is dead
            auto [offset, len] = unpack_offset_len(packed_posting);
            size_t doc_id = dataset.offset_to_id(offset);
            if (!dataset.is_alive(doc_id)) continue;

            auto [components, values] = dataset.get(doc_id);
                
            for (size_t i = 0; i < components.size(); ++i) {
                uint16_t component_id = components[i];
                float value = values[i];
                auto it = hash.find(component_id);
                if (it == hash.end() || it->second < value) {
                    hash[component_id] = value;
                }
            }
        }

        // Convert hash to vector
        std::vector<std::pair<uint16_t, float>> components_values(hash.begin(), hash.end());

        // If summary vector too large, truncate
        if (components_values.size() > n_components) {
            std::sort(components_values.begin(), components_values.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            components_values.resize(n_components);
        }

        // Sort by component
        std::sort(components_values.begin(), components_values.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

        return components_values;
    }

    static SparseVector summary_init_packed_energy_preserving(
        const SparseDatasetMut<float>& dataset,
        const std::vector<size_t>& block,
        const float fraction)
    {
        std::unordered_map<uint16_t, float> hash;
            
        // For each document in the block
        for (size_t packed_posting : block) {
            // Skip doc is doc is dead
            auto [offset, len] = unpack_offset_len(packed_posting);
            size_t doc_id = dataset.offset_to_id(offset);
            if (!dataset.is_alive(doc_id)) continue;

            // For each component in the document, store the largest value seen so far
            auto [components, values] = dataset.get(doc_id);
            
            for (size_t i = 0; i < components.size(); ++i) {
                uint16_t component_id = components[i];
                float value = values[i];
                auto it = hash.find(component_id);
                if (it == hash.end() || it->second < value) {
                    hash[component_id] = value;
                }
            }
        }

        // Convert to vector of pairs for sorting
        std::vector<std::pair<uint16_t, float>> components_values(hash.begin(), hash.end());

        // Sort by decreasing values
        std::sort(components_values.begin(), components_values.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Calculate total energy
        float total_sum = 0.0f;
        for (const auto& [_, value] : components_values) {
            total_sum += static_cast<float>(value);
        }

        // Select components that preserve the desired energy fraction
        SparseVector summary_vector;
        float acc = 0.0f;
        for (const auto& [tid, value] : components_values) {
            acc += static_cast<float>(value);
            summary_vector.emplace_back(tid, value);

            if ((acc / total_sum) > fraction) {
                break;
            }
        }

        // Sort summary vector by component
        std::sort(summary_vector.begin(), summary_vector.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });

        return summary_vector;
    }

    static SparseVector summary_init_energy_preserving(
        const SparseDatasetMut<float>& dataset,
        const std::vector<size_t>& block,
        const float fraction
        )
    {
        std::unordered_map<uint16_t, float> hash;
            
        // For each document in the block
        for (size_t doc_id : block) {
            // For each component in the document, store the largest value seen so far
            auto [components, values] = dataset.get(doc_id);
            
            for (size_t i = 0; i < components.size(); ++i) {
                uint16_t component_id = components[i];
                float value = values[i];
                auto it = hash.find(component_id);
                if (it == hash.end() || it->second < value) {
                    hash[component_id] = value;
                }
            }
        }

        // Convert to vector of pairs for sorting
        std::vector<std::pair<uint16_t, float>> components_values(hash.begin(), hash.end());

        // Sort by decreasing values
        std::sort(components_values.begin(), components_values.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Calculate total energy
        float total_sum = 0.0f;
        for (const auto& [_, value] : components_values) {
            total_sum += static_cast<float>(value);
        }

        // Select components that preserve the desired energy fraction
        SparseVector summary_vector;
        float acc = 0.0f;
        for (const auto& [tid, value] : components_values) {
            acc += static_cast<float>(value);
            summary_vector.emplace_back(tid, value);

            if ((acc / total_sum) > fraction) {
                break;
            }
        }

        // Sort summary vector by component
        std::sort(summary_vector.begin(), summary_vector.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });

        return summary_vector;
    }

    static SparseVector summary_init_fixed_size(
        const SparseDatasetMut<float>& dataset,
        const std::vector<size_t>& block,
        const size_t n_components)
    {
        // For each component in the document, store the largest value seen so far
        std::unordered_map<uint16_t, float> hash;
            
        // For each document in the block
        for (size_t doc_id : block) {
            // Skip doc is doc is dead
            if (!dataset.is_alive(doc_id)) continue;

            auto [components, values] = dataset.get(doc_id);
                
            for (size_t i = 0; i < components.size(); ++i) {
                uint16_t component_id = components[i];
                float value = values[i];
                auto it = hash.find(component_id);
                if (it == hash.end() || it->second < value) {
                    hash[component_id] = value;
                }
            }
        }

        // Convert hash to vector
        std::vector<std::pair<uint16_t, float>> components_values(hash.begin(), hash.end());

        // If summary vector too large, truncate
        if (components_values.size() > n_components) {
            std::sort(components_values.begin(), components_values.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            components_values.resize(n_components);
        }

        // Sort by component
        std::sort(components_values.begin(), components_values.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

        return components_values;
    }

    static SparseVector summary_insert(
        const SparseVector& summary,
        const SparseVector& vec)
    {
        size_t i = 0, j = 0;
        SparseVector result;

        while (i < summary.size() && j < vec.size()) {
            uint16_t summary_c = summary[i].first;
            uint16_t vec_c = vec[j].first;

            if (summary_c == vec_c) {
                result.emplace_back(summary_c, std::max(summary[i].second, vec[j].second));
                ++i;
                ++j;
            } else if (summary_c < vec_c) {
                result.emplace_back(summary_c, summary[i].second);
                ++i;
            } else if (summary_c > vec_c) {
                result.emplace_back(vec_c, vec[j].second);
                ++j;
            }
        }

        while (i < summary.size()) {
            result.emplace_back(summary[i].first, summary[i].second);
            ++i;
        }

        while (j < vec.size()) {
            result.emplace_back(vec[j].first, vec[j].second);
            ++j;
        }

        // If summary vector too large, resize
        // if (result.size() > n_components) {
        //     std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        //     result.resize(n_components);
        //     std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        // }
        
        return result;
    }

    static SparseVector summary_delete(
        const SparseDatasetMut<float>& dataset,
        const SparseVector& summary, 
        const SparseVector& vec,
        const std::vector<size_t>& block,
        const size_t n_components,
        const float fraction)
    {
        (void)summary;
        (void)vec;
        (void)n_components;
        return summary_init_packed_energy_preserving(dataset, block, fraction);
    }
};

class CentroidSummary
{
public:
    CentroidSummary() = default;

    static SparseVector summary_init_energy_preserving(
        const SparseDatasetMut<float>& dataset,
        const std::vector<size_t>& block,
        const float fraction
        )
    {
        std::unordered_map<uint16_t, float> hash;
            
        // For each document in the block
        for (size_t doc_id : block) {
            // For each component in the document, accumulate values
            auto [components, values] = dataset.get(doc_id);
            
            for (size_t i = 0; i < components.size(); ++i) {
                uint16_t component_id = components[i];
                float value = values[i];
                auto it = hash.find(component_id);
                if (it == hash.end()){
                    hash[component_id] = value;
                } else {
                    hash[component_id] += value;
                }
            }
        }

        // Convert to vector of pairs for sorting
        std::vector<std::pair<uint16_t, float>> components_values(hash.begin(), hash.end());

        // Sort by decreasing values
        std::sort(components_values.begin(), components_values.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Calculate total energy
        float total_sum = 0.0f;
        for (const auto& [_, value] : components_values) {
            total_sum += static_cast<float>(value);
        }

        // Select components that preserve the desired energy fraction
        SparseVector summary_vector;
        float acc = 0.0f;
        for (const auto& [tid, value] : components_values) {
            acc += static_cast<float>(value);
            summary_vector.emplace_back(tid, value/block.size());

            if ((acc / total_sum) > fraction) {
                break;
            }
        }

        // Sort summary vector by component
        std::sort(summary_vector.begin(), summary_vector.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });

        return summary_vector;
    }

    static SparseVector summary_init_fixed_size(
        const SparseDatasetMut<float>& dataset,
        const std::vector<size_t>& block,
        const size_t n_components)
    {
        // For each component in the document, store the cumulative sum
        std::unordered_map<uint16_t, float> hash;
            
        // For each document in the block
        for (size_t doc_id : block) {
            // Skip doc is doc is dead
            if (!dataset.is_alive(doc_id)) continue;

            auto [components, values] = dataset.get(doc_id);
                
            for (size_t i = 0; i < components.size(); ++i) {
                uint16_t component_id = components[i];
                auto it = hash.find(component_id);
                if (it == hash.end()) {
                    hash[component_id] = 0;
                }
                hash[component_id] += values[i];
            }
        }

        // Convert hash to vector
        std::vector<std::pair<uint16_t, float>> components_values(hash.begin(), hash.end());

        // If summary vector too large, truncate
        if (components_values.size() > n_components) {
            std::sort(components_values.begin(), components_values.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            components_values.resize(n_components);
        }

        // Sort by component
        std::sort(components_values.begin(), components_values.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

        // Normalize values
        for (size_t i = 0; i < components_values.size(); ++i) {
            components_values[i].second /= components_values.size();
        }

        return components_values;
    }

    static SparseVector summary_insert(
        const SparseVector& summary,
        const SparseVector& vec,
        const size_t num_docs)
    {
        size_t i = 0, j = 0;
        SparseVector result;

        while (i < summary.size() && j < vec.size()) {
            uint16_t summary_c = summary[i].first;
            uint16_t vec_c = vec[j].first;

            if (summary_c == vec_c) {
                result.emplace_back(summary_c, (summary[i].second*num_docs+vec[j].second)/(num_docs+1));
                ++i;
                ++j;
            } else if (summary_c < vec_c) {
                result.emplace_back(summary_c, (summary[i].second*num_docs)/(num_docs+1));
                ++i;
            } else if (summary_c > vec_c) {
                result.emplace_back(vec_c, (vec[j].second*num_docs)/(num_docs+1));
                ++j;
            }
        }

        while (i < summary.size()) {
            result.emplace_back(summary[i].first, (summary[i].second*num_docs)/(num_docs+1));
            ++i;
        }

        while (j < vec.size()) {
            result.emplace_back(vec[j].first, (vec[j].second*num_docs)/(num_docs+1));
            ++j;
        }

        // // If summary vector too large, resize
        // if (result.size() > n_components) {
        //     std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        //     result.resize(n_components);
        //     std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        // }
        
        return result;
    }

    static SparseVector summary_delete(
        const SparseVector& summary,
        const SparseVector& vec,
        const size_t num_docs)
    {
        size_t i = 0, j = 0;
        SparseVector result;

        while (i < summary.size() && j < vec.size()) {
            uint16_t summary_c = summary[i].first;
            uint16_t vec_c = vec[j].first;

            if (summary_c == vec_c) {
                result.emplace_back(summary_c, (summary[i].second*num_docs-vec[j].second)/(num_docs-1));
                ++i;
                ++j;
            } else if (summary_c < vec_c) {
                result.emplace_back(summary_c, (summary[i].second*num_docs)/(num_docs-1));
                ++i;
            } else if (summary_c > vec_c) {
                ++j;
            }
        }

        while (i < summary.size()) {
            result.emplace_back(summary[i].first, (summary[i].second*num_docs)/(num_docs-1));
            ++i;
        }

        // // If summary vector too large, resize
        // if (result.size() > n_components) {
        //     std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        //     result.resize(n_components);
        //     std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        // }
        
        return result;
    }
};

} // namespace seismic

#endif // SUMMARY_STRATEGIES_H