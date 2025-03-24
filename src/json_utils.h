// json_utils.h
#ifndef JSON_UTILS_H
#define JSON_UTILS_H

#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <variant>
#include <nlohmann/json.hpp>
#include "utils.h"

namespace utils {

/**
 * Represents document ID that can be either a string or a numeric value
 */
class DocIdType {
public:
    // Variant to store either string or size_t
    std::variant<std::string, size_t> value;

    // Constructors
    DocIdType() : value(std::string()) {}
    DocIdType(const std::string& s) : value(s) {}
    DocIdType(size_t n) : value(n) {}

    // Get ID as string regardless of internal type
    std::string as_string() const {
        if (std::holds_alternative<std::string>(value)) {
            return std::get<std::string>(value);
        } else {
            return std::to_string(std::get<size_t>(value));
        }
    }

    // For debugging
    friend std::ostream& operator<<(std::ostream& os, const DocIdType& id) {
        os << id.as_string();
        return os;
    }
};

// Custom JSON serialization for DocIdType
void to_json(nlohmann::json& j, const DocIdType& id) {
    if (std::holds_alternative<std::string>(id.value)) {
        j = std::get<std::string>(id.value);
    } else {
        j = std::get<size_t>(id.value);
    }
}

// Custom JSON deserialization for DocIdType
void from_json(const nlohmann::json& j, DocIdType& id) {
    if (j.is_string()) {
        id.value = j.get<std::string>();
    } else if (j.is_number_unsigned()) {
        id.value = j.get<size_t>();
    } else {
        throw std::runtime_error("Invalid DocIdType format");
    }
}

/**
 * JSON format structure for vector data
 */
class JsonFormat {
private:
    DocIdType id;
    std::unordered_map<std::string, float> vector;

public:
    // Default constructor
    JsonFormat() = default;

    // Constructor with parameters
    JsonFormat(const DocIdType& doc_id, const std::unordered_map<std::string, float>& vec)
        : id(doc_id), vector(vec) {}

    // Getters
    const std::unordered_map<std::string, float>& get_vector() const {
        return vector;
    }

    std::string get_id_as_string() const {
        return id.as_string();
    }

    const DocIdType& get_id() const {
        return id;
    }
};

// Custom JSON serialization for JsonFormat
void to_json(nlohmann::json& j, const JsonFormat& format) {
    j = nlohmann::json{
        {"id", format.get_id()},
        {"vector", format.get_vector()}
    };
}

// Custom JSON deserialization for JsonFormat
void from_json(const nlohmann::json& j, JsonFormat& format) {
    format = JsonFormat(
        j.at("id").get<DocIdType>(),
        j.at("vector").get<std::unordered_map<std::string, float>>()
    );
}

/**
 * Extract data from JsonFormat into a tuple of (id, coordinates, values)
 */
template <typename T>
std::tuple<std::string, std::vector<std::string>, std::vector<T>> 
extract_jsonl(const JsonFormat& json_format) {
    static_assert(utils::DataType<T>, "T must satisfy DataType concept");
    
    std::vector<std::string> coords;
    std::vector<T> values;

    for (const auto& [key, val] : json_format.get_vector()) {
        coords.push_back(key);
        
        T typed_val;
        typed_val.from_f32(val);
        values.push_back(typed_val);
    }

    return {json_format.get_id_as_string(), coords, values};
}

/**
 * Read queries from a JSONL file
 */
std::vector<std::tuple<std::string, std::vector<std::string>, std::vector<float>>> 
read_queries(const std::string& input_file) {
    std::vector<std::tuple<std::string, std::vector<std::string>, std::vector<float>>> result;
    std::ifstream file(input_file);
    
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open " + input_file);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        try {
            auto json = nlohmann::json::parse(line);
            JsonFormat json_format = json.get<JsonFormat>();
            
            auto [id, coords, values] = extract_jsonl<Float>(json_format);
            
            // Convert Float to float for the return type
            std::vector<float> float_values;
            for (const auto& val : values) {
                float_values.push_back(val.as_());
            }
            
            result.emplace_back(id, coords, float_values);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing JSON line: " << e.what() << std::endl;
            std::cerr << "Line content: " << line << std::endl;
        }
    }

    return result;
}

} // namespace utils

#endif // JSON_UTILS_H
