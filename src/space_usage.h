#ifndef SPACE_USAGE_H
#define SPACE_USAGE_H

#include <cstddef>
#include <vector>
#include <memory>
#include <type_traits>

namespace seismic {

/**
 * A trait to report the space usage of a data structure.
 */
class SpaceUsage {
public:
    virtual ~SpaceUsage() = default;

    /**
     * Gives the space usage of the data structure in bytes.
     */
    virtual std::size_t space_usage_byte() const = 0;

    /**
     * Gives the space usage of the data structure in KiB.
     */
    double space_usage_KiB() const {
        std::size_t bytes = space_usage_byte();
        return static_cast<double>(bytes) / 1024.0;
    }

    /**
     * Gives the space usage of the data structure in MiB.
     */
    double space_usage_MiB() const {
        std::size_t bytes = space_usage_byte();
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    }

    /**
     * Gives the space usage of the data structure in GiB.
     */
    double space_usage_GiB() const {
        std::size_t bytes = space_usage_byte();
        return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    }
};

/**
 * Implementation of SpaceUsage for std::vector<T>
 * Note: This is correct only for primitive types or types that properly implement SpaceUsage.
 */
template<typename T>
class VectorSpaceUsage : public SpaceUsage {
private:
    const std::vector<T>& vec;

public:
    explicit VectorSpaceUsage(const std::vector<T>& v) : vec(v) {}

    std::size_t space_usage_byte() const override {
        return sizeof(std::vector<T>) + sizeof(T) * vec.capacity();
    }
};

/**
 * Implementation of SpaceUsage for std::unique_ptr<T[]>
 * Note: This is correct only for primitive types or types that properly implement SpaceUsage.
 */
template<typename T>
class ArraySpaceUsage : public SpaceUsage {
private:
    const T* array;
    std::size_t length;

public:
    ArraySpaceUsage(const T* arr, std::size_t len) : array(arr), length(len) {}

    std::size_t space_usage_byte() const override {
        if (length == 0) {
            return sizeof(std::unique_ptr<T[]>);
        } else {
            // For primitive types, just use sizeof(T) * length
            if constexpr (std::is_arithmetic_v<T>) {
                return sizeof(std::unique_ptr<T[]>) + sizeof(T) * length;
            } 
            // For types that implement SpaceUsage
            else if constexpr (std::is_base_of_v<SpaceUsage, T>) {
                return sizeof(std::unique_ptr<T[]>) + array[0].space_usage_byte() * length;
            }
            // Default fallback
            else {
                return sizeof(std::unique_ptr<T[]>) + sizeof(T) * length;
            }
        }
    }
};

/**
 * Macro to implement SpaceUsage for primitive types
 */
#define IMPLEMENT_SPACE_USAGE(TYPE) \
    template<> \
    class PrimitiveSpaceUsage<TYPE> : public SpaceUsage { \
    public: \
        std::size_t space_usage_byte() const override { \
            return sizeof(TYPE); \
        } \
    }

/**
 * Template class for primitive types space usage
 */
template<typename T>
class PrimitiveSpaceUsage : public SpaceUsage {
public:
    std::size_t space_usage_byte() const override {
        return sizeof(T);
    }
};

// Implement space usage for primitive types
using Bool = PrimitiveSpaceUsage<bool>;
using Int8 = PrimitiveSpaceUsage<int8_t>;
using UInt8 = PrimitiveSpaceUsage<uint8_t>;
using Int16 = PrimitiveSpaceUsage<int16_t>;
using UInt16 = PrimitiveSpaceUsage<uint16_t>;
using Int32 = PrimitiveSpaceUsage<int32_t>;
using UInt32 = PrimitiveSpaceUsage<uint32_t>;
using Int64 = PrimitiveSpaceUsage<int64_t>;
using UInt64 = PrimitiveSpaceUsage<uint64_t>;
using Int128 = PrimitiveSpaceUsage<__int128_t>;
using UInt128 = PrimitiveSpaceUsage<__uint128_t>;
using SizeT = PrimitiveSpaceUsage<std::size_t>;
using Float32 = PrimitiveSpaceUsage<float>;
using Float64 = PrimitiveSpaceUsage<double>;

// Half-precision float (f16) implementation
class Float16 : public SpaceUsage {
private:
    uint16_t value; // Simplified representation of f16

public:
    Float16() : value(0) {}
    explicit Float16(uint16_t v) : value(v) {}

    std::size_t space_usage_byte() const override {
        return sizeof(Float16);
    }
};

} // namespace seismic

#endif // SPACE_USAGE_H
