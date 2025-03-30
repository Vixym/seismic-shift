#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <cstdint>
#include <optional>
#include <type_traits>
#include <limits>
#include "space_usage.h"

namespace seismic {

/**
 * Marker for types used as values in a dataset
 * Equivalent to the Rust DataType trait
 */
template <typename T>
struct is_data_type {
    static constexpr bool value = 
        std::is_copy_constructible_v<T> && 
        std::is_copy_assignable_v<T> &&
        std::is_default_constructible_v<T> &&
        std::is_move_constructible_v<T> &&
        std::is_move_assignable_v<T>;
};

// Specializations for numeric types
template <> struct is_data_type<float> : std::true_type {};
template <> struct is_data_type<double> : std::true_type {};
template <> struct is_data_type<Float16> : std::true_type {};

// Extend the existing Float16 class with DataType functionality
float to_f32(const Float16& f16);

// Create a Float16 from a float value
Float16 from_float(float value);

// Zero value for Float16
inline Float16 zero(const Float16&) { return Float16(); }

// Comparison operators for Float16
inline bool operator<(const Float16& a, const Float16& b) {
    return to_f32(a) < to_f32(b);
}

inline bool operator<=(const Float16& a, const Float16& b) {
    return to_f32(a) <= to_f32(b);
}

inline bool operator>(const Float16& a, const Float16& b) {
    return to_f32(a) > to_f32(b);
}

inline bool operator>=(const Float16& a, const Float16& b) {
    return to_f32(a) >= to_f32(b);
}

inline bool operator==(const Float16& a, const Float16& b) {
    return a.get_raw() == b.get_raw();
}

inline bool operator!=(const Float16& a, const Float16& b) {
    return a.get_raw() != b.get_raw();
}

// Arithmetic operators for Float16
inline Float16 operator+(const Float16& a, const Float16& b) {
    return from_float(to_f32(a) + to_f32(b));
}

inline Float16 operator-(const Float16& a, const Float16& b) {
    return from_float(to_f32(a) - to_f32(b));
}

inline Float16 operator*(const Float16& a, const Float16& b) {
    return from_float(to_f32(a) * to_f32(b));
}

inline Float16 operator/(const Float16& a, const Float16& b) {
    return from_float(to_f32(a) / to_f32(b));
}

// Extension methods for built-in types

// float
inline float to_f32(float value) { return value; }
inline float zero(float) { return 0.0f; }

// double
inline float to_f32(double value) { return static_cast<float>(value); }
inline double zero(double) { return 0.0; }

} // namespace seismic

#endif // DATA_TYPE_H
