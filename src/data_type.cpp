#include "data_type.h"
#include <cmath>
#include <limits>

namespace seismic {

// The get_raw method is already defined in space_usage.h

// IEEE 754 half-precision binary floating-point format: binary16
// 1 bit sign, 5 bits exponent, 10 bits mantissa
Float16 from_float(float v) {
    // Handle special cases
    if (std::isnan(v)) {
        return Float16(0x7E00);  // NaN
    }
    if (std::isinf(v)) {
        uint16_t value = 0x7C00;  // Infinity
        if (v < 0) value |= 0x8000;  // Negative infinity
        return Float16(value);
    }
    
    // Extract sign, exponent, and mantissa
    uint32_t bits = *reinterpret_cast<uint32_t*>(&v);
    uint16_t sign = (bits >> 31) & 0x1;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = bits & 0x7FFFFF;
    
    // Handle special cases
    if (exponent == -127) {  // Zero or subnormal
        return Float16(sign << 15);
    }
    
    // Check if the value is too small for f16
    if (exponent < -14) {
        return Float16(sign << 15);  // Underflow to zero
    }
    
    // Check if the value is too large for f16
    if (exponent > 15) {
        return Float16((sign << 15) | 0x7C00);  // Overflow to infinity
    }
    
    // Convert to f16 format
    uint16_t f16_exponent = exponent + 15;
    uint16_t f16_mantissa = mantissa >> 13;
    
    return Float16((sign << 15) | (f16_exponent << 10) | f16_mantissa);
}

float to_f32(const Float16& f16) {
    // Extract sign, exponent, and mantissa from f16
    uint16_t value = f16.get_raw();
    uint16_t sign = (value >> 15) & 0x1;
    uint16_t exponent = (value >> 10) & 0x1F;
    uint16_t mantissa = value & 0x3FF;
    
    // Handle special cases
    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            return sign ? -0.0f : 0.0f;
        } else {
            // Subnormal number
            float result = std::ldexp(static_cast<float>(mantissa), -24);
            return sign ? -result : result;
        }
    } else if (exponent == 0x1F) {
        if (mantissa == 0) {
            // Infinity
            return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        } else {
            // NaN
            return std::numeric_limits<float>::quiet_NaN();
        }
    }
    
    // Normalized number
    float result = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, exponent - 15);
    return sign ? -result : result;
}

} // namespace seismic
