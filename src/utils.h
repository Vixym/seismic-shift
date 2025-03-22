// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <unordered_set>
#include <string>
#include <functional>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace utils
{

    /*
     * Computes the size of the intersection of two unsorted lists.
     */
    // TODO: enforce Rust's "T: Eq + Hash + Clone" analogously in C++
    template <typename T>
    inline size_t intersection(const std::vector<T> &s, const std::vector<T> &groundtruth)
    {
        std::unordered_set<T> s_set(s.begin(), s.end());
        size_t size = 0;
        for (const auto &v : groundtruth)
        {
            if (s_set.count(v) > 0)
            {
                size++;
            }
        }
        return size;
    }

    /*
     * Prefetch data for read with NTA (non-temporal access) hint
     */
    template <typename T>
    inline void prefetch_read_NTA(const T *data, size_t offset)
    {
        [[maybe_unused]] const int8_t *p = reinterpret_cast<const int8_t *>(data + offset);

#if defined(PREFETCH_ENABLED) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
        _mm_prefetch(reinterpret_cast<const char *>(p), _MM_HINT_NTA);
#elif defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
        _mm_prefetch(reinterpret_cast<const char *>(p), _MM_HINT_NTA);
#endif
#elif defined(PREFETCH_ENABLED) && defined(__aarch64__)
#include <arm_neon.h>
        __builtin_prefetch(p, 0, 0); // PRFM PLDL1KEEP
#endif
    }

    /**
     * Returns the typename of the argument.
     */
    template <typename T>
    inline std::string_view type_of(const T &)
    {
        return typeid(T).name();
    }

    /**
     * Binary search implementation with reduced branching.
     */
    inline size_t binary_search_branchless(const std::vector<uint16_t> &data, uint16_t target)
    {
        size_t base = 0;
        size_t size = data.size();

        while (size > 1)
        {
            size_t mid = size / 2;
            bool cmp = data[base + mid - 1] < target;
            base += cmp * mid; // Avoids a branch by using multiplication
            size -= mid;
        }

        return base;
    }

} // namespace utils

#endif // UTILS_H