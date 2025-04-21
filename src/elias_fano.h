// Minimal implementation of Elias-Fano encoding.
// This implementation is inspired by C++ implementation by Giuseppe Ottaviano
// (https://github.com/ot/succinct/blob/master/elias_fano.hpp)
//
// TODO: translated from Seismic's Rust as an exercise, but will likely
// later just redirect this Elias-Fano usage to Ottaviano's implementation.

#ifndef ELIAS_FANO_H
#define ELIAS_FANO_H

#include <vector>
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <optional>

namespace seismic
{

    // Forward declarations
    class BitVector;
    class BitVectorMut;
    template <bool UseSelect>
    class DArray;
    class EliasFanoBuilder;
    class EliasFano;

    // Utility functions
    inline uint64_t msb(uint64_t x)
    {
        return x ? 63 - __builtin_clzll(x) : 0;
    }

    // BitVector class for storing and accessing bits
    class BitVector
    {
    private:
        std::vector<uint64_t> data_;
        size_t size_ = 0;

    public:
        BitVector() = default;
        BitVector(const std::vector<uint64_t> &data, size_t size) : data_(data), size_(size) {}

        size_t size() const
        {
            return size_;
        }

        bool get(size_t pos) const
        {
            assert(pos < size_);
            return (data_[pos / 64] >> (pos % 64)) & 1;
        }

        uint64_t get_bits_unchecked(size_t pos, size_t len) const
        {
            if (len == 0)
                return 0;

            size_t block = pos / 64;
            size_t offset = pos % 64;

            uint64_t result = data_[block] >> offset;

            // If we cross a block boundary, we need to get bits from the next block
            if (offset + len > 64 && block + 1 < data_.size())
            {
                result |= data_[block + 1] << (64 - offset);
            }

            // Mask out any extra bits
            return result & ((1ULL << len) - 1);
        }

        size_t space_usage_byte() const
        {
            return data_.size() * sizeof(uint64_t) + sizeof(size_);
        }

        template <class Archive>
        void serialize(Archive& archive) {
            archive(data_, size_); \
        }
    };

    // Mutable BitVector for building
    class BitVectorMut
    {
    private:
        std::vector<uint64_t> data_;
        size_t size_ = 0;

    public:
        BitVectorMut() = default;

        explicit BitVectorMut(size_t num_bits) : size_(0)
        {
            data_.resize((num_bits + 63) / 64, 0);
        }

        static BitVectorMut with_zeros(size_t num_bits)
        {
            BitVectorMut bv;
            bv.data_.resize((num_bits + 63) / 64, 0);
            bv.size_ = num_bits;
            return bv;
        }

        void set(size_t pos, bool value)
        {
            assert(pos < size_);
            if (value)
            {
                data_[pos / 64] |= (1ULL << (pos % 64));
            }
            else
            {
                data_[pos / 64] &= ~(1ULL << (pos % 64));
            }
        }

        void append_bits(uint64_t bits, size_t len)
        {
            if (len == 0)
                return;

            size_t pos = size_;
            size_ += len;

            // Ensure we have enough space
            if ((size_ + 63) / 64 > data_.size())
            {
                data_.resize((size_ + 63) / 64, 0);
            }

            size_t block = pos / 64;
            size_t offset = pos % 64;

            // Set bits in the current block
            data_[block] |= (bits << offset);

            // If we cross a block boundary, set bits in the next block
            if (offset + len > 64 && block + 1 < data_.size())
            {
                data_[block + 1] |= (bits >> (64 - offset));
            }
        }

        BitVector into() const
        {
            return BitVector(data_, size_);
        }
    };

    // DArray class for rank/select operations
    template <bool UseSelect>
    class DArray
    {
    private:
        BitVector bv_;
        std::vector<size_t> rank_samples_;
        std::vector<size_t> select_samples_;
        size_t num_ones_ = 0;
        static constexpr size_t sample_rate_ = 64; // Typical sampling rate

    public:
        DArray() = default;

        explicit DArray(const BitVector &bv) : bv_(bv)
        {
            build_index();
        }

        size_t rank1(size_t pos) const
        {
            if (pos >= bv_.size())
                return num_ones_;

            size_t sample_idx = pos / sample_rate_;
            size_t base_rank = rank_samples_[sample_idx];
            size_t start_pos = sample_idx * sample_rate_;

            // Count 1s in the range [start_pos, pos)
            for (size_t i = start_pos; i < pos; ++i)
            {
                if (bv_.get(i))
                    base_rank++;
            }

            return base_rank;
        }

        std::optional<size_t> select1(size_t idx) const
        {
            if (idx >= num_ones_)
                return std::nullopt;

            // Use select samples to get an approximate position
            size_t sample_idx = idx / sample_rate_;
            size_t start_pos = (sample_idx < select_samples_.size()) ? select_samples_[sample_idx] : 0;
            size_t count = sample_idx * sample_rate_;

            // Linear search from the approximate position
            while (count <= idx)
            {
                if (bv_.get(start_pos))
                    count++;
                if (count > idx)
                    return start_pos;
                start_pos++;
            }

            return start_pos;
        }

        size_t space_usage_byte() const
        {
            return bv_.space_usage_byte() +
                   rank_samples_.size() * sizeof(size_t) +
                   select_samples_.size() * sizeof(size_t) +
                   sizeof(num_ones_);
        }
        
        template <class Archive>
        void serialize(Archive& archive) {
            archive(bv_, rank_samples_, select_samples_, num_ones_, sample_rate_); \
        }

    private:
        void build_index()
        {
            size_t n = bv_.size();
            rank_samples_.resize((n + sample_rate_ - 1) / sample_rate_ + 1, 0);

            // Build rank samples
            size_t cur_rank = 0;
            for (size_t i = 0; i < n; ++i)
            {
                if (i % sample_rate_ == 0)
                {
                    rank_samples_[i / sample_rate_] = cur_rank;
                }
                if (bv_.get(i))
                    cur_rank++;
            }

            num_ones_ = cur_rank;

            // Build select samples
            select_samples_.resize((num_ones_ + sample_rate_ - 1) / sample_rate_, 0);
            cur_rank = 0;
            for (size_t i = 0; i < n; ++i)
            {
                if (bv_.get(i))
                {
                    if (cur_rank % sample_rate_ == 0)
                    {
                        select_samples_[cur_rank / sample_rate_] = i;
                    }
                    cur_rank++;
                }
            }
        }
    };

    // Builder for EliasFano
    class EliasFanoBuilder
    {
    private:
        BitVectorMut high_bits;
        BitVectorMut low_bits;
        size_t universe;
        size_t num_vals;
        size_t pos = 0;
        size_t last = 0;
        size_t low_len;

    public:
        // Creates a new, empty Elias-Fano builder to store n values up to u (u excluded)
        EliasFanoBuilder(size_t universe, size_t num_vals)
            : universe(universe), num_vals(num_vals)
        {
            assert(num_vals > 0 && "The number of values num_vals must not be zero.");

            low_len = msb(universe / num_vals);
            high_bits = BitVectorMut::with_zeros((num_vals + 1) + (universe >> low_len) + 1);
            low_bits = BitVectorMut();
        }

        // Pushes integer val at the end
        void push(size_t val)
        {
            assert(last <= val &&
                   "val must be no less than the last inserted one.");
            assert(val < universe &&
                   "val must be less than universe().");
            assert(pos < num_vals &&
                   "The number of pushed integers must not exceed num_vals().");

            last = val;
            uint64_t low_mask = (1ULL << low_len) - 1;

            if (low_len != 0)
            {
                low_bits.append_bits(val & low_mask, low_len);
            }

            high_bits.set((val >> low_len) + pos, true);
            pos++;
        }

        // Appends integers at the end
        void extend(const std::vector<size_t> &vals)
        {
            for (size_t x : vals)
            {
                push(x);
            }
        }

        // Builds EliasFano from the pushed integers
        EliasFano build() const;

        // Returns the universe, i.e., the (exclusive) upper bound of possible integers
        size_t get_universe() const
        {
            return universe;
        }

        // Returns the number of integers that can be stored
        size_t get_num_vals() const
        {
            return num_vals;
        }

        friend class EliasFano;
    };

    // Compressed monotone increasing sequence through Elias-Fano encoding
    class EliasFano
    {
    private:
        DArray<false> high_bits;
        BitVector low_bits;
        size_t low_len = 0;
        size_t universe = 0;
        size_t num_vals = 0;

    public:
        EliasFano() = default;

        // Constructs EliasFano from a vector
        static EliasFano from(const std::vector<size_t> &data)
        {
            if (data.empty())
            {
                return EliasFano();
            }

            // Check that the sequence is monotonically increasing
            for (size_t i = 1; i < data.size(); ++i)
            {
                assert(data[i - 1] <= data[i] && "The sequence must be monotonically increasing.");
            }

            EliasFanoBuilder efb(1 + *data.rbegin(), data.size());
            efb.extend(data);

            return efb.build();
        }

        // Returns the position of the k-th smallest integer, or std::nullopt if k >= num_vals
        std::optional<size_t> select(size_t k) const
        {
            if (k >= num_vals)
            {
                return std::nullopt;
            }

            auto high_pos = high_bits.select1(k);
            if (!high_pos)
                return std::nullopt;

            size_t high_part = (*high_pos - k) << low_len;
            size_t low_part = (low_len > 0) ? low_bits.get_bits_unchecked(k * low_len, low_len) : 0;

            return high_part | low_part;
        }

        // Returns true if the sequence is empty
        bool is_empty() const
        {
            return num_vals == 0;
        }

        // Returns the length of the compressed sequence
        size_t len() const
        {
            return num_vals;
        }

        // Returns the universe, i.e., the (exclusive) upper bound of possible integers
        size_t get_universe() const
        {
            return universe;
        }

        // Returns the space usage in bytes
        size_t space_usage_byte() const
        {
            return high_bits.space_usage_byte() +
                   low_bits.space_usage_byte() +
                   sizeof(low_len) +
                   sizeof(universe) +
                   sizeof(num_vals);
        }

        friend class EliasFanoBuilder;

        template <class Archive>
        void serialize(Archive& archive) {
            archive(high_bits, low_bits, low_len, universe, num_vals); \
        }
    };

    // Implementation of EliasFanoBuilder::build
    inline EliasFano EliasFanoBuilder::build() const
    {
        EliasFano ef;
        ef.high_bits = DArray<false>(high_bits.into());
        ef.low_bits = low_bits.into();
        ef.num_vals = num_vals;
        ef.low_len = low_len;
        ef.universe = universe;
        return ef;
    }

} // namespace seismic

#endif // ELIAS_FANO_H