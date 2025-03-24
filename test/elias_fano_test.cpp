#include <gtest/gtest.h>
#include "../src/elias_fano.h"
#include <vector>

using namespace seismic;

// Test the construction of an EliasFano from a vector
TEST(EliasFanoTest, FromVector)
{
    std::vector<size_t> v = {1, 3, 3, 7};
    EliasFano ef = EliasFano::from(v);

    EXPECT_EQ(ef.len(), 4);
    EXPECT_EQ(ef.get_universe(), 8);
    EXPECT_FALSE(ef.is_empty());
}

// Test the select operation
TEST(EliasFanoTest, Select)
{
    std::vector<size_t> v = {1, 3, 3, 7};
    EliasFano ef = EliasFano::from(v);

    EXPECT_EQ(ef.select(0), std::make_optional<size_t>(1));
    EXPECT_EQ(ef.select(1), std::make_optional<size_t>(3));
    EXPECT_EQ(ef.select(2), std::make_optional<size_t>(3));
    EXPECT_EQ(ef.select(3), std::make_optional<size_t>(7));
    EXPECT_EQ(ef.select(4), std::nullopt);
}

// Test an empty EliasFano
TEST(EliasFanoTest, EmptyEliasFano)
{
    std::vector<size_t> v;
    EliasFano ef = EliasFano::from(v);

    EXPECT_EQ(ef.len(), 0);
    EXPECT_EQ(ef.get_universe(), 0);
    EXPECT_TRUE(ef.is_empty());
}

// Test EliasFanoBuilder
TEST(EliasFanoBuilderTest, BasicBuilder)
{
    EliasFanoBuilder efb(8, 4);

    efb.push(1);
    efb.push(3);
    efb.push(3);
    efb.push(7);

    EliasFano ef = efb.build();

    EXPECT_EQ(ef.len(), 4);
    EXPECT_EQ(ef.get_universe(), 8);

    EXPECT_EQ(ef.select(0), std::make_optional<size_t>(1));
    EXPECT_EQ(ef.select(1), std::make_optional<size_t>(3));
    EXPECT_EQ(ef.select(2), std::make_optional<size_t>(3));
    EXPECT_EQ(ef.select(3), std::make_optional<size_t>(7));
}

// Test extend operation of EliasFanoBuilder
TEST(EliasFanoBuilderTest, ExtendBuilder)
{
    EliasFanoBuilder efb(16, 6);

    std::vector<size_t> v1 = {1, 3, 3};
    efb.extend(v1);

    std::vector<size_t> v2 = {7, 10, 15};
    efb.extend(v2);

    EliasFano ef = efb.build();

    EXPECT_EQ(ef.len(), 6);
    EXPECT_EQ(ef.get_universe(), 16);

    EXPECT_EQ(ef.select(0), std::make_optional<size_t>(1));
    EXPECT_EQ(ef.select(1), std::make_optional<size_t>(3));
    EXPECT_EQ(ef.select(2), std::make_optional<size_t>(3));
    EXPECT_EQ(ef.select(3), std::make_optional<size_t>(7));
    EXPECT_EQ(ef.select(4), std::make_optional<size_t>(10));
    EXPECT_EQ(ef.select(5), std::make_optional<size_t>(15));
}

// Test a larger sequence
TEST(EliasFanoTest, LargerSequence)
{
    std::vector<size_t> v;
    for (size_t i = 0; i < 100; i++)
    {
        v.push_back(i * 10);
    }

    EliasFano ef = EliasFano::from(v);

    EXPECT_EQ(ef.len(), 100);
    EXPECT_EQ(ef.get_universe(), 991);

    for (size_t i = 0; i < 100; i++)
    {
        EXPECT_EQ(ef.select(i), std::make_optional<size_t>(i * 10));
    }
    EXPECT_EQ(ef.select(100), std::nullopt);
}

// Test bit vector operations
TEST(BitVectorTest, BitOperations)
{
    BitVectorMut bvm;
    bvm.append_bits(0x5, 3); // 101
    bvm.append_bits(0x7, 3); // 111

    BitVector bv = bvm.into();

    EXPECT_EQ(bv.size(), 6);
    EXPECT_TRUE(bv.get(0));
    EXPECT_FALSE(bv.get(1));
    EXPECT_TRUE(bv.get(2));
    EXPECT_TRUE(bv.get(3));
    EXPECT_TRUE(bv.get(4));
    EXPECT_TRUE(bv.get(5));

    EXPECT_EQ(bv.get_bits_unchecked(0, 3), 0x5); // 101
    EXPECT_EQ(bv.get_bits_unchecked(3, 3), 0x7); // 111
    EXPECT_EQ(bv.get_bits_unchecked(1, 4), 0xe); // 1110
}

// Death tests (these will cause assertions when run with NDEBUG undefined)
#ifndef NDEBUG
TEST(EliasFanoDeathTest, OutOfOrderSequence)
{
    std::vector<size_t> v = {3, 2, 1}; // Not monotonically increasing
    EXPECT_DEATH(EliasFano::from(v), "");
}

TEST(EliasFanoDeathTest, ZeroNumVals)
{
    EXPECT_DEATH(EliasFanoBuilder(10, 0), "");
}

TEST(EliasFanoDeathTest, PushTooManyValues)
{
    EliasFanoBuilder efb(10, 2);
    efb.push(1);
    efb.push(2);
    EXPECT_DEATH(efb.push(3), "");
}

TEST(EliasFanoDeathTest, PushValueOutOfUniverse)
{
    EliasFanoBuilder efb(10, 2);
    efb.push(1);
    EXPECT_DEATH(efb.push(10), ""); // 10 is out of universe (0-9)
}

TEST(EliasFanoDeathTest, PushValueLessThanPrevious)
{
    EliasFanoBuilder efb(10, 2);
    efb.push(5);
    EXPECT_DEATH(efb.push(4), ""); // 4 < 5, not monotonically increasing
}
#endif // NDEBUG