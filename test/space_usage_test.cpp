#include <gtest/gtest.h>
#include "../src/space_usage.h"
#include <vector>
#include <memory>

using namespace seismic;

// Test for primitive type space usage
TEST(SpaceUsageTest, PrimitiveTypes) {
    Bool boolUsage;
    Int8 int8Usage;
    UInt8 uint8Usage;
    Int16 int16Usage;
    UInt16 uint16Usage;
    Int32 int32Usage;
    UInt32 uint32Usage;
    Int64 int64Usage;
    UInt64 uint64Usage;
    SizeT sizeUsage;
    Float32 float32Usage;
    Float64 float64Usage;
    Float16 float16Usage;

    EXPECT_EQ(boolUsage.space_usage_byte(), sizeof(bool));
    EXPECT_EQ(int8Usage.space_usage_byte(), sizeof(int8_t));
    EXPECT_EQ(uint8Usage.space_usage_byte(), sizeof(uint8_t));
    EXPECT_EQ(int16Usage.space_usage_byte(), sizeof(int16_t));
    EXPECT_EQ(uint16Usage.space_usage_byte(), sizeof(uint16_t));
    EXPECT_EQ(int32Usage.space_usage_byte(), sizeof(int32_t));
    EXPECT_EQ(uint32Usage.space_usage_byte(), sizeof(uint32_t));
    EXPECT_EQ(int64Usage.space_usage_byte(), sizeof(int64_t));
    EXPECT_EQ(uint64Usage.space_usage_byte(), sizeof(uint64_t));
    EXPECT_EQ(sizeUsage.space_usage_byte(), sizeof(std::size_t));
    EXPECT_EQ(float32Usage.space_usage_byte(), sizeof(float));
    EXPECT_EQ(float64Usage.space_usage_byte(), sizeof(double));
    EXPECT_EQ(float16Usage.space_usage_byte(), sizeof(Float16));
}

// Test for vector space usage
TEST(SpaceUsageTest, VectorSpaceUsage) {
    std::vector<int> vec(10);
    VectorSpaceUsage<int> vecUsage(vec);
    
    // Expected size: sizeof(vector) + sizeof(int) * capacity
    std::size_t expected = sizeof(std::vector<int>) + sizeof(int) * vec.capacity();
    EXPECT_EQ(vecUsage.space_usage_byte(), expected);
    
    // Test with a larger vector
    std::vector<double> largeVec(100);
    VectorSpaceUsage<double> largeVecUsage(largeVec);
    expected = sizeof(std::vector<double>) + sizeof(double) * largeVec.capacity();
    EXPECT_EQ(largeVecUsage.space_usage_byte(), expected);
}

// Test for array space usage
TEST(SpaceUsageTest, ArraySpaceUsage) {
    // Create an array of integers
    auto intArray = std::make_unique<int[]>(10);
    ArraySpaceUsage<int> intArrayUsage(intArray.get(), 10);
    
    // Expected size: sizeof(unique_ptr) + sizeof(int) * length
    std::size_t expected = sizeof(std::unique_ptr<int[]>) + sizeof(int) * 10;
    EXPECT_EQ(intArrayUsage.space_usage_byte(), expected);
    
    // Test with an empty array
    auto emptyArray = std::make_unique<float[]>(0);
    ArraySpaceUsage<float> emptyArrayUsage(emptyArray.get(), 0);
    EXPECT_EQ(emptyArrayUsage.space_usage_byte(), sizeof(std::unique_ptr<float[]>));
}

// Test for unit conversions
TEST(SpaceUsageTest, UnitConversions) {
    // Create a test object with a known byte size
    class TestSpaceUsage : public SpaceUsage {
    public:
        std::size_t space_usage_byte() const override {
            return 1024 * 1024; // 1 MiB
        }
    };
    
    TestSpaceUsage testUsage;
    EXPECT_EQ(testUsage.space_usage_byte(), 1024 * 1024);
    EXPECT_DOUBLE_EQ(testUsage.space_usage_KiB(), 1024.0);
    EXPECT_DOUBLE_EQ(testUsage.space_usage_MiB(), 1.0);
    EXPECT_DOUBLE_EQ(testUsage.space_usage_GiB(), 1.0 / 1024.0);
    
    // Test with a larger size
    class LargeTestSpaceUsage : public SpaceUsage {
    public:
        std::size_t space_usage_byte() const override {
            return 1024 * 1024 * 1024; // 1 GiB
        }
    };
    
    LargeTestSpaceUsage largeTestUsage;
    EXPECT_EQ(largeTestUsage.space_usage_byte(), 1024 * 1024 * 1024);
    EXPECT_DOUBLE_EQ(largeTestUsage.space_usage_KiB(), 1024.0 * 1024.0);
    EXPECT_DOUBLE_EQ(largeTestUsage.space_usage_MiB(), 1024.0);
    EXPECT_DOUBLE_EQ(largeTestUsage.space_usage_GiB(), 1.0);
}
