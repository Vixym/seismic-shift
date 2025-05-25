cmake_minimum_required(VERSION 3.5)
project(seismic-shift LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g")

# Find OpenMP
if(APPLE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Fetch dependencies
include(FetchContent)

# Fetch nlohmann/json for JSON parsing
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# Fetch cereal for serialization
FetchContent_Declare(
    cereal
    GIT_REPOSITORY https://github.com/USCiLab/cereal.git
    GIT_TAG v1.3.2
)
FetchContent_MakeAvailable(cereal)

# Create a library target for the core functionality
add_library(seismic_core STATIC
    src/configuration.cpp
    src/inverted_index_space_usage.cpp
    src/knn_space_usage.cpp
)
target_include_directories(seismic_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${cereal_SOURCE_DIR}/include)
target_link_libraries(seismic_core PRIVATE nlohmann_json::nlohmann_json cereal OpenMP::OpenMP_CXX)

# Create the build_inverted_index executable
add_executable(build_inverted_index src/bin/build_inverted_index.cpp)
target_link_libraries(build_inverted_index PRIVATE seismic_core nlohmann_json::nlohmann_json cereal OpenMP::OpenMP_CXX)
target_include_directories(build_inverted_index PRIVATE ${cereal_SOURCE_DIR}/include)

# Create the perf_inverted_index executable
add_executable(perf_inverted_index src/bin/perf_inverted_index.cpp)
target_link_libraries(perf_inverted_index PRIVATE seismic_core nlohmann_json::nlohmann_json cereal OpenMP::OpenMP_CXX)
target_include_directories(perf_inverted_index PRIVATE ${cereal_SOURCE_DIR}/include)

# Set output directory for binaries
set_target_properties(build_inverted_index perf_inverted_index PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Fetch GoogleTest for unit testing
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(INSTALL_GTEST OFF)
FetchContent_MakeAvailable(googletest)

# Define test executable
set(TEST_SOURCES
    test/distances_test.cpp
    test/elias_fano_test.cpp
    test/json_utils_test.cpp
    test/main_test.cpp
    test/utils_test.cpp
    test/top_k_selector_test.cpp
    test/space_usage_test.cpp
    test/sparse_dataset_test.cpp
    test/quantized_summary_test.cpp
    test/inverted_index_test.cpp
    test/seismic_index_test.cpp
    # Add our implementation files
    src/configuration.cpp
    src/inverted_index_space_usage.cpp
    src/knn_space_usage.cpp
)

add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests PRIVATE 
    GTest::gtest 
    GTest::gtest_main 
    nlohmann_json::nlohmann_json 
    cereal 
    OpenMP::OpenMP_CXX
)
target_include_directories(tests PRIVATE ${cereal_SOURCE_DIR}/include)
enable_testing()
add_test(NAME AllTests COMMAND tests)

# Add Python bindings subdirectory
add_subdirectory(pylib)