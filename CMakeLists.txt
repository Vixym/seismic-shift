cmake_minimum_required(VERSION 3.14)
project(seismic-shift LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g")

# Fetch dependencies
include(FetchContent)

# Fetch nlohmann/json for JSON parsing
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Define source files
file(GLOB_RECURSE SOURCES src/*.cpp)

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann_json::nlohmann_json)

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
)

add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests gtest gtest_main nlohmann_json::nlohmann_json)
enable_testing()
add_test(NAME AllTests COMMAND tests)
