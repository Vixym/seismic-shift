cmake_minimum_required(VERSION 3.14)
project(seismic-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g")

# Define source files
file(GLOB_RECURSE SOURCES src/*.cpp)

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Fetch GoogleTest for unit testing
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(INSTALL_GTEST OFF)
FetchContent_MakeAvailable(googletest)

# Define test executable
file(GLOB_RECURSE TEST_SOURCES test/*.cpp)
if(TEST_SOURCES)
    add_executable(tests ${TEST_SOURCES})
    target_link_libraries(tests gtest gtest_main)
    enable_testing()
    add_test(NAME AllTests COMMAND tests)
endif()
