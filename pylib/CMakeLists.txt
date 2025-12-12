cmake_minimum_required(VERSION 3.14)
project(seismic_cpp_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Define source files for the main library
file(GLOB_RECURSE MAIN_SOURCES 
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.h
)
# Remove binary files from sources
list(FILTER MAIN_SOURCES EXCLUDE REGEX ".*/bin/.*")

# EXCLUDE sparse_jlt.cpp because it is included via headers (ODR issue)
list(FILTER MAIN_SOURCES EXCLUDE REGEX ".*/sparse_jlt\\.cpp$")

# Add our implementation files explicitly
list(APPEND MAIN_SOURCES 
    ${CMAKE_SOURCE_DIR}/src/configuration.cpp
    ${CMAKE_SOURCE_DIR}/src/inverted_index_space_usage.cpp
    ${CMAKE_SOURCE_DIR}/src/knn_space_usage.cpp
)

# Add the Python module
pybind11_add_module(seismic_cpp seismic_bindings.cpp ${MAIN_SOURCES})

# Include directories
target_include_directories(seismic_cpp PRIVATE 
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CEREAL_INCLUDE_DIR}
)

# Link with nlohmann_json and cereal
target_link_libraries(seismic_cpp PRIVATE 
    nlohmann_json::nlohmann_json
    cereal
    OpenMP::OpenMP_CXX
)

# Install the module
install(TARGETS seismic_cpp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
