# Seismic C++ Python Bindings

This directory contains Python bindings for the Seismic C++ library, allowing you to use the powerful Seismic indexing and search capabilities from Python.

## Installation

### Prerequisites

- C++20 compatible compiler (GCC 10+, Clang 10+, or MSVC 19.29+)
- Python 3.7 or higher
- pybind11 (installed automatically during setup)
- CMake 3.14 or higher (for CMake build)

### Option 1: Install using pip

```bash
cd pylib
pip install -e .
```

This will build the extension module and install it in development mode.

### Option 2: Build using CMake

First, make sure the main Seismic C++ library is built:

```bash
mkdir -p build && cd build
cmake ..
make
```

Then build the Python bindings:

```bash
cd ../pylib
mkdir -p build && cd build
cmake ..
make
```

## Usage

### Basic Example

```python
from seismic_cpp import SeismicIndex, get_seismic_string

# Build an index from a JSONL file
index = SeismicIndex.build(
    input_path="data.jsonl",
    n_postings=3500,
    centroid_fraction=0.1,
    min_cluster_size=2,
    summary_energy=0.4
)

# Search the index
results = index.search(
    query_id="query1",
    query_components=["word1", "word2"],
    query_values=[1.0, 2.0],
    k=10,
    query_cut=10,
    heap_factor=0.7,
    n_knn=0,
    sorted=False
)

# Print results
for query_id, score, doc_id in results:
    print(f"Query: {query_id}, Score: {score}, Document: {doc_id}")

# Save the index
index.save("my_index")

# Load an existing index
loaded_index = SeismicIndex.load("my_index.index.seismic")
```

### Building a KNN Graph

```python
# Build a KNN graph with 10 neighbors per vector
index.build_knn(nknn=10)

# Save the KNN graph
index.save_knn("my_index.knn")

# Load a KNN graph
index.load_knn("my_index.knn")

# Search with KNN refinement
results = index.search(
    query_id="query1",
    query_components=["word1", "word2"],
    query_values=[1.0, 2.0],
    k=10,
    query_cut=10,
    heap_factor=0.7,
    n_knn=5,  # Use 5 KNN neighbors for refinement
    sorted=False
)
```

### Batch Search

```python
# Batch search
query_ids = ["query1", "query2"]
query_components = [["word1", "word2"], ["word2", "word4"]]
query_values = [[1.0, 2.0], [1.0, 2.0]]

batch_results = index.batch_search(
    queries_ids=query_ids,
    query_components=query_components,
    query_values=query_values,
    k=10,
    query_cut=10,
    heap_factor=0.7,
    n_knn=0,
    sorted=False,
    num_threads=4  # Use 4 threads for parallel processing
)

# Process results
for i, results in enumerate(batch_results):
    print(f"Results for query {query_ids[i]}:")
    for query_id, score, doc_id in results:
        print(f"  Score: {score}, Document: {doc_id}")
```

### Raw Index (without document mapping)

```python
from seismic_cpp import SeismicIndexRaw

# Build an index from a binary file
raw_index = SeismicIndexRaw.build(
    input_file="vectors.bin",
    n_postings=3500,
    centroid_fraction=0.1,
    min_cluster_size=2,
    summary_energy=0.4
)

# Search with component IDs
import numpy as np
results = raw_index.search(
    query_components=np.array([0, 1, 2], dtype=np.int32),
    query_values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
    k=10,
    query_cut=10,
    heap_factor=0.7,
    n_knn=0,
    sorted=False
)

# Print results
for score, doc_id in results:
    print(f"Score: {score}, Document ID: {doc_id}")
```

## API Reference

### `SeismicIndex`

- `dim` - Get the dimensionality of the index
- `len` - Get the number of vectors in the index
- `nnz` - Get the number of non-zero components in the index
- `knn_len` - Get the number of neighbors in the KNN graph
- `print_space_usage_byte()` - Print the space usage of the index
- `get(id)` - Get the components and values of a vector
- `vector_len(id)` - Get the length of a vector
- `save(path)` - Save the index to a file
- `load(path)` - Load an index from a file
- `build_knn(nknn)` - Build a KNN graph
- `save_knn(path)` - Save the KNN graph to a file
- `load_knn(path, nknn=None)` - Load a KNN graph from a file
- `search(query_id, query_components, query_values, k, query_cut, heap_factor, n_knn, sorted)` - Search the index
- `batch_search(queries_ids, query_components, query_values, k, query_cut, heap_factor, n_knn, sorted, num_threads=0)` - Batch search the index
- `build(input_path, n_postings=3500, centroid_fraction=0.1, min_cluster_size=2, summary_energy=0.4, nknn=0, knn_path=None, batched_indexing=None, input_token_to_id_map=None, num_threads=0)` - Build an index from a JSONL file

### `SeismicIndexRaw`

- `dim` - Get the dimensionality of the index
- `len` - Get the number of vectors in the index
- `nnz` - Get the number of non-zero components in the index
- `knn_len` - Get the number of neighbors in the KNN graph
- `is_empty` - Check if the index is empty
- `print_space_usage_byte()` - Print the space usage of the index
- `get(id)` - Get the components and values of a vector
- `vector_len(id)` - Get the length of a vector
- `save(path)` - Save the index to a file
- `load(path)` - Load an index from a file
- `build_knn(nknn)` - Build a KNN graph
- `save_knn(path)` - Save the KNN graph to a file
- `load_knn(path, nknn=None)` - Load a KNN graph from a file
- `search(query_components, query_values, k, query_cut, heap_factor, n_knn, sorted)` - Search the index
- `batch_search(query_path, k, query_cut, heap_factor, n_knn, sorted, num_threads=0)` - Batch search the index with queries from a binary file
- `build(input_file, n_postings=3500, centroid_fraction=0.1, min_cluster_size=2, summary_energy=0.4, nknn=0, knn_path=None, batched_indexing=None)` - Build an index from a binary file

### Utility Functions

- `get_seismic_string()` - Get the seismic string identifier
