#!/usr/bin/env python3
"""
Example usage of the seismic_cpp Python bindings.
"""

import numpy as np
import json
import tempfile
import os
from seismic_cpp import SeismicIndex, SeismicIndexRaw, get_seismic_string

def create_sample_data():
    """Create sample data for testing."""
    data = [
        {"id": "doc1", "tokens": ["word1", "word2", "word3"], "values": [1.0, 2.0, 3.0]},
        {"id": "doc2", "tokens": ["word2", "word4"], "values": [4.0, 5.0]},
        {"id": "doc3", "tokens": ["word1", "word2", "word3", "word4"], "values": [1.0, 2.0, 3.0, 4.0]}
    ]
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
        temp_path = f.name
    
    return temp_path

def example_seismic_index():
    """Example usage of SeismicIndex."""
    print("\n=== SeismicIndex Example ===")
    
    # Create sample data
    input_path = create_sample_data()
    print(f"Created sample data at: {input_path}")
    
    # Build the index
    index = SeismicIndex.build(
        input_path=input_path,
        n_postings=10,
        centroid_fraction=0.1,
        min_cluster_size=2,
        summary_energy=0.4
    )
    
    print(f"Index dimensions: {index.dim}")
    print(f"Index length: {index.len}")
    print(f"Index nnz: {index.nnz}")
    print(f"Index KNN length: {index.knn_len}")
    
    # Search the index
    query_id = "query1"
    query_tokens = ["word1", "word2"]
    query_values = [1.0, 2.0]
    
    results = index.search(
        query_id=query_id,
        query_components=query_tokens,
        query_values=query_values,
        k=2,
        query_cut=2,
        heap_factor=0.7,
        n_knn=0,
        sorted=False
    )
    
    print("\nSearch results:")
    for query_id, score, doc_id in results:
        print(f"  Query: {query_id}, Score: {score}, Document: {doc_id}")
    
    # Batch search
    query_ids = ["query1", "query2"]
    query_components = [["word1", "word2"], ["word2", "word4"]]
    query_values = [[1.0, 2.0], [1.0, 2.0]]
    
    batch_results = index.batch_search(
        queries_ids=query_ids,
        query_components=query_components,
        query_values=query_values,
        k=2,
        query_cut=2,
        heap_factor=0.7,
        n_knn=0,
        sorted=False
    )
    
    print("\nBatch search results:")
    for i, results in enumerate(batch_results):
        print(f"Query {i+1}:")
        for query_id, score, doc_id in results:
            print(f"  Query: {query_id}, Score: {score}, Document: {doc_id}")
    
    # Build KNN graph
    print("\nBuilding KNN graph...")
    index.build_knn(2)
    print(f"KNN length after building: {index.knn_len}")
    
    # Save the index
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_path = f.name
    
    print(f"\nSaving index to: {save_path}")
    index.save(save_path)
    
    # Save KNN graph
    knn_path = save_path + ".knn"
    print(f"Saving KNN graph to: {knn_path}")
    index.save_knn(knn_path)
    
    # Clean up
    os.unlink(input_path)
    os.unlink(save_path + ".index.seismic")
    os.unlink(knn_path)
    
    print("\nExample completed successfully!")

def example_seismic_index_raw():
    """Example usage of SeismicIndexRaw."""
    print("\n=== SeismicIndexRaw Example ===")
    
    # This would typically use a binary file, but for demonstration, we'll skip this
    print("SeismicIndexRaw requires binary input files, skipping example.")
    
    # In a real scenario, you would use:
    # index = SeismicIndexRaw.build(
    #     input_file="path/to/binary/file",
    #     n_postings=10,
    #     centroid_fraction=0.1,
    #     min_cluster_size=2,
    #     summary_energy=0.4
    # )
    
    # And search with:
    # results = index.search(
    #     query_components=np.array([0, 1], dtype=np.int32),
    #     query_values=np.array([1.0, 2.0], dtype=np.float32),
    #     k=2,
    #     query_cut=2,
    #     heap_factor=0.7,
    #     n_knn=0,
    #     sorted=False
    # )

if __name__ == "__main__":
    print(f"Seismic string: {get_seismic_string()}")
    example_seismic_index()
    example_seismic_index_raw()
