# SeismicShift

Building a C++ version of Seismic (https://github.com/TusKANNy/seismic) with support for dynamic updates + potential other optimizations.

## Testing out the translated end-to-end pipeline (original Seismic version)

### Prepare a dataset

Download this `seismic-msmarco-splade-bin`, provided by the Seismic authors:

https://huggingface.co/datasets/tuskanny/seismic-msmarco-splade-bin/tree/main

Once unpacked, place its contents into a separate directory named `/sparse_datasets`. The unpacked contents should already look like this:

```
sparse_datasets
└── msmarco_v1_passage
    └── cocondenser
        ├── data
            ├── doc_ids.npy
            ├── documents.bin
            ├── ...
        ├── indexes (initially empty)
    └── qrels.dev.small.tsv
```

Edit `seismic-shift/experiments/sigir2024/splade.toml` if needed for SeismicShift to locate this `/sparse_datsets` directory. Specifically, edit this section:

```
...
[folder] 
data =          "[path to sparse_datasets]/sparse_datasets/msmarco_v1_passage/cocondenser/data"
index =         "[path to sparse_datasets]/sparse_datasets/msmarco_v1_passage/cocondenser/indexes"
qrels_path =    "[path to sparse_datasets]/sparse_datasets/msmarco_v1_passage/qrels.dev.small.tsv"
experiment =    "."     # stdout and stderr here of running the experiment is saved here. in a specific subfolder for the current execution
...
```

### Run the helper script

From the root directory, run:

```
// If freshly cloned copy...
cmake -B build -S .

// Build...
cd build
make clean; make

// If testing runtime performance, you can explicitly restrict the number of permitted threads by:
export OMP_NUM_THREADS=[number of threads]
echo $OMP_NUM_THREADS // Should be [number of threads]

// Run the main wrapper script
python3 script/run_experiments.py --exp experiments/sigir2024/splade.toml
```

At a high level, `run_experiments.py` is a wrapper that:
1. First builds the InvertedIndex data structure from the provided dataset using `seismic-shift/src/bin/build_inverted_index.cpp`.
2. Serializes the InvertedIndex into an intermediate format, `[path].index.seismic`. This will be stored in `sparse_datasets/msmarco_v1_passage/cocondenser/indexes`.
3. Executes a given dataset's queries on this InvertedIndex by deserializing from this intermediate format and running `seismic-shift/src/bin/perf_inverted_index.cpp`.

### Example results (for `experiments/sigir2024/splade.toml`):

For the portion that runs `perf_inverted_index.cpp` (step #3), the observed output is pasted below. Note that the `Metric of the run: RR@10: ...` and `Metric of the gt : RR@10: ...` (precalculated groundtruth) are observably close, which provides a good sanity check for the accuracy of our C++ Seismic translation.

```
...

Git info
Current Branch: main
Commit ID: 2e6b42f219c4337f1539b728edc44433967673e8

Compiling the code
Compiling code with cmake --build build --config
[  7%] Built target seismic_core
[ 11%] Built target build_inverted_index
[ 15%] Built target perf_inverted_index
[ 19%] Built target gtest
[ 23%] Built target gtest_main
[ 52%] Built target tests
[ 56%] Built target sandbox
[ 60%] Built target sandbox_json
[ 64%] Built target sandbox_rtti
[ 70%] Built target sandbox_vs_dll
[ 74%] Built target sandbox_vs
[ 78%] Built target performance
[ 82%] Built target gmock
[ 86%] Built target gmock_main
[100%] Built target seismic_cpp
Code compiled successfully.
Skipping index building step entirely.
Evaluation runs with metric RR@10
Executing query for subsection 'recall_90' with command:
 ./build/bin/perf_inverted_index --index-file /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4 -k 10 --query-file /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin --query-cut 3 --heap-factor 0.9 --n-runs 1 --output-path ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_90
Running query for subsection: recall_90...
Performance testing with the following parameters:
  Index file: /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4
  Query file: /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin
  Output path: ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_90
  N queries: 10000
  K: 10
  N runs: 1
  Query cut: 3
  Heap factor: 0.9
  N KNN: 0
  First sorted: false
Loading index from /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4.index.seismic...
Index loaded successfully
Number of documents: 8841823

Loading queries from /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin...
Number of Queries: 6980
Number of Dimensions: 28313
Avg number of components: 43.9454
Searching for top-10 results
Number of evaluated queries: 6980
Number of documents: 8841823
Avg number of non-zero components: 121.046
Time 5008 microsecs per query
5008
Total space usage: 11306567207 bytes
Writing results to ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_90...
Results written successfully
Query for subsection 'recall_90' executed successfully.
Metric of the run: RR@10: 0.36741739436939974
Metric of the gt : RR@10: 0.38280017737754163
Executing query for subsection 'recall_91' with command:
 ./build/bin/perf_inverted_index --index-file /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4 -k 10 --query-file /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin --query-cut 4 --heap-factor 0.9 --n-runs 1 --output-path ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_91
Running query for subsection: recall_91...
Performance testing with the following parameters:
  Index file: /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4
  Query file: /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin
  Output path: ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_91
  N queries: 10000
  K: 10
  N runs: 1
  Query cut: 4
  Heap factor: 0.9
  N KNN: 0
  First sorted: false
Loading index from /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4.index.seismic...
Index loaded successfully
Number of documents: 8841823

Loading queries from /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin...
Number of Queries: 6980
Number of Dimensions: 28313
Avg number of components: 43.9454
Searching for top-10 results
Number of evaluated queries: 6980
Number of documents: 8841823
Avg number of non-zero components: 121.046
Time 5813 microsecs per query
5813
Total space usage: 11306567207 bytes
Writing results to ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_91...
Results written successfully
Query for subsection 'recall_91' executed successfully.
Metric of the run: RR@10: 0.3700439464228858
Metric of the gt : RR@10: 0.38280017737754163
Executing query for subsection 'recall_92' with command:
 ./build/bin/perf_inverted_index --index-file /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4 -k 10 --query-file /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin --query-cut 4 --heap-factor 0.9 --n-runs 1 --output-path ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_92
Running query for subsection: recall_92...
Performance testing with the following parameters:
  Index file: /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4
  Query file: /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin
  Output path: ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_92
  N queries: 10000
  K: 10
  N runs: 1
  Query cut: 4
  Heap factor: 0.9
  N KNN: 0
  First sorted: false
Loading index from /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4.index.seismic...
Index loaded successfully
Number of documents: 8841823

Loading queries from /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin...
Number of Queries: 6980
Number of Dimensions: 28313
Avg number of components: 43.9454
Searching for top-10 results
Number of evaluated queries: 6980
Number of documents: 8841823
Avg number of non-zero components: 121.046
Time 5904 microsecs per query
5904
Total space usage: 11306567207 bytes
Writing results to ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_92...
Results written successfully
Query for subsection 'recall_92' executed successfully.
Metric of the run: RR@10: 0.3700439464228858
Metric of the gt : RR@10: 0.38280017737754163
Executing query for subsection 'recall_93' with command:
 ./build/bin/perf_inverted_index --index-file /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4 -k 10 --query-file /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin --query-cut 5 --heap-factor 0.9 --n-runs 1 --output-path ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_93
Running query for subsection: recall_93...
Performance testing with the following parameters:
  Index file: /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4
  Query file: /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin
  Output path: ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_93
  N queries: 10000
  K: 10
  N runs: 1
  Query cut: 5
  Heap factor: 0.9
  N KNN: 0
  First sorted: false
Loading index from /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4.index.seismic...
Index loaded successfully
Number of documents: 8841823

Loading queries from /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin...
Number of Queries: 6980
Number of Dimensions: 28313
Avg number of components: 43.9454
Searching for top-10 results
Number of evaluated queries: 6980
Number of documents: 8841823
Avg number of non-zero components: 121.046
Time 6549 microsecs per query
6549
Total space usage: 11306567207 bytes
Writing results to ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_93...
Results written successfully
Query for subsection 'recall_93' executed successfully.
Metric of the run: RR@10: 0.37107552189930354
Metric of the gt : RR@10: 0.38280017737754163
Executing query for subsection 'recall_94' with command:
 ./build/bin/perf_inverted_index --index-file /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4 -k 10 --query-file /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin --query-cut 8 --heap-factor 0.9 --n-runs 1 --output-path ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_94
Running query for subsection: recall_94...
Performance testing with the following parameters:
  Index file: /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4
  Query file: /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin
  Output path: ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_94
  N queries: 10000
  K: 10
  N runs: 1
  Query cut: 8
  Heap factor: 0.9
  N KNN: 0
  First sorted: false
Loading index from /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4.index.seismic...
Index loaded successfully
Number of documents: 8841823

Loading queries from /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin...
Number of Queries: 6980
Number of Dimensions: 28313
Avg number of components: 43.9454
Searching for top-10 results
Number of evaluated queries: 6980
Number of documents: 8841823
Avg number of non-zero components: 121.046
Time 8477 microsecs per query
8477
Total space usage: 11306567207 bytes
Writing results to ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_94...
Results written successfully
Query for subsection 'recall_94' executed successfully.
Metric of the run: RR@10: 0.37399235912129836
Metric of the gt : RR@10: 0.38280017737754163
Executing query for subsection 'recall_95' with command:
 ./build/bin/perf_inverted_index --index-file /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4 -k 10 --query-file /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin --query-cut 5 --heap-factor 0.8 --n-runs 1 --output-path ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_95
Running query for subsection: recall_95...
Performance testing with the following parameters:
  Index file: /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4
  Query file: /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin
  Output path: ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_95
  N queries: 10000
  K: 10
  N runs: 1
  Query cut: 5
  Heap factor: 0.8
  N KNN: 0
  First sorted: false
Loading index from /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4.index.seismic...
Index loaded successfully
Number of documents: 8841823

Loading queries from /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin...
Number of Queries: 6980
Number of Dimensions: 28313
Avg number of components: 43.9454
Searching for top-10 results
Number of evaluated queries: 6980
Number of documents: 8841823
Avg number of non-zero components: 121.046
Time 9377 microsecs per query
9377
Total space usage: 11306567207 bytes
Writing results to ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_95...
Results written successfully
Query for subsection 'recall_95' executed successfully.
Metric of the run: RR@10: 0.3768639991813341
Metric of the gt : RR@10: 0.38280017737754163
Executing query for subsection 'recall_96' with command:
 ./build/bin/perf_inverted_index --index-file /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4 -k 10 --query-file /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin --query-cut 6 --heap-factor 0.8 --n-runs 1 --output-path ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_96
Running query for subsection: recall_96...
Performance testing with the following parameters:
  Index file: /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4
  Query file: /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin
  Output path: ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_96
  N queries: 10000
  K: 10
  N runs: 1
  Query cut: 6
  Heap factor: 0.8
  N KNN: 0
  First sorted: false
Loading index from /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4.index.seismic...
Index loaded successfully
Number of documents: 8841823

Loading queries from /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin...
Number of Queries: 6980
Number of Dimensions: 28313
Avg number of components: 43.9454
Searching for top-10 results
Number of evaluated queries: 6980
Number of documents: 8841823
Avg number of non-zero components: 121.046
Time 10188 microsecs per query
10188
Total space usage: 11306567207 bytes
Writing results to ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_96...
Results written successfully
Query for subsection 'recall_96' executed successfully.
Metric of the run: RR@10: 0.37785731341247053
Metric of the gt : RR@10: 0.38280017737754163
Executing query for subsection 'recall_97' with command:
 ./build/bin/perf_inverted_index --index-file /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4 -k 10 --query-file /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin --query-cut 6 --heap-factor 0.7 --n-runs 1 --output-path ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_97
Running query for subsection: recall_97...
Performance testing with the following parameters:
  Index file: /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4
  Query file: /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin
  Output path: ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_97
  N queries: 10000
  K: 10
  N runs: 1
  Query cut: 6
  Heap factor: 0.7
  N KNN: 0
  First sorted: false
Loading index from /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/sigir_2024_splade_cocondenser_centroid-fraction_0.1_clustering-algorithm_random-kmeans_kmeans-doc-cut_15_kmeans-pruning-factor_0.005_knn_0_n-postings_4000_summary-energy_0.4.index.seismic...
Index loaded successfully
Number of documents: 8841823

Loading queries from /storage/vxma/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin...
Number of Queries: 6980
Number of Dimensions: 28313
Avg number of components: 43.9454
Searching for top-10 results
Number of evaluated queries: 6980
Number of documents: 8841823
Avg number of non-zero components: 121.046
Time 14716 microsecs per query
14716
Total space usage: 11306567207 bytes
Writing results to ./splade_cocondenser_msmarco_2025-05-25_18:54:30/results_recall_97...
Results written successfully
Query for subsection 'recall_97' executed successfully.
Metric of the run: RR@10: 0.379833765406831
Metric of the gt : RR@10: 0.38280017737754163
```

## Testing out a toy example

From the root directory, run:

```
// If fresh copy...
cmake -B build -S .

// To build...
cd build
make clean; make

// To run the example...
cd ..
python3 pylib/example.py
```

This toy example runs Seismic on a miniature dataset of 3 document vectors and 2 queries. The expected output is:

```
vxma@dhcp-10-29-193-115 seismic-cpp % python3 pylib/example.py
Seismic string: S30

=== SeismicIndex Example ===
Created sample data at: /var/folders/7q/d0q9fdbj5cv076r6_p5xf_jm0000gn/T/tmpxrm79xyc

Building the index...
Configuration: 
  Pruning: Global threshold with 10 postings
  Blocking: Random kmeans with centroid fraction 0.1
  Summarization: Energy preserving with 0.4 energy
Distributing and pruning postings: 0 secs
        Number of posting lists: 4
Building summaries: 0 secs
Index dimensions: 4
Index length: 3
Index nnz: 9
Index KNN length: 0

Search results:
  Query: query1, Score: 5.0, Document: doc1
  Query: query1, Score: 8.0, Document: doc2

Batch search results:
Query 1:
  Query: query1, Score: 5.0, Document: doc1
  Query: query1, Score: 8.0, Document: doc2
Query 2:
  Query: query2, Score: 10.0, Document: doc3
  Query: query2, Score: 14.0, Document: doc2

Building KNN graph...
Building KNN graph with 2 neighbors
KNN graph built successfully
KNN length after building: 0

Saving index to: /var/folders/7q/d0q9fdbj5cv076r6_p5xf_jm0000gn/T/tmp0li3arff
Saving ... /var/folders/7q/d0q9fdbj5cv076r6_p5xf_jm0000gn/T/tmp0li3arff.index.seismic
Save completed successfully
Saving KNN graph to: /var/folders/7q/d0q9fdbj5cv076r6_p5xf_jm0000gn/T/tmp0li3arff.knn
Saving KNN graph to /var/folders/7q/d0q9fdbj5cv076r6_p5xf_jm0000gn/T/tmp0li3arff.knn
KNN graph saved successfully

Example completed successfully!

=== SeismicIndexRaw Example ===
SeismicIndexRaw requires binary input files, skipping example.
```

## Testing out SeismicShift's sparse JLT implementation

From the root directory, run:

```
// If fresh copy...
cmake -B build -S .

// To build...
cd build
make clean; make
./seismic-shift -n [num vectors] -d [original dimension] -k [target dimension]
```

Note that more CLI options are displayed when running `./seismic-shift -help`.

As an example, serially, it takes ~2 minutes to run:

```
./seismic-shift -n 1000 -d 100000 -k 100 -t 1 -v 
```

The serial results of reducing the dimension of 1,000 randomly generated vectors from 100,000 to 100 are:

<img width="601" alt="image" src="https://github.com/user-attachments/assets/76e798b2-ca35-467e-a43d-32e687d74828" />

To give another example, with 8 threads, it takes ~1 minute (without dense JLT, only sparse JLT) to run:

```
./seismic-shift -n 10000 -d 100000 -k 100 -t 8 -N -v
```

The multicore results of reducing the dimension fo 10,000 randomly generated vectors from 100,000 to 100 are:

<img width="607" alt="image" src="https://github.com/user-attachments/assets/89a0e6e2-070a-4961-989c-bf09a07b85ab" />

## Unit tests

To see the unit tests for each component of the SeismicShift algorithm, run the following from the root directory:

```
// If fresh copy...
cmake -B build -S .

// To build...
cd build
make clean; make
./tests
```
All tests should pass, with the following breakdown:

<img width="430" alt="image" src="https://github.com/user-attachments/assets/0241d29a-e3fb-40f2-a19e-e43daf8ceeb1" />



