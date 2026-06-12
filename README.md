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

## Testing dynamic updates (insert / delete)

SeismicShift supports building a *dynamic* index that can be updated in place via
`insert_doc`, `delete_doc`, and a `resize` (compaction) pass. To exercise and verify
this path end-to-end there is a dedicated driver, `test_dynamic`, which loads a built
index, inserts a batch of documents, deletes them again, and checks both correctness
(inserted docs become retrievable; deleted docs disappear after `resize`) and the
per-operation latency.

### 1. Build a dynamic-support index

The index must be built with `--dynamic-support true` so the document→block membership
needed for in-place updates is recorded. Two knobs control how summaries behave under
updates:

- `--summarization max | centroid` — `max` summaries must be **recomputed** on delete
  (a max cannot be decremented), so deletes are expensive; `centroid` summaries support
  cheap incremental updates.
- `--transform none | jlt` — when `jlt` is set, insert/delete apply the JLT to the
  document vector before updating the summary. With `none` the JLT is not used.

```
./build/bin/build_inverted_index \
  --input-file  [path]/documents.bin \
  --output-file [path]/indexes/my_dynamic_index \
  --n-postings 4000 --summary-energy 0.4 --centroid-fraction 0.1 --knn 0 \
  --clustering-algorithm random-kmeans \
  --dynamic-support true --summarization max --transform none
```

This writes `[path]/indexes/my_dynamic_index.index.seismic`.

### 2. Run the dynamic verification driver

`test_dynamic` takes the index base path (without the `.index.seismic` suffix). It uses
the query vectors as the documents to insert.

```
./build/bin/test_dynamic \
  --index-file [path]/indexes/my_dynamic_index \
  --query-file [path]/data/queries.bin \
  --n-ops 200
```

Expected output (numbers are from a 1M-document SPLADE subset, `summarization max`,
`transform none`):

```
Loaded. documents=1000000 dim=28674
[baseline] search returned 10 results (top score 13.9759)
[insert] inserting 200 docs ...
[insert] len 1000000 -> 1000200 | retrievable 198/200 | 19510.5 us/insert
[delete] deleting the 200 just-inserted docs ...
[delete] removed 200 docs | gone 200/200 | 430275 us/delete | resize 19422 ms
[final] search returned 10 results after insert+delete+resize
==== DYNAMIC VERIFICATION PASSED ====
```

Notes:
- `retrievable k/N` counts how many freshly inserted docs are returned when searching
  for their own vector; a couple of misses on an approximate index (heap-factor < 1,
  coarse blocks) are expected, not a correctness failure.
- `gone N/N` confirms deleted docs are no longer retrievable after `resize()`.
- Delete latency is high with `--summarization max` because each delete recomputes the
  affected block summaries; use `--summarization centroid` for delete-heavy workloads.

### 3. Centroid summaries + JLT (recommended for dynamic workloads)

`--summarization centroid` keeps a running centroid per block, so deletes (and inserts)
update the summary **incrementally** instead of recomputing it. Combining it with
`--transform jlt` (which projects summaries into a low-dimensional dense space) gives a
fully dynamic index. Build it the same way, swapping the two flags:

```
./build/bin/build_inverted_index \
  --input-file  [path]/documents.bin \
  --output-file [path]/indexes/my_centroid_jlt_index \
  --n-postings 4000 --summary-energy 0.4 --centroid-fraction 0.1 --knn 0 \
  --clustering-algorithm random-kmeans \
  --dynamic-support true --summarization centroid --transform jlt
```

Then verify it exactly as above:

```
./build/bin/test_dynamic \
  --index-file [path]/indexes/my_centroid_jlt_index \
  --query-file [path]/data/queries.bin \
  --n-ops 200
```

On the 1M SPLADE subset, centroid summaries make updates dramatically cheaper than
`max` while staying correct (per-operation latency, 200 ops, single-threaded):

| Summarization / transform | Insert | Delete | Insert correctness | Delete correctness |
| ------------------------- | ------ | ------ | ------------------ | ------------------ |
| `max` / `none`            | 19.5 ms | 430 ms | 198/200 retrievable | 200/200 gone |
| `centroid` / `jlt`        | 5.1 ms  | 0.40 ms | 198/200 retrievable | 200/200 gone |

The ~1000x faster delete comes from `max` having to recompute each affected block
summary (a max cannot be decremented) whereas `centroid` simply removes the doc's
contribution from the running mean. (`resize()` compaction is summary-independent and
takes ~19 s over the 1M index in both cases.)

## Full-scale reproduction (8.8M MS MARCO SPLADE, static)

With paper-faithful settings (`--n-postings 4000 --block-size 400 --summarization max
--transform none`, no `--dynamic-support`) the C++ port reproduces the published Seismic
recall, at sub-millisecond-to-low-millisecond latency. Build ~13 min; index ~12 GB.

| Subsection | query-cut / heap-factor | Latency | RR@10 |
| ---------- | ----------------------- | ------- | ----- |
| recall_90  | 3 / 0.9                 | 855 µs  | 0.369 |
| recall_91  | 4 / 0.9                 | 1.11 ms | 0.373 |
| recall_93  | 5 / 0.9                 | 1.31 ms | 0.373 |
| recall_94  | 8 / 0.9                 | 1.95 ms | 0.376 |
| recall_95  | 5 / 0.8                 | 1.54 ms | 0.376 |
| recall_97  | 6 / 0.7                 | 2.16 ms | 0.377 |

(Exact-search groundtruth RR@10 = 0.383.) Run with
`python3 script/run_experiments.py --exp experiments/sigir2024/splade_full_repro.toml`,
or sweep an existing index with `script/sweep_pruned.sh`.

NOTE: paper-faithful results require `--block-size ~400` (≈ `0.1 × n-postings`, the
original Seismic granularity). Smaller values build faster but coarsen blocks and hurt
query latency/recall.

## Dynamic summary-metric tradeoff (centroid+jlt vs max, 8.8M)

Comparing the two dynamic configurations on the full collection (both
`--dynamic-support true --block-size 400`, 500 insert/delete ops). The headline: the
choice is a **query-speed vs update-speed tradeoff that is essentially recall-neutral**.

| Axis              | `max` / `none` | `centroid` / `jlt` | Winner            |
| ----------------- | -------------- | ------------------ | ----------------- |
| Recall (RR@10)    | 0.370–0.377    | 0.371–0.378        | tie               |
| Query latency     | 0.85–2.1 ms    | 5.7–14.9 ms        | max (~6–7×)       |
| Insert            | 18.5 ms        | 10.1 ms            | centroid (~1.8×)  |
| Delete            | 16.1 ms        | 0.42 ms            | centroid (~38×)   |
| resize()          | 20.7 s         | 20.4 s             | tie               |
| Index size        | 13.4 GB        | 11.4 GB            | centroid          |

`max` summaries are true upper bounds, enabling principled threshold pruning (skips
aggressively, ~500 docs/query) — fast queries but expensive deletes (block summary must
be recomputed). `centroid` summaries have no upper bound, so query pruning falls back to
a count-based heuristic (evaluates ~75% of blocks) — slower queries, but deletes are an
O(summary) incremental update. Recall ends up equivalent. Rule of thumb: `max` for
read-heavy/mostly-static collections, `centroid+jlt` for delete/update-heavy workloads
(smaller index, far cheaper deletes, equal recall). Reproduce with
`script/sweep_dyn_compare.sh`.

## Centroid + radius: fast queries AND cheap deletes

`centroid` summaries delete cheaply (incremental mean update) but query slowly: a
centroid is not an upper bound on the block's scores, so the search can't prune
principledly and falls back to a count-based heuristic (evaluates ~75% of blocks).
`max` summaries are upper bounds (fast pruning) but delete slowly (the block summary
must be recomputed). The **centroid + radius** scheme gets both.

Store, per block, one extra scalar — the residual radius `R = max_d ||d - mu||`. Then
for a non-negative sparse query `q`,
```
U = q·mu + alpha * ||q|| * R    (alpha in (0,1])
```
is an (approximate) upper bound on any doc's score in the block (Cauchy–Schwarz, with
`alpha` discounting the worst-case slack — see `diag_radius`), which lets `centroid`
reuse the same threshold-skip as `max`. `alpha` is set at query time via
`perf_inverted_index --alpha A` (`alpha = 0` falls back to the count-based heuristic).

`R` is maintained incrementally on update in `O(nnz)` — no block scan — using the fact
that an insert/delete shifts the centroid by a bounded `delta` (triangle inequality):
`delete: R += ||d-mu||/(n-1)`, `insert: R = max(R + ||d-mu||/(n+1), ||d - mu_new||)`.
The update only ever over-estimates, so it is recall-safe (never wrongly skips; it just
loosens until the next `resize()` recomputes a tight `R`).

Build a centroid+radius index (radius is stored when `transform == none`) and sweep
`alpha` with `script/sweep_alpha.sh`:
```
./build/bin/build_inverted_index --input-file [path]/documents.bin \
  --output-file [path]/indexes/centroid_radius \
  --n-postings 4000 --block-size 400 --summarization centroid --transform none
./build/bin/perf_inverted_index --index-file [path]/indexes/centroid_radius -k 10 \
  --query-file [path]/data/queries.bin --query-cut 3 --heap-factor 0.9 --alpha 0.2 \
  --output-path /tmp/out
```

Full 8.8M results (recall_90, cut3/hf0.9). `alpha` trades query latency for recall;
the sweet spot (`alpha ~ 0.15-0.2`) gives full recall at several-fold speedup:

| alpha | Query latency | RR@10 |
| ----- | ------------- | ----- |
| 0 (count-based) | 7.67 ms | 0.3735 |
| 0.1   | 0.87 ms | 0.3625 |
| 0.15  | 1.59 ms | 0.3713 |
| 0.2   | 2.90 ms | 0.3732 |
| >=0.3 | ~7-8 ms (bound too loose; no pruning) | 0.3734 |

Net, on the same index: **query 1.6-2.9 ms (near the `max` config's 0.86 ms), delete
0.125 ms (vs `max`'s 16 ms), recall ~0.371-0.373 (full)** — fast queries, cheap deletes,
and full recall together. Verify update cost/correctness with
`test_dynamic --index-file [path]/indexes/centroid_radius --query-file [path]/data/queries.bin`.

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



