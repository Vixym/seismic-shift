# SeismicShift

Building a C++ version of Seismic (https://github.com/TusKANNy/seismic) with support for dynamic updates + potential other optimizations.

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



