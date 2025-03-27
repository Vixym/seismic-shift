# SeismicShift

Building a C++ version of Seismic (https://github.com/TusKANNy/seismic) with support for dynamic updates + potential other optimizations.

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

<img width="429" alt="image" src="https://github.com/user-attachments/assets/0dc13c3f-ed04-4cb4-932f-9b915ea5aec8" />



