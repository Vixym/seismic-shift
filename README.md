Building a C++ version of Seismic (https://github.com/TusKANNy/seismic) with support for dynamic updates + potential other optimizations.

## Testing out SeismicShift's sparse JLT implementation

From the root directory, run:

```
// If fresh copy...
cmake -B build -S .

// To build...
cd build
make clean; make
./seismic-cpp [num vectors] [original dimension] [target dimension]
```

As an example, serially, it takes ~2 minutes to run:

```
./seismic-cpp 1000 100000 100
```

The serial results of reducing the dimension of 1,000 randomly generated vectors from 100,000 to 100 are:

<img width="594" alt="image" src="https://github.com/user-attachments/assets/1c4467aa-ac72-4629-92cd-716487f8c8aa" />

To give another example, with 8 threads, it takes ~1 minute (without dense JLT, only sparse JLT) to run:

`./sparse-jlt-benchmark -n 10000 -d 100000 -k 100 -N -v`

The multicore results of reducing the dimension fo 10,000 randomly generated vectors from 100,000 to 100 are:

<img width="514" alt="image" src="https://github.com/user-attachments/assets/885cb6e7-b9c8-4ef4-9af6-a5a5c21a16fc" />

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



