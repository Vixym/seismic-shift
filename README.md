# SeismicShift

Building a C++ version of Seismic (https://github.com/TusKANNy/seismic) with support for dynamic updates + potential other optimizations.

## Testing out SeismicShift's sparse JLT implementation

From the root directory, run:

```
cd build
make clean; make
./seismic-cpp [num vectors] [original dimension] [target dimension]
```

As an example, serially, it takes ~2 minutes to run:

```
./seismic-cpp 1000 100000 100
```

The results of reducing the dimension of 1,000 randomly generated vectors from 100,000 to 100 are:

<img width="594" alt="image" src="https://github.com/user-attachments/assets/1c4467aa-ac72-4629-92cd-716487f8c8aa" />

## Unit tests

To see the unit tests for each component of the SeismicShift algorithm, run the following from the root directory:

```
cd build
make clean; make
./tests
```
All tests should pass, with the following breakdown:

<img width="386" alt="image" src="https://github.com/user-attachments/assets/07faef3d-b245-4560-a934-e4aa736bbdc9" />
