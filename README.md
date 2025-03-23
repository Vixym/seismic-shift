# SeismicShift

Building a C++ version of Seismic (https://github.com/TusKANNy/seismic) with support for dynamic updates + potential other optimizations.

## Testing out SeismicShift's sparse JLT implementation

From the root directory, run:

```
cd build
make
./seismic-cpp [num vectors] [original dimension] [target dimension]
```

As an example, serially, it takes ~2.5 minutes to run:

```
./seismic-cpp 1000 100000 100
```

The results of reducing the dimension of 1,000 randomly generated vectors from 100,000 to 100 are:
