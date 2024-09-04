[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13683379.svg)](https://doi.org/10.5281/zenodo.13683379)

# Many-task-based HPX implementation of LULESH

This is an HPX implementatoin of the LULESH proxy application using a many-task-based approach.
It is based on the OpenMP/MPI-based reference implementation found at https://github.com/LLNL/LULESH. A non-task-based HPX implementation can be found at https://github.com/weilewei/lulesh-hpx.

## Publication

This implementation is used for the following publication. You may use this repository to reproduce the presented results.

[Kalkhof2024] Torben Kalkhof, and Andreas Koch. 2024. **Speeding-Up LULESH on HPX: Useful Tricks and Lessons Learned using a Many-Task-Based Approach**. In *The 7th Annual Parallel Applications Workshop, Alternatives To MPI+X (PAW-ATM)*.

## Build

In the following, we describe all required steps to compile our implementation including the required dependencies and the reference implementation.

### Prerequisites

The artifacts were build using GCC 13.1.1. On *Fedora*, install all required dependencies using:

```bash
dnf install wget git gcc-c++ cmake autoconf autogen automake python3 pip
pip install seaborn
```

On *Ubuntu*, use:

```bash
apt install wget git g++ cmake autoconf autogen automake python3 pip
pip install seaborn
```

### Option #1: Build with compile script

Use our provided compile script which automatically build the evaluation software, including fetching and building dependencies and the reference implementation.

```bash
cd scripts && bash compile.sh
```

### Option #2: Build manually

Clone this repository and export base directory. We refer to the base directory of this repository as ```$BASE```. Return to ```$BASE``` for each of the following steps.
```bash
git clone https://github.com/esa-tu-darmstadt/lulesh-task-based-hpx.git
cd lulesh-task-based-hpx && export BASE=$PWD
```

- Create required install directories and export environment variables:
```bash
export Jemalloc_ROOT=$BASE/jem-install
export HPX_DIR=$BASE/hpx-install
mkdir $Jemalloc_ROOT $HPX_DIR
 ```

- Clone JEMalloc from https://github.com/jemalloc/jemalloc and build using:
```bash
git clone https://github.com/jemalloc/jemalloc.git && cd jemalloc
./autogen.sh --prefix=$Jemalloc_ROOT && make -j && make install
```

- Clone HPX from https://github.com/STEllAR-GROUP/hpx and checkout tag v1.10.0:
```bash
git clone https://github.com/STEllAR-GROUP/hpx.git && cd hpx && git checkout v1.10.0
```

- Create build directory and build using *CMake*:
```bash
mkdir hpx-build && cd hpx-build
cmake -DHPX_WITH_MALLOC=jemalloc \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HPX_DIR \
  -DHPX_WITH_FETCH_ASIO=ON \
  -DHPX_WITH_FETCH_BOOST=ON \
  -DHPX_WITH_FETCH_HWLOC=ON \
  -DHPX_WITH_EXAMPLES=OFF $BASE/hpx
cmake --build . --target install
```

- Create build directory for LULESH HPX implementation and build using *CMake*:
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
```

- Clone LULESH reference implementation from https://github.com/LLNL/LULESH and apply our patch. This patch adds the same compiler flags as used in our implementation andCSV compatible output to simplify result analysis.
```bash
git clone https://github.com/LLNL/LULESH.git && cd LULESH && git apply $BASE/patches/ae.patch
```

- Create build directory for LULESH reference implementation and build using *CMake*:
```bash
mkdir build-ref && cd build-ref
cmake -DCMAKE_BUILD_TYPE=Release $BASE/LULESH && make -j
```

## Run

In the following we describe how to run the experiments to reproduce the results presented in our publication.

### Option #1: Run script

We provide two different scripts to run the experiments. `run-full.sh` performs a full evaluation which may take several days. `run-reduced.sh` limits the iterations of the main algorithm and is expected to take about 4 hours. Run the script as follows:
```bash
cd scripts && bash run-reduced.sh
```

### Option #2: Run experiments manually

The following table lists the relevant flags for our HPX implementation and the OpenMP reference.

HPX flag      | OpenMP flag     | Description
--------------|-----------------|------------
--s           | -s              | Set problem size
--r           | -r              | Set number of regions (default: 11)
--i           | -i              | Number of iterations
--q           | -q              | Suppress verbose output
--hpx:threads | OMP_NUM_THREADS | Number of execution threads

Note that the number of execution threads is not passed as program argument but set by the environment variable `OMP_NUM_THREADS` in OpenMP.

The command for a run with problem size 60, 21 regions, 24 execution threads and 2000 iterations would look as follows for our HPX implementation:
```bash
build/lulesh-hpx --s 60 --r 21 --q --i 2000 --hpx:threads=24
```
The corresponding command for running the LULESH reference implementation is:
```bash
OMP_NUM_THREADS=24 build-ref/lulesh2.0 -s 60 -r 21 -q -i 2000
```

To reproduce the results in our publication, run the following two experiments.

1. For each problem size out of [45, 60, 75, 90, 120, 150], run with 1, 2, 4, 8, 16, 24, 32 and 48 execution threads. Leave out the `--r` flag or set it to 11 regions. You may vary the number of threads if you do not evaluate on a 24-core CPU.
2. Now set the number of regions to 11, 16 and 21 regions for each problem size, but set the number of execution threads to 24 for all runs.

The output of each run is in a CSV format. Create two separate CSV files, one for the HPX and reference implementation each, and save the output of each run in the respective file. Use the following CSV header:
```
size,regions,iterations,threads,runtime,result
```

Since a full evaluation may take several days, a reduced evaluation can be performed by limiting the number of iterations of the main algorithm. For a reduced evaluation, we suggest using the following numbers of maximum iterations for problem sizes above 60:

Problem size | Number of iterations
------------:|--------------------:
 75          | 1500
 90          |  770
120          |  360
150          |  180

### Additional Note

On some systems, you may have to set `LD_LIBRARY_PATH` to run the HPX software:

```bash
export LD_LIBRARY_PATH=$BASE/hpx-build/hpx-build/_deps/hwloc-installed/lib
```

## Analysis

We provide a Python script which generates graphs out of the measurement results comparable to the graphs presented in our publication. For the first experiment, runtime is plotted over the number of execution threads for each problem size. For the second experiment, first the speed-up of the HPX implementation is calculated by dividing the OpenMP runtime through the HPX runtime, and then plotted for each problem size and number of regions.

If the CSV files have not been generated with the provided run script, they should be passed to the Python script as follows:
```bash
cd scripts && python3 generate-graphs.py /path/to/hpx.csv /path/to/reference.csv
```

The script requires the `pandas`, `matplotlib`and `seaborn` packages.
