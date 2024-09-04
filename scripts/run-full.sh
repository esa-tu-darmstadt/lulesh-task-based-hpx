#!/bin/bash

# create build directories
BASE=$PWD/..
RESULT_DIR=$BASE/results
LULESH_HPX_EXEC=$BASE/build/lulesh-hpx
LULESH_REF_EXEC=$BASE/LULESH-reference-build/lulesh2.0
LULESH_HPX_RESULT_FILE=$RESULT_DIR/hpx_results.csv
LULESH_REF_RESULT_FILE=$RESULT_DIR/omp_results.csv
HWLOC_LIB_PATH=$BASE/hpx-build/hpx-build/_deps/hwloc-installed/lib

mkdir -p $RESULT_DIR

echo "Execute runs for HPX implementation"
echo "size,regions,iterations,threads,runtime,result" > $LULESH_HPX_RESULT_FILE
for s in 45 60 75 90 120 150
do
  echo "Runs with problem size $s"
  for t in 1 2 4 8 16 24 32 48
  do
    LD_LIBRARY_PATH=$HWLOC_LIB_PATH $LULESH_HPX_EXEC --s $s --q --hpx:threads=$t >> $LULESH_HPX_RESULT_FILE
  done
  for r in 16 21
  do
    LD_LIBRARY_PATH=$HWLOC_LIB_PATH $LULESH_HPX_EXEC --s $s --r $r --q --hpx:threads=24 >> $LULESH_HPX_RESULT_FILE
  done
done

echo "Execute runs for reference implementation"
echo "size,regions,iterations,threads,runtime,result" > $LULESH_REF_RESULT_FILE
for s in 45 60 75 90 120 150
do
  echo "Runs with problem size $s"
  for t in 1 2 4 8 16 24 32 48
  do
    OMP_NUM_THREADS=$t $LULESH_REF_EXEC -s $s -q >> $LULESH_REF_RESULT_FILE
  done
  for r in 16 21
  do
    OMP_NUM_THREADS=24 $LULESH_REF_EXEC -s $s -r $r -q >> $LULESH_REF_RESULT_FILE
  done
done
