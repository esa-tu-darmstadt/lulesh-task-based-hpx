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
# s = 45
echo "Runs with problem size 45"
for t in 1 2 4 8 16 24 32 48
do
  LD_LIBRARY_PATH=$HWLOC_LIB_PATH $LULESH_HPX_EXEC --s 45 --q --hpx:threads=$t >> $LULESH_HPX_RESULT_FILE
done
for r in 16 21
do
  LD_LIBRARY_PATH=$HWLOC_LIB_PATH $LULESH_HPX_EXEC --s 45 --r $r --q --hpx:threads=24 >> $LULESH_HPX_RESULT_FILE
done

# s = 60
echo "Runs with problem size 60"
for t in 1 2 4 8 16 24 32 48
do
  LD_LIBRARY_PATH=$HWLOC_LIB_PATH $LULESH_HPX_EXEC --s 60 --q --hpx:threads=$t >> $LULESH_HPX_RESULT_FILE
done
for r in 16 21
do
  LD_LIBRARY_PATH=$HWLOC_LIB_PATH $LULESH_HPX_EXEC --s 60 --r $r --q --hpx:threads=24 >> $LULESH_HPX_RESULT_FILE
done

# s = 75 (1500 iterations)
echo "Runs with problem size 75"
for t in 1 2 4 8 16 24 32 48
do
  LD_LIBRARY_PATH=$HWLOC_LIB_PATH $LULESH_HPX_EXEC --s 75 --q --i 1500 --hpx:threads=$t >> $LULESH_HPX_RESULT_FILE
done
for r in 16 21
do
  LD_LIBRARY_PATH=$HWLOC_LIB_PATH $LULESH_HPX_EXEC --s 75 --r $r --q --i 1500 --hpx:threads=24 >> $LULESH_HPX_RESULT_FILE
done

# s = 90 (770 iterations)
echo "Runs with problem size 90"
for t in 1 2 4 8 16 24 32 48
do
  LD_LIBRARY_PATH=$HWLOC_LIB_PATH $LULESH_HPX_EXEC --s 90 --q --i 770 --hpx:threads=$t >> $LULESH_HPX_RESULT_FILE
done
for r in 16 21
do
  LD_LIBRARY_PATH=$HWLOC_LIB_PATH $LULESH_HPX_EXEC --s 90 --r $r --q --i 770 --hpx:threads=24 >> $LULESH_HPX_RESULT_FILE
done

# s = 120 (360 iterations)
echo "Runs with problem size 120"
for t in 1 2 4 8 16 24 32 48
do
  LD_LIBRARY_PATH=$HWLOC_LIB_PATH $LULESH_HPX_EXEC --s 120 --q --i 360 --hpx:threads=$t >> $LULESH_HPX_RESULT_FILE
done
for r in 16 21
do
  LD_LIBRARY_PATH=$HWLOC_LIB_PATH $LULESH_HPX_EXEC --s 120 --r $r --q --i 360 --hpx:threads=24 >> $LULESH_HPX_RESULT_FILE
done

# s = 150 (180 iterations)
echo "Runs with problem size 150"
for t in 1 2 4 8 16 24 32 48
do
  LD_LIBRARY_PATH=$HWLOC_LIB_PATH $LULESH_HPX_EXEC --s 150 --q --i 180 --hpx:threads=$t >> $LULESH_HPX_RESULT_FILE
done
for r in 16 21
do
  LD_LIBRARY_PATH=$HWLOC_LIB_PATH $LULESH_HPX_EXEC --s 150 --r $r --q --i 180 --hpx:threads=24 >> $LULESH_HPX_RESULT_FILE
done

echo "Execute runs for reference implementation"
echo "size,regions,iterations,threads,runtime,result" > $LULESH_REF_RESULT_FILE
# s = 45
echo "Runs with problem size 45"
for t in 1 2 4 8 16 24 32 48
do
  OMP_NUM_THREADS=$t $LULESH_REF_EXEC -s 45 -q >> $LULESH_REF_RESULT_FILE
done
for r in 16 21
do
  OMP_NUM_THREADS=24 $LULESH_REF_EXEC -s 45 -r $r -q >> $LULESH_REF_RESULT_FILE
done

# s = 60
echo "Runs with problem size 60"
for t in 1 2 4 8 16 24 32 48
do
  OMP_NUM_THREADS=$t $LULESH_REF_EXEC -s 60 -q >> $LULESH_REF_RESULT_FILE
done
for r in 16 21
do
  OMP_NUM_THREADS=24 $LULESH_REF_EXEC -s 60 -r $r -q >> $LULESH_REF_RESULT_FILE
done

# s = 75 (1500 iterations)
echo "Runs with problem size 75"
for t in 1 2 4 8 16 24 32 48
do
  OMP_NUM_THREADS=$t $LULESH_REF_EXEC -s 75 -q -i 1500 >> $LULESH_REF_RESULT_FILE
done
for r in 16 21
do
  OMP_NUM_THREADS=24 $LULESH_REF_EXEC -s 75 -r $r -q -i 1500 >> $LULESH_REF_RESULT_FILE
done

# s = 90 (770 iterations)
echo "Runs with problem size 90"
for t in 1 2 4 8 16 24 32 48
do
  OMP_NUM_THREADS=$t $LULESH_REF_EXEC -s 90 -q -i 770 >> $LULESH_REF_RESULT_FILE
done
for r in 16 21
do
  OMP_NUM_THREADS=24 $LULESH_REF_EXEC -s 90 -r $r -q -i 770 >> $LULESH_REF_RESULT_FILE
done

# s = 120 (360 iterations)
echo "Runs with problem size 120"
for t in 1 2 4 8 16 24 32 48
do
  OMP_NUM_THREADS=$t $LULESH_REF_EXEC -s 120 -q -i 360 >> $LULESH_REF_RESULT_FILE
done
for r in 16 21
do
  OMP_NUM_THREADS=24 $LULESH_REF_EXEC -s 120 -r $r -q -i 360 >> $LULESH_REF_RESULT_FILE
done

# s = 150 (180 iterations)
echo "Runs with problem size 150"
for t in 1 2 4 8 16 24 32 48
do
  OMP_NUM_THREADS=$t $LULESH_REF_EXEC -s 150 -q -i 180 >> $LULESH_REF_RESULT_FILE
done
for r in 16 21
do
  OMP_NUM_THREADS=24 $LULESH_REF_EXEC -s 150 -r $r -q -i 180 >> $LULESH_REF_RESULT_FILE
done
