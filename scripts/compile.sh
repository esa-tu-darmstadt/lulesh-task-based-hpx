#!/bin/bash

# create build directories
BASE=$PWD/..
HPX_BASE=$BASE/hpx-build
JEMALLOC_GIT=$HPX_BASE/jemalloc-git
JEMALLOC_INSTALL=$HPX_BASE/jemalloc-install
HPX_INSTALL=$HPX_BASE/install
HPX_GITHUB=$HPX_BASE/hpx-github
HPX_BUILD=$HPX_BASE/hpx-build
LULESH_REFERENCE_GIT=$BASE/LULESH-reference-git
LULESH_REFERENCE_BUILD=$BASE/LULESH-reference-build
BUILDDIR=$BASE/build
mkdir -p $HPX_BASE $JEMALLOC_GIT $JEMALLOC_INSTALL\
  $HPX_INSTALL $HPX_GITHUB $HPX_BUILD $BUILDDIR \
  $LULESH_REFERENCE_GIT $LULESH_REFERENCE_BUILD

# build jemalloc
cd $HPX_BASE && git clone https://github.com/jemalloc/jemalloc.git $JEMALLOC_GIT && cd $JEMALLOC_GIT
./autogen.sh --prefix=$JEMALLOC_INSTALL && make -j && make install

# build HPX
export Jemalloc_ROOT=$JEMALLOC_INSTALL
git clone https://github.com/STEllAR-GROUP/hpx.git $HPX_GITHUB && cd $HPX_GITHUB && git checkout v1.10.0
cd $HPX_BUILD
cmake -DHPX_WITH_MALLOC=jemalloc \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HPX_INSTALL \
  -DHPX_WITH_FETCH_ASIO=ON \
  -DHPX_WITH_FETCH_BOOST=ON \
  -DHPX_WITH_FETCH_HWLOC=ON \
  -DHPX_WITH_EXAMPLES=OFF \
  $HPX_GITHUB
cmake --build . --target install -j48

# build LULESH HPX
export HPX_DIR=$HPX_INSTALL
cd $BUILDDIR && cmake -DCMAKE_BUILD_TYPE=Release $BASE && make -j

# build LULESH reference code
git clone https://github.com/LLNL/LULESH.git $LULESH_REFERENCE_GIT && cd $LULESH_REFERENCE_GIT
git apply $BASE/patches/ae.patch
cd $LULESH_REFERENCE_BUILD && cmake -DCMAKE_BUILD_TYPE=Release $LULESH_REFERENCE_GIT && make -j
