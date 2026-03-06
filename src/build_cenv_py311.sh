#!/bin/bash
set -ex

# Activate the right environment
export CONDA_PREFIX=/shared/data1/Users/l1062811/envs/hpc311alf
export RDBASE=/home/l1062811/data/rdkit
export PATH=$CONDA_PREFIX/bin:$PATH

# Set up build directory
mkdir -p build_py311 && cd build_py311

cmake .. \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DRDBASE=$RDBASE \
    -DBoost_INCLUDE_DIR=$CONDA_PREFIX/include \
    -DBoost_LIBRARY_DIR=$CONDA_PREFIX/lib \
    -DBOOST_ROOT=$CONDA_PREFIX \
    -DBoost_NO_SYSTEM_PATHS=ON \
    -DBoost_USE_STATIC_LIBS=OFF \
    -DBoost_PYTHON_VERSION="311" \
    -DPYTHON_NUMPY_INCLUDE_PATH=$($CONDA_PREFIX/bin/python -c "import numpy; print(numpy.get_include())") \
    -DPYTHON_INCLUDE_DIRS=$CONDA_PREFIX/include/python3.11 \
    -DCMAKE_CXX_FLAGS="-fpermissive -fPIC -shared -O3 -I$CONDA_PREFIX/include/python3.11"

make -j8

# Copy the result
cp cenv.so ../cenv.so
echo "Build complete!"
