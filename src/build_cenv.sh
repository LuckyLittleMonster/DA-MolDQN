#!/bin/bash
set -ex

# =============================================================================
# Universal cenv.so build script for GH200 (aarch64) / maple cluster
#
# Usage:
#   conda activate rl4    # or any target env
#   cd src && bash build_cenv.sh
#
# Requirements:
#   - CONDA_PREFIX set (auto by conda activate)
#   - RDKit source tree with Code/ and lib/ (default: ~/data/rdkit.old2)
#   - Boost (python, numpy, iostreams, regex, filesystem, system) in conda env
#   - Cairo (system)
# =============================================================================

# --- Configuration ---
DEFAULT_RDBASE=/home/l1062811/data/rdkit-Release_2025_09_5

if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: CONDA_PREFIX not set. Run 'conda activate <env>' first."
    exit 1
fi

# Use RDBASE if set AND valid, otherwise fall back to default
if [ -n "$RDBASE" ] && [ -f "$RDBASE/Code/cmake/Modules/FindNumPy.cmake" ]; then
    : # RDBASE is valid, keep it
else
    if [ -n "$RDBASE" ]; then
        echo "WARNING: RDBASE=$RDBASE is invalid (missing cmake modules), using default"
    fi
    RDBASE="$DEFAULT_RDBASE"
fi

# Auto-detect Python version tag (e.g., 313, 311)
PYVER=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
PYVER_DOT=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
NUMPY_INC=$(python -c "import numpy; print(numpy.get_include())")
PYTHON_INC="$CONDA_PREFIX/include/python${PYVER_DOT}"

echo "=== Build config ==="
echo "CONDA_PREFIX = $CONDA_PREFIX"
echo "RDBASE       = $RDBASE"
echo "Python       = ${PYVER_DOT} (tag: ${PYVER})"
echo "NumPy inc    = $NUMPY_INC"
echo "Python inc   = $PYTHON_INC"
echo "===================="

# Verify key paths
[ -f "$RDBASE/Code/cmake/Modules/FindNumPy.cmake" ] || { echo "ERROR: RDBASE missing cmake modules"; exit 1; }
[ -d "$RDBASE/lib" ] || { echo "ERROR: RDBASE missing lib/"; exit 1; }
[ -f "$PYTHON_INC/pyconfig.h" ] || { echo "ERROR: Python headers missing at $PYTHON_INC"; exit 1; }
[ -f "$CONDA_PREFIX/lib/libboost_python${PYVER}.so" ] || { echo "ERROR: libboost_python${PYVER}.so not found"; exit 1; }
[ -f "$CONDA_PREFIX/lib/libboost_numpy${PYVER}.so" ] || { echo "ERROR: libboost_numpy${PYVER}.so not found"; exit 1; }

# --- Build ---
BUILD_DIR="build_py${PYVER}"
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DRDBASE="$RDBASE" \
    -DBoost_INCLUDE_DIR="$CONDA_PREFIX/include" \
    -DBoost_LIBRARY_DIR="$CONDA_PREFIX/lib" \
    -DBOOST_ROOT="$CONDA_PREFIX" \
    -DBoost_No_SYSTEM_PATHS=ON \
    -DBoost_USE_STATIC_LIBS=OFF \
    -DBoost_PYTHON_VERSION="$PYVER" \
    -DPYTHON_NUMPY_INCLUDE_PATH="$NUMPY_INC" \
    -Wno-dev

# Workaround: CMakeLists.txt doesn't reliably pass Python include dir to compiler.
# Use CPLUS_INCLUDE_PATH to inject it.
CPLUS_INCLUDE_PATH="$PYTHON_INC" make -j$(nproc)

# --- Verify output ---
# CMakeLists.txt sets EXECUTABLE_OUTPUT_PATH to src/, so cenv.so lands there directly.
SCRIPT_DIR="$(cd .. && pwd)"
if [ ! -f "$SCRIPT_DIR/cenv.so" ]; then
    # Fallback: copy from build dir if output path wasn't set
    cp cenv.so "$SCRIPT_DIR/cenv.so" 2>/dev/null || { echo "ERROR: cenv.so not found"; exit 1; }
fi

echo ""
echo "=== Build complete ==="
echo "Output: src/cenv.so (Python ${PYVER_DOT}, $(uname -m))"
echo ""
echo "Runtime requirement:"
echo "  export LD_LIBRARY_PATH=/home/l1062811/data/envs/rdkit/lib:${RDBASE}/lib:\$LD_LIBRARY_PATH"
