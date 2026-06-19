# Spike CMake: build cenv.so against CONDA librdkit (no RDBASE, no source-built rdkit, no Cairo).
# Configure with: cmake -B build -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake_minimum_required(VERSION 3.18)
project(cenv CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- RDKit from conda (provides rdkit-config.cmake + RDKit::* imported targets) ---
find_package(rdkit CONFIG REQUIRED)

# --- Python + NumPy ---
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

# --- Boost: core components via Config; python/numpy libs by version-suffixed path ---
# NB: in Boost >=1.90 `system` is header-only (no boost_system config) -> use Boost::headers.
find_package(Boost REQUIRED COMPONENTS iostreams regex filesystem)
set(PYSUF "${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}")
set(BOOST_PYTHON_LIB "$ENV{CONDA_PREFIX}/lib/libboost_python${PYSUF}.so")
set(BOOST_NUMPY_LIB  "$ENV{CONDA_PREFIX}/lib/libboost_numpy${PYSUF}.so")
foreach(_l "${BOOST_PYTHON_LIB}" "${BOOST_NUMPY_LIB}")
  if(NOT EXISTS "${_l}")
    message(FATAL_ERROR "Missing boost lib: ${_l}")
  endif()
endforeach()
message(STATUS "Boost python/numpy: ${BOOST_PYTHON_LIB} ; ${BOOST_NUMPY_LIB}")

# --- The module: import name is `cenv` (BOOST_PYTHON_MODULE(cenv)) -> file cenv.so ---
add_library(cenv MODULE environment.cpp incremental_morgan_fingerprint.cpp)
set_target_properties(cenv PROPERTIES PREFIX "" OUTPUT_NAME "cenv" SUFFIX ".so")
target_compile_options(cenv PRIVATE -fpermissive -O3 -Wall)
target_include_directories(cenv PRIVATE
  ${Python3_INCLUDE_DIRS}
  ${Python3_NumPy_INCLUDE_DIRS}
  "$ENV{CONDA_PREFIX}/include")

target_link_libraries(cenv PRIVATE
  # RDKit (imported targets carry include dirs + transitive deps)
  RDKit::SmilesParse RDKit::FileParsers RDKit::Fingerprints RDKit::SubstructMatch
  RDKit::Subgraphs RDKit::GraphMol RDKit::Depictor RDKit::ChemReactions
  RDKit::DistGeometry RDKit::DistGeomHelpers RDKit::MolAlign RDKit::Optimizer
  RDKit::ForceField RDKit::ForceFieldHelpers RDKit::Alignment RDKit::MolTransforms
  RDKit::EigenSolvers RDKit::RDGeometryLib RDKit::RDGeneral
  # Boost
  Boost::headers Boost::iostreams Boost::regex Boost::filesystem
  ${BOOST_PYTHON_LIB} ${BOOST_NUMPY_LIB}
  # Python
  Python3::Python Python3::NumPy)
