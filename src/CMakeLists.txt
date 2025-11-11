cmake_minimum_required( VERSION 3.5 )

project(RDKitSV)

# Set C++ standard to C++20 (required for std::span and other C++20 features used by RDKit)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(RDBASE $ENV{RDBASE})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${RDBASE}/Code/cmake/Modules")

# note that if you haven't installed/built the toolkit with CoordGen, you'll
# have problems with this.
add_definitions("-DRDK_BUILD_COORDGEN_SUPPORT=ON")
add_compile_options(-Wall)

set(Boost_USE_MULTITHREADED OFF)
set(Boost_USE_STATIC_RUNTIME ON)
set(Boost_NO_BOOST_CMAKE ON)
# Set CXX flags - Boost include will be added after find_package
set(CMAKE_CXX_FLAGS "-fpermissive -fPIC -shared -O3")

# Set Boost paths using CONDA_PREFIX environment variable
if(DEFINED ENV{CONDA_PREFIX})
    set(Boost_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
    set(Boost_LIBRARY_DIR "$ENV{CONDA_PREFIX}/lib")
    set(BOOST_ROOT "$ENV{CONDA_PREFIX}")  # Note: must be BOOST_ROOT (all caps), not Boost_ROOT
    set(Boost_NO_SYSTEM_PATHS ON)
    set(Boost_USE_STATIC_LIBS OFF)  # Conda uses dynamic libraries
    set(Boost_PYTHON_VERSION "310")
    message(STATUS "Using conda environment: $ENV{CONDA_PREFIX}")
else()
    message(FATAL_ERROR "CONDA_PREFIX environment variable is not set.")
endif()
# Note: Boost_INCLUDE_DIR will be set properly after find_package(Boost)

find_package(NumPy REQUIRED)
include_directories(${PYTHON_NUMPY_INCLUDE_PATH})
message(STATUS "PYTHON_NUMPY_INCLUDE_PATH='${PYTHON_NUMPY_INCLUDE_PATH}'")

# Find Boost components (without python/numpy - we'll handle those manually)
find_package( Boost REQUIRED COMPONENTS iostreams regex filesystem system)

# Ensure Boost include directories are set correctly after find_package
# find_package(Boost) sets Boost_INCLUDE_DIRS (plural) or Boost_INCLUDE_DIR (singular)
if(Boost_FOUND)
    # Check what find_package actually set
    if(DEFINED Boost_INCLUDE_DIRS)
        set(BOOST_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
        message(STATUS "Using Boost_INCLUDE_DIRS: ${BOOST_INCLUDE_DIRS}")
    elseif(DEFINED Boost_INCLUDE_DIR)
        set(BOOST_INCLUDE_DIRS ${Boost_INCLUDE_DIR})
        message(STATUS "Using Boost_INCLUDE_DIR: ${BOOST_INCLUDE_DIRS}")
    else()
        # Fallback: use conda prefix
        if(DEFINED ENV{CONDA_PREFIX})
            set(BOOST_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
            message(STATUS "Using CONDA_PREFIX for Boost includes: ${BOOST_INCLUDE_DIRS}")
        else()
            message(FATAL_ERROR "Cannot determine Boost include directory!")
        endif()
    endif()
    message(STATUS "Final BOOST_INCLUDE_DIRS='${BOOST_INCLUDE_DIRS}'")
    # Add Boost include directory globally (for backward compatibility)
    include_directories(${BOOST_INCLUDE_DIRS})
    # Also add directly to CXX flags to ensure it's included
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${BOOST_INCLUDE_DIRS}")
    message(STATUS "Added Boost include to CMAKE_CXX_FLAGS: -I${BOOST_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Boost not found!")
endif()

# Manually add Python and NumPy Boost libraries
if(DEFINED ENV{CONDA_PREFIX})
    set(BOOST_PYTHON_LIB "$ENV{CONDA_PREFIX}/lib/libboost_python310.so")
    set(BOOST_NUMPY_LIB "$ENV{CONDA_PREFIX}/lib/libboost_numpy310.so")
    
    if(EXISTS "${BOOST_PYTHON_LIB}")
        message(STATUS "Found Boost Python: ${BOOST_PYTHON_LIB}")
    else()
        message(FATAL_ERROR "Boost Python library not found: ${BOOST_PYTHON_LIB}")
    endif()
    
    if(EXISTS "${BOOST_NUMPY_LIB}")
        message(STATUS "Found Boost NumPy: ${BOOST_NUMPY_LIB}")
    else()
        message(FATAL_ERROR "Boost NumPy library not found: ${BOOST_NUMPY_LIB}")
    endif()
endif()

find_package( Cairo REQUIRED )

# specify where CMake can find the RDKit libraries

# Find Python include directory from conda environment
if(DEFINED ENV{CONDA_PREFIX})
    find_program(PYTHON_EXECUTABLE python3 PATHS "$ENV{CONDA_PREFIX}/bin" NO_DEFAULT_PATH)
    if(PYTHON_EXECUTABLE)
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
            OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        include_directories(${RDBASE}/Code ${CAIRO_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIR})
        message(STATUS "Python include directory: ${PYTHON_INCLUDE_DIR}")
    else()
        include_directories(${RDBASE}/Code ${CAIRO_INCLUDE_DIRS})
    endif()
else()
    include_directories(${RDBASE}/Code ${CAIRO_INCLUDE_DIRS})
endif()
link_directories ( ${RDBASE}/lib )

set(RDKit_LIBS RDKitChemReactions RDKitFileParsers RDKitSmilesParse RDKitDepictor
         RDKitRDGeometryLib RDKitRDGeneral RDKitSubstructMatch RDKitSubgraphs
         RDKitMolDraw2D RDKitGraphMol RDKitDistGeometry RDKitDistGeomHelpers
           RDKitMolAlign RDKitOptimizer RDKitForceField RDKitForceFieldHelpers
         RDKitAlignment RDKitForceField  RDKitMolTransforms RDKitEigenSolvers RDKitFingerprints)

         

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})

find_package (Threads)
set(RDKit_THREAD_LIBS Threads::Threads)

# set( LIBS ${RDKIT_LIBRARIES} Boost::iostreams Boost::regex Boost::python Boost::numpy ${RDKit_THREAD_LIBS}
#      ${CAIRO_LIBRARIES} )
set( LIBS ${RDKIT_LIBRARIES} 
     Boost::iostreams 
     Boost::regex 
     Boost::filesystem 
     Boost::system
     ${BOOST_PYTHON_LIB}
     ${BOOST_NUMPY_LIB}
     ${RDKit_THREAD_LIBS}
     ${CAIRO_LIBRARIES} )


include_directories(${RDKIT_INCLUDE_DIR})

# Ensure Boost include is available before creating executable
if(DEFINED BOOST_INCLUDE_DIRS)
    include_directories(${BOOST_INCLUDE_DIRS})
elseif(DEFINED ENV{CONDA_PREFIX})
    include_directories("$ENV{CONDA_PREFIX}/include")
endif()

add_executable( cenv.so environment.cpp incremental_morgan_fingerprint.cpp)
# Ensure Boost include directories are added to the target
# Use PUBLIC to ensure include directories are propagated to compilation
if(DEFINED BOOST_INCLUDE_DIRS)
    target_include_directories( cenv.so PUBLIC ${BOOST_INCLUDE_DIRS})
    message(STATUS "Added Boost include to target (PUBLIC): ${BOOST_INCLUDE_DIRS}")
elseif(DEFINED ENV{CONDA_PREFIX})
    target_include_directories( cenv.so PUBLIC "$ENV{CONDA_PREFIX}/include")
    message(STATUS "Added Boost include to target (from CONDA_PREFIX): $ENV{CONDA_PREFIX}/include")
else()
    message(FATAL_ERROR "Cannot determine Boost include directory for target!")
endif()
target_link_libraries( cenv.so ${LIBS} ${RDKit_LIBS})
