cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(PSGDEnsemble LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Include pybind11
# add_subdirectory(lib/pybind11)

###################################################################
# TARGET CONFIGURATION
###################################################################

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
# SET(CMAKE_CXX_EXTENSIONS OFF)
# SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -g")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native -mtune=native")

###################################################################
# TARGETS
###################################################################

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

find_package(xtensor-python REQUIRED)
find_package(pybind11 REQUIRED)
find_package(NumPy REQUIRED)

message(STATUS "Found numpy: ${NUMPY_INCLUDE_DIRS}")

# These are my includes
include_directories("./include")

# xtensor-python requires numpy includes
include_directories(${NUMPY_INCLUDE_DIRS})

# We assume that we run / install this via a conda environment
#include_directories("${CONDA_PREFIX}/include/")
#message("${CONDA_PREFIX}/include/")

add_executable(run_magic ${SOURCES} magic/run.cpp)

target_link_libraries(run_magic xtensor xtensor::optimize xtensor::use_xsimd xtensor-python)
pybind11_add_module(PyBPE include/Python.cpp)

# add_executable(main tests/main.cpp)

# add_subdirectory(pybind11)
# pybind11_add_module(PySSM include/Python.cpp)

# if (CREATE_EXPERIMENTS)
#     # target_link_libraries(threesieves-tests gtest gtest_main gcov)
#     # target_compile_options(threesieves-tests PUBLIC -DTESTFILES_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/test-files/")
#     # set_target_properties(threesieves-tests PROPERTIES LINKER_FLAGS "-fprofile-arcs -ftest-coverage")
# endif ()
