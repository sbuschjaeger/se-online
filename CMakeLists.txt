cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(PSGDEnsemble LANGUAGES CXX)

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

# Since most of our machine have older gcc's installed I installed a newer version into a conda environment.
# CMake has some troubles using this compiler which leads to it not finding xtensor. If we unset the
# implicit directories it works, but this may break other stuff. See:
#   https://stackoverflow.com/questions/58541131/why-is-cmake-removing-some-include-directories-from-include-directories-calls/58589570#58589570 
#   https://gitlab.kitware.com/cmake/cmake/-/issues/17966

# unset(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES)
# unset(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)

add_subdirectory(pybind11)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

include_directories("./include")

add_executable(run_magic magic/run.cpp)
target_include_directories(run_magic PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(run_magic PRIVATE xtensor xtensor::optimize xtensor::use_xsimd)

add_executable(run_covtyp covertype/run.cpp)
target_include_directories(run_covtyp PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(run_covtyp PRIVATE xtensor xtensor::optimize xtensor::use_xsimd)

pybind11_add_module(PyBPE include/Python.cpp)
target_include_directories(PyBPE PRIVATE ${xtensor_INCLUDE_DIRS})
target_link_libraries(PyBPE PRIVATE xtensor xtensor::optimize xtensor::use_xsimd)
