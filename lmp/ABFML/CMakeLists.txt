cmake_minimum_required(VERSION 3.14)

project(pair_abfml_plugin LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional: Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# To locate libtorch, specify its root using CMAKE_PREFIX_PATH if it's not in a standard path
# Example: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# Find libtorch
find_package(Torch REQUIRED)

# Manually specify the LAMMPS path (recommended via environment variable)
if(NOT DEFINED ENV{LAMMPS_DIR})
    message(FATAL_ERROR "❌ LAMMPS_DIR environment variable is not set")
endif()

set(LAMMPS_DIR $ENV{LAMMPS_DIR})
set(LAMMPS_HEADER_DIR ${LAMMPS_DIR}/src)
find_path(LAMMPS_INCLUDE_DIR lammps.h PATHS ${LAMMPS_DIR}/src)
find_library(LAMMPS_LIB NAMES lammps PATHS ${LAMMPS_DIR}/build)

if(NOT LAMMPS_INCLUDE_DIR OR NOT LAMMPS_LIB)
    message(FATAL_ERROR "❌ Failed to find LAMMPS headers or library. Please check LAMMPS_DIR")
endif()

# Include header paths
include_directories(${LAMMPS_INCLUDE_DIR})
include_directories(${TORCH_INCLUDE_DIRS})

# Add source files
add_library(abfmlplugin MODULE
    pair_abfml.cpp
    abfmlplugin.cpp
)

# Set include and link dependencies
target_include_directories(abfmlplugin PRIVATE ${LAMMPS_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(abfmlplugin PRIVATE ${LAMMPS_LIB} ${TORCH_LIBRARIES})

# Set output properties
set_target_properties(abfmlplugin PROPERTIES
    OUTPUT_NAME "pair_abfml"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output
)

# Print configuration summary
message(STATUS "======================= Configuration Summary =======================")
message(STATUS "LAMMPS_DIR         : ${LAMMPS_DIR}")
message(STATUS "LAMMPS_INCLUDE_DIR : ${LAMMPS_INCLUDE_DIR}")
message(STATUS "LAMMPS_HEADER_DIR  : ${LAMMPS_HEADER_DIR}")
message(STATUS "LAMMPS_LIB         : ${LAMMPS_LIB}")
message(STATUS "Torch Include Dirs : ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch Libraries    : ${TORCH_LIBRARIES}")
message(STATUS "Plugin Output      : ${CMAKE_BINARY_DIR}/output")
message(STATUS "=====================================================================")
