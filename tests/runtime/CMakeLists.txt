cmake_minimum_required(VERSION 3.14)
project(deepx_runtime_tests CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------
# Locate the runtime library and its headers.
#
# Override defaults by passing -DRUNTIME_LIB_DIR=<path> to cmake.
# -----------------------------------------------------------------------
if(NOT DEFINED RUNTIME_LIB_DIR)
    set(RUNTIME_LIB_DIR "${CMAKE_SOURCE_DIR}/../../runtime-library/artifacts"
        CACHE PATH "Directory containing libRuntimeLibrary.so")
endif()

if(NOT DEFINED RUNTIME_INCLUDE_DIR)
    set(RUNTIME_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../runtime-library/include"
        CACHE PATH "Directory containing runtime_core.h")
endif()

find_library(RUNTIME_LIB
    NAMES RuntimeLibrary
    PATHS ${RUNTIME_LIB_DIR}
    PATH_SUFFIXES
        x86_64-ubuntu22.04
        x86_64-ubuntu20.04
        x86_64-ubuntu24.04
        aarch64-ubuntu22.04
        aarch64-ubuntu20.04
        aarch64-ubuntu24.04
    REQUIRED
)

message(STATUS "Using runtime library: ${RUNTIME_LIB}")
message(STATUS "Using runtime headers: ${RUNTIME_INCLUDE_DIR}")

find_package(Threads REQUIRED)

add_executable(inference_runner inference_runner.cpp)

target_include_directories(inference_runner PRIVATE ${RUNTIME_INCLUDE_DIR})
target_link_libraries(inference_runner PRIVATE ${RUNTIME_LIB} Threads::Threads)

# Use $ORIGIN so the binary finds libRuntimeLibrary.so from its own directory
set_target_properties(inference_runner PROPERTIES
    BUILD_RPATH  "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)
