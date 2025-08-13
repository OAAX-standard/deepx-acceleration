cmake_minimum_required(VERSION 3.14)
project(dx_oaax_standard DESCRIPTION "OAAX Standard for DEEPX AI Accelerator.")

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# Common compiler flags
add_compile_options(-W -Wall -pthread -fstrict-volatile-bitfields -fPIC)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -rdynamic)
endif()

# DXRT 
find_package(dxrt REQUIRED)
if(dxrt_FOUND)
    message(STATUS "dxrt package found!")
else()
    message(FATAL_ERROR "dxrt package not found!")
endif()

# OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# Add source files
set(SOURCES
    src/runtime_core.cpp
    deps/tools/c-utilities/src/queue.c
    deps/tools/c-utilities/src/logger.c
    deps/tools/c-utilities/src/tensors_struct.c
)

# Add header files
set(HEADERS
    include/runtime_core.h
    include/concurrentqueue.h
    deps/tools/c-utilities/include/queue.h
    deps/tools/c-utilities/include/logger.h
    deps/tools/c-utilities/include/tensors_struct.h
)

# Create shared library
add_library(RuntimeLibrary SHARED ${SOURCES} ${HEADERS})

# Set C++ standard
set_target_properties(RuntimeLibrary PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(RuntimeLibrary 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/deps/tools/c-utilities/include
        ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(RuntimeLibrary 
    PRIVATE 
        dxrt 
        pthread
        ${OpenCV_LIBS}
)

# Linker flags (basic flags, specific version compatibility should be handled by toolchain)
set_property(TARGET RuntimeLibrary APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--as-needed -Wl,--hash-style=gnu")

# Install rules
install(TARGETS RuntimeLibrary
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)