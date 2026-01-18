cmake_minimum_required(VERSION 3.14)
project(dx_oaax_standard DESCRIPTION "OAAX Standard for DEEPX AI Accelerator.")

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# Read OAAX_VERSION from user-defined argument
if(NOT DEFINED OAAX_RUNTIME_VERSION)
    message(FATAL_ERROR "OAAX_RUNTIME_VERSION is not set. Please set it to the desired version.")
endif()
add_definitions(-DOAAX_RUNTIME_VERSION="${OAAX_RUNTIME_VERSION}")

# Common compiler flags (platform-specific)
if (UNIX)
    add_compile_options(-W -Wall -fstrict-volatile-bitfields -fPIC)

    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -rdynamic)
    endif()
elseif (WIN32)
    # Enable UTF-8 source/diagnostics on MSVC (required by spdlog/fmt)
    add_compile_options(/utf-8)
endif()

# DXRT
if (WIN32)
    # On Windows, dx_rt is provided as a prebuilt SDK; use DXRT_ROOT to locate it
    set(DXRT_ROOT "" CACHE PATH "Path to DX-RT Windows SDK (must contain include/ and lib/)")
    if (NOT DXRT_ROOT)
        message(FATAL_ERROR "DXRT_ROOT is not set. Please configure it with -DDXRT_ROOT=PATH to the DX-RT Windows SDK.")
    endif()
    # Silence MSVC "unsafe" CRT and POSIX name deprecation warnings for third-party C utilities
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE)
else()
    find_package(dxrt REQUIRED)
    if(dxrt_FOUND)
        message(STATUS "dxrt package found!")
    else()
        message(FATAL_ERROR "dxrt package not found!")
    endif()
endif()

# Threads (for std::thread/std::condition_variable)
find_package(Threads REQUIRED)

# Add source files
set(SOURCES
    src/runtime_core.cpp
    deps/src/tensors_struct.c
)

# Add header files
set(HEADERS
    include/runtime_core.h
    deps/include/tensors_struct.h
)

# Create shared library
add_library(RuntimeLibrary SHARED ${SOURCES} ${HEADERS})

# Define export macro when building the library (used by runtime_core.h)
target_compile_definitions(RuntimeLibrary PRIVATE RUNTIME_LIBRARY_EXPORTS)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/deps/include
)

if (WIN32)
    target_include_directories(RuntimeLibrary
        PRIVATE
            ${DXRT_ROOT}/include
    )
endif()

# Link libraries
target_link_libraries(RuntimeLibrary 
    PRIVATE 
        dxrt 
        Threads::Threads
)

if (WIN32)
    # Link against dxrt.lib from the SDK
    target_link_directories(RuntimeLibrary
        PRIVATE
            ${DXRT_ROOT}/lib
    )
endif()

# Linker flags (basic flags, specific version compatibility should be handled by toolchain)
if (UNIX)
    set_property(TARGET RuntimeLibrary APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--as-needed -Wl,--hash-style=gnu")
endif()

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

# Install auxiliary public headers that live under deps/include
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/deps/include/tensors_struct.h
    DESTINATION include
)