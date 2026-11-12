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
    # The DX-RT SDK: the DX-SDK Windows Installer's install directory, or dx_rt built from source
    # and installed (see build-windows.ps1).  Both have include/dxrt/, lib/dxrt.lib, lib/cmake/dxrt/.
    set(DXRT_ROOT "" CACHE PATH "DX-RT SDK directory (contains include/dxrt/ and lib/cmake/dxrt/)")
    if (NOT DXRT_ROOT)
        message(FATAL_ERROR "DXRT_ROOT is not set. Configure with -DDXRT_ROOT=<DX-RT SDK directory>.")
    endif()
    if (POLICY CMP0144)
        cmake_policy(SET CMP0144 NEW)  # DXRT_ROOT is meant as dxrt's root (searched explicitly below anyway)
    endif()
    find_package(dxrt REQUIRED CONFIG PATHS "${DXRT_ROOT}" NO_DEFAULT_PATH)
    message(STATUS "dxrt: ${DXRT_LIBRARY}")
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
)

# Add header files
set(HEADERS
    include/runtime_core.h
)

# Create shared library
add_library(RuntimeLibrary SHARED ${SOURCES} ${HEADERS})

# Define export macro when building the library (used by runtime_core.h)
target_compile_definitions(RuntimeLibrary PRIVATE RUNTIME_LIBRARY_EXPORTS)

# Set C++ standard
set_target_properties(RuntimeLibrary PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    # Export only the functions marked RUNTIME_API
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Include directories
target_include_directories(RuntimeLibrary
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/deps/include
)

# Link libraries (on Windows the SDK's imported target, which also adds its include directory)
if (WIN32)
    target_link_libraries(RuntimeLibrary PRIVATE dxrt::dxrt)
else()
    target_link_libraries(RuntimeLibrary PRIVATE dxrt)
endif()
target_link_libraries(RuntimeLibrary PRIVATE Threads::Threads)

# Linker flags (basic flags, specific version compatibility should be handled by toolchain)
if (UNIX)
    set_property(TARGET RuntimeLibrary APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--as-needed -Wl,--hash-style=gnu")

    # libstdc++ template instantiations keep default visibility regardless of the presets above,
    # so a version script is what limits the dynamic symbol table to the OAAX API.
    set_property(TARGET RuntimeLibrary APPEND_STRING PROPERTY LINK_FLAGS
        " -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/runtime_core.map")
    set_property(TARGET RuntimeLibrary APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/runtime_core.map)
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
