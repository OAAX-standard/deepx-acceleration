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
        CACHE PATH "Directory containing libRuntimeLibrary.so (Windows: RuntimeLibrary.dll and .lib)")
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
        x86_64-windows
    REQUIRED
)

message(STATUS "Using runtime library: ${RUNTIME_LIB}")
message(STATUS "Using runtime headers: ${RUNTIME_INCLUDE_DIR}")

find_package(Threads REQUIRED)

add_executable(inference_runner inference_runner.cpp)

target_include_directories(inference_runner PRIVATE ${RUNTIME_INCLUDE_DIR})
target_link_libraries(inference_runner PRIVATE ${RUNTIME_LIB} Threads::Threads)

if(WIN32)
    # RUNTIME_LIB is the import library; the DLL next to it is copied next to the binary,
    # where Windows looks first.  dxrt.dll comes from PATH (DX-SDK Windows Installer).
    target_compile_options(inference_runner PRIVATE /utf-8)
    target_compile_definitions(inference_runner PRIVATE _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE)
    get_filename_component(RUNTIME_LIB_FOUND_DIR "${RUNTIME_LIB}" DIRECTORY)
    add_custom_command(TARGET inference_runner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${RUNTIME_LIB_FOUND_DIR}/RuntimeLibrary.dll" "$<TARGET_FILE_DIR:inference_runner>")
else()
    # libRuntimeLibrary.so depends on libdxrt which is only present on target
    # hardware — allow unresolved transitive symbols at link time.
    target_link_options(inference_runner PRIVATE "-Wl,--allow-shlib-undefined")

    # Use $ORIGIN so the binary finds libRuntimeLibrary.so from its own directory
    set_target_properties(inference_runner PROPERTIES
        BUILD_RPATH  "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endif()
