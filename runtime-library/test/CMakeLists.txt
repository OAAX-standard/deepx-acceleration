cmake_minimum_required(VERSION 3.14)
project(dx_oaax_test DESCRIPTION "OAAX 2.0 Runtime Test with Mock dxrt")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Version define
add_definitions(-DOAAX_RUNTIME_VERSION="2.0.0-test")

# Silence MSVC warnings for test code
if (MSVC)
    add_compile_options(/utf-8)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE)
endif()

# Include directories:
# - mock_dxrt/ overrides real <dxrt/dxrt_api.h>
# - spdlog from deps
# - interface_2.h and runtime_core.h from include/
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/mock_dxrt
    ${CMAKE_CURRENT_SOURCE_DIR}/../deps/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Build the runtime source with mock dxrt
add_executable(test_runtime
    test_runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/runtime_core.cpp
)

# Define RUNTIME_LIBRARY_EXPORTS so RUNTIME_API resolves to dllexport (not dllimport)
target_compile_definitions(test_runtime PRIVATE RUNTIME_LIBRARY_EXPORTS)

# Threads required
find_package(Threads REQUIRED)
target_link_libraries(test_runtime PRIVATE Threads::Threads)
