# OAAX DEEPX Runtime Library

This directory contains the DeepX Runtime Library build system that creates `libRuntimeLibrary.so` (Linux) and `RuntimeLibrary.dll` (Windows) for loading and executing DXNN models on DeepX acceleration platforms.

---
## Overview

The Runtime Library is a C++ library built on top of DX-RT, providing core functionalities such as:
- Loading DXNN models
- Managing input/output tensors for inference
- Executing model inference on DeepX NPU

It implements the **OAAX 2.0** runtime C API declared in [`include/runtime_core.h`](include/runtime_core.h).
See [Runtime behaviour](#runtime-behaviour) for the contract callers must follow.

On Linux, the library is built using **Docker** containers to ensure consistent build environments across different host systems. On Windows, it is built natively using **CMake** and **Visual Studio**.

---
## Prerequisites

### System Requirements
- **Linux builds (Docker-based)**
    - **Docker**: Must be installed and running
    - **Supported Host Architectures**: x86_64, aarch64
    - **Supported Ubuntu Versions**: 20.04, 22.04, 24.04

- **Windows builds (native)**
    - **Host OS**: Windows 10/11 x86_64
    - **CMake** and **Visual Studio 2022** with C++ workload
    - **DX-RT Windows SDK** (prebuilt, containing `include/` and `lib/`)
    - **PowerShell** to run the build script

### Installing DX-RT

The DX-RT versions this runtime is built and tested against are pinned in this directory:
`dxrt_version.txt` (Linux, currently `3.3.2`) and `dxrt_windows_commit.txt` (Windows SDK commit).
CI downloads exactly those; use the same versions locally.

#### Linux (Ubuntu) DX-RT

Simply clone the DX-RT repository from GitHub:

```bash
cd runtime-library
git clone https://github.com/DEEPX-AI/dx_rt.git
```

The repository contains all necessary source code and build scripts for the DeepX Runtime (DX-RT).

##### Using Specific Versions (Linux)

If you need to use a specific version of DX-RT, you can checkout a particular tag after cloning:

```bash
# List available tags
git tag

# Checkout the pinned version (see dxrt_version.txt, e.g. v3.3.2)
git checkout "v$(cat ../dxrt_version.txt)"
```

Available versions can be found in the [DX-RT releases page](https://github.com/DEEPX-AI/dx_rt/releases).

Once cloned, follow the `DX-RT` repository's documentation to build and install DX-RT on your Ubuntu system so that it is available to CMake (e.g., via `find_package(dxrt)` as used by this runtime-library).

#### Windows DX-RT

For Windows builds, use the dedicated DX-RT Windows repository. 

```powershell
cd runtime-library
git clone https://github.com/DEEPX-AI/dx_rt_windows.git
git -C dx_rt_windows checkout "$(Get-Content dxrt_windows_commit.txt)"
```

This repository provides prebuilt Windows runtime libraries, drivers, and SDK files for DeepX NPUs.

For installation and usage details, refer to the `DX-RT Windows` repository's documentation.

---
## Usage

### Linux builds (Docker)

#### Build for the host architecture
Builds a runtime library for the same architecture as your host machine. Works on both x86_64 and aarch64 hosts.
```bash
# Example: build for Ubuntu 22.04 matching the host architecture
./build-runtimes.sh --ubuntu_version 22.04
```

#### Cross-compile aarch64 on an x86_64 host
Builds an aarch64 runtime library on an x86_64 host. Not supported in the reverse direction (aarch64 host → x86_64 target).
```bash
# Example: cross-compile aarch64 for Ubuntu 22.04 (x86_64 host only)
./build-runtimes.sh --ubuntu_version 22.04 --cross-aarch64
```

### Windows builds (native)

Use the PowerShell build script to generate `RuntimeLibrary.dll` using the prebuilt DX-RT Windows SDK (run this from a PowerShell prompt):

```powershell
./build-windows.ps1 -DxrtRoot "C:\path\to\DX_RT_SDK" -Configuration Release
```

When dx_rt_windows is cloned under `runtime-library` as shown above, a concrete example is:

```powershell
./build-windows.ps1 `
    -DxrtRoot "dx_rt_windows\m1\<version>\dx_rt" `
    -Configuration Release
```

- `-DxrtRoot`: Path to the DX-RT Windows SDK root directory (must contain `include/` and `lib/`).
- `-Configuration`: Optional build configuration (`Release` by default, `Debug` also supported).
- `-RuntimeVersion`: Optional OAAX runtime version. If omitted, the script reads the repository `VERSION` file (`../VERSION`).

 If PowerShell reports that running scripts is disabled on your system, you can temporarily relax the policy for the current session only:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

### Artifacts

The compiled runtime libraries are saved under the `artifacts/` directory.

- **Linux**: `libRuntimeLibrary.so` plus `library-dxrt-<dxrt version>.tar.gz` (the archive CI uploads), structured by target architecture and Ubuntu version based on the build environment.
- **Windows**: `RuntimeLibrary.dll`, structured by target architecture and OS.

#### Output Structure
```
artifacts/
├── x86_64-ubuntu20.04/
│   └── libRuntimeLibrary.so
├── x86_64-ubuntu22.04/
│   └── libRuntimeLibrary.so
├── x86_64-ubuntu24.04/
│   └── libRuntimeLibrary.so
├── aarch64-ubuntu20.04/
│   └── libRuntimeLibrary.so
├── aarch64-ubuntu22.04/
│   └── libRuntimeLibrary.so
├── aarch64-ubuntu24.04/
│   └── libRuntimeLibrary.so
└── x86_64-windows/
    └── RuntimeLibrary.dll
```

---
## Runtime behaviour

Lifecycle: `runtime_init` → `runtime_load_models` → loop of `runtime_enqueue_input` / `runtime_retrieve_output` → `runtime_cleanup`.

- **Single model** per process for now; `runtime_load_models` rejects `num_models != 1`.
- **Input ownership**: on `RUNTIME_STATUS_SUCCESS` the runtime owns the `Tensors` passed to `runtime_enqueue_input` and frees it itself (input is passed to DX-RT zero-copy). On *any other* return the caller keeps ownership and must free it.
- **Output ownership**: the `Tensors` returned by `runtime_retrieve_output` is owned by the caller.
- **Input validation**: the first tensor's `data_size` must equal the model's input size, otherwise `RUNTIME_STATUS_TENSOR_SHAPE_MISMATCH`.
- **Backpressure**: the output buffer pool is bounded by DX-RT's in-flight limit × device count. When exhausted, `runtime_enqueue_input` returns `RUNTIME_STATUS_TIMEOUT` immediately — retry later or drop the frame.
- **Results are unordered**: match them to requests via the echoed `Tensors::id`.
- **Errors**: `runtime_get_error()` returns the last error text (a per-thread copy, valid until that thread's next call).
- **Logging**: written to stderr and to `./runtime.log`; set `DEEPX_RUNTIME_LOG_FILE` to change the file and `SPDLOG_LEVEL` for verbosity.
- **DX-RT dynamic CPU threads**: `DXRT_DYNAMIC_CPU_THREAD=ON` is set by `runtime_init` unless the variable is already set (so `OFF` is respected).

---
## Unit tests

`test/` compiles `src/runtime_core.cpp` against a mock of the DX-RT API (`test/mock_dxrt/`), so no DeepX hardware, DX-RT install or Docker is needed:

```bash
cmake -S test -B /tmp/rt-tests -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/rt-tests -j && (cd /tmp/rt-tests && ./test_runtime)
```

CI also runs them under ASan/UBSan. If you change which DX-RT calls `runtime_core.cpp` makes, update the mock accordingly. Hardware validation is done with the test suite in [`../tests`](../tests/README.md).
