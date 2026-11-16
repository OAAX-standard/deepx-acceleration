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
    - **Visual Studio 2022** with the *Desktop development with C++* workload (it includes CMake and vcpkg)
    - **DX-RT**: the DX-SDK Windows Installer, or the public `dx_rt` source (see [Windows DX-RT](#windows-dx-rt))
    - **PowerShell** to run the build script

### Installing DX-RT

The DX-RT version this runtime is built and tested against is pinned in `dxrt_version.txt`
(currently `3.5.0`), for Linux and Windows. CI builds exactly that tag of the public
[`dx_rt`](https://github.com/DEEPX-AI/dx_rt) source; use the same version locally.

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

# Checkout the pinned version (see dxrt_version.txt, e.g. v3.5.0)
git checkout "v$(cat ../dxrt_version.txt)"
```

Available versions can be found in the [DX-RT releases page](https://github.com/DEEPX-AI/dx_rt/releases).

Once cloned, follow the `DX-RT` repository's documentation to build and install DX-RT on your Ubuntu system so that it is available to CMake (e.g., via `find_package(dxrt)` as used by this runtime-library).

#### Windows DX-RT

Running the runtime on Windows needs the **DX-SDK Windows Installer 1.3.2 or later** (DX-RT 3.5.0,
Windows driver 2.0.1), from [developer.deepx.ai/sw-download](https://developer.deepx.ai/sw-download)
(login required). Install at least the *Driver*, *Runtime* and *C/C++ SDK* components (the
`Standard` profile). It installs to `C:\Program Files\DEEPX\M1_M1M_Series\DX-RUNTIME\<version>`,
sets `DEEPX_M1_DIR` to that directory and puts its `bin` (with `dxrt.dll`) on `PATH`. DX-RT 3.5.0
also needs DX-M1 firmware 2.7.6 or later: check with `dxrt-cli -s` and update with
`dxrt-cli -u "$env:DEEPX_M1_DIR\fw\m1\fw.bin"` if needed.

To build, the installed SDK is all you need. Alternatively, build against the public `dx_rt` source,
as CI does (the build script builds and installs it under `dx_rt\out\install\` first):

```powershell
cd runtime-library
git clone --branch "v$(Get-Content dxrt_version.txt)" https://github.com/DEEPX-AI/dx_rt.git
```

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

Use the PowerShell build script (run it from a PowerShell prompt) against the installed DX-SDK:

```powershell
./build-windows.ps1 -DxrtRoot $env:DEEPX_M1_DIR
```

or, without `-DxrtRoot`, against the public `dx_rt` source cloned into `runtime-library\dx_rt`:

```powershell
./build-windows.ps1
```

- `-DxrtRoot`: Optional DX-RT SDK directory, containing `include\dxrt\`, `lib\dxrt.lib` and `lib\cmake\dxrt\` (the DX-SDK Windows Installer's install directory). If omitted, `dx_rt` is built from source with its `x64-Release` CMake preset (needs vcpkg: `VCPKG_ROOT`, set by the Visual Studio developer environment).
- `-Configuration`: Optional build configuration (`Release` by default, `Debug` also supported).
- `-RuntimeVersion`: Optional OAAX runtime version. If omitted, the script reads the repository `VERSION` file (`../VERSION`).

 If PowerShell reports that running scripts is disabled on your system, you can temporarily relax the policy for the current session only:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

### Artifacts

The compiled runtime libraries are saved under the `artifacts/` directory.

- **Linux**: `libRuntimeLibrary.so` plus `library-dxrt-<dxrt version>.tar.gz` (the archive CI uploads), structured by target architecture and Ubuntu version based on the build environment.
- **Windows**: `RuntimeLibrary.dll`, its import library `RuntimeLibrary.lib`, and `library-dxrt-<dxrt version>.tar.gz` (the archive CI uploads), which holds `RuntimeLibrary.dll` only. `dxrt.dll` is not redistributed: the DLL loads the one the DX-SDK Windows Installer put on `PATH`.

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
    ├── RuntimeLibrary.dll
    └── RuntimeLibrary.lib
```

---
## Runtime behaviour

Lifecycle: `runtime_init` → `runtime_load_models` → loop of `runtime_enqueue_input` / `runtime_retrieve_output` → `runtime_cleanup`.

- **Loading models**: `runtime_load_models` loads all `num_models` models in one call, or none. Model `i` of the array gets model id `i`; an error message starts with `model[<i>]: `. Models can be loaded once per `runtime_init`: a second call returns `RUNTIME_STATUS_ALREADY_INITIALIZED`. Loading and `runtime_cleanup` are serialized.
- **Model source**: if `model_data` is set the model is read from memory (`model_size` must be non-zero), otherwise from `file_path`; neither set gives `RUNTIME_STATUS_INVALID_ARGUMENT`. When both are set, `model_data` is used. The caller may free `model_data` as soon as the call returns (DX-RT copies the model; the runtime keeps no reference).
- **Input ownership**: on `RUNTIME_STATUS_SUCCESS` the runtime owns the `Tensors` passed to `runtime_enqueue_input` and frees it itself (input is passed to DX-RT zero-copy). On *any other* return the caller keeps ownership and must free it.
- **Output ownership**: the `Tensors` returned by `runtime_retrieve_output` is owned by the caller.
- **Input validation**: the first tensor's `data_size` must equal the model's input size, otherwise `RUNTIME_STATUS_TENSOR_SHAPE_MISMATCH`.
- **Queueing and backpressure**: each model has its own output buffer pool, grown lazily up to DX-RT's in-flight limit (`dxrt_get_task_max_load()`) × device count buffers of that model's output size, so worst-case memory grows with the number of models. A request accepted while its model's pool is full waits in that model's pending queue (the input is kept, not copied) and is sent to the device, oldest first, as soon as `runtime_retrieve_output` frees a buffer. The queue holds up to `max_pending_requests` requests per model (`runtime_init` config key, non-negative integer, default 256; `0` disables queueing; any other value gives `RUNTIME_STATUS_INVALID_ARGUMENT`). When pool and queue are both full, `runtime_enqueue_input` for that model returns `RUNTIME_STATUS_TIMEOUT` immediately — retry later or drop the frame; other models are not affected. Queued requests trade latency for completeness: a request waits behind those already queued, so keeping more requests outstanding than `pool_capacity` (see `runtime_get_info()`) adds queueing time to each one. Real-time callers that prefer dropping frames should set `max_pending_requests` low or to `0`.
- **Rejected or failed requests**: if DX-RT rejects a request sent right away, `runtime_enqueue_input` returns `RUNTIME_STATUS_INFERENCE_ERROR` (the caller keeps the input). A queued request was already accepted, so its rejection is reported by `runtime_retrieve_output` instead: `RUNTIME_STATUS_INFERENCE_ERROR`, with the model id set and the request id in that thread's `runtime_get_error()`. A request that fails on the device is reported the same way.
- **Results are unordered**: outputs of all models share one queue. `runtime_retrieve_output` returns the model id with each output; match outputs to requests via the echoed `Tensors::id`.
- **Info**: in `runtime_get_info()`, `requests_in_flight` counts requests accepted by `runtime_enqueue_input` and not yet retrieved, including the `pending_requests` still waiting in a model's queue.
- **Errors**: `runtime_get_error()` returns the calling thread's last error (like `errno`: other threads' failures do not replace it), valid until that thread's next failing call. A successful `runtime_init`, and `runtime_cleanup` of an initialized runtime, clear every thread's error.
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
