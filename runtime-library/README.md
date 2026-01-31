# OAAX DEEPX Runtime Library

This directory contains the DeepX Runtime Library build system that creates `libRuntimeLibrary.so` (Linux) and `RuntimeLibrary.dll` (Windows) for loading and executing DXNN models on DeepX acceleration platforms.

---
## Overview

The Runtime Library is a C++ library built on top of DX-RT, providing core functionalities such as:
- Loading DXNN models
- Managing input/output tensors for inference
- Executing model inference on DeepX NPU

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

# Checkout a specific version (e.g., v3.0.0)
git checkout v3.0.0
```

Available versions can be found in the [DX-RT releases page](https://github.com/DEEPX-AI/dx_rt/releases).

Once cloned, follow the `DX-RT` repository's documentation to build and install DX-RT on your Ubuntu system so that it is available to CMake (e.g., via `find_package(dxrt)` as used by this runtime-library).

#### Windows DX-RT

For Windows builds, use the dedicated DX-RT Windows repository. 

```powershell1
cd runtime-library
git clone https://github.com/DEEPX-AI/dx_rt_windows.git
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
    -DxrtRoot "dx_rt_windows\m1\v.3.1.1\dx_rt" `
    -Configuration Release
```

- `-DxrtRoot`: Path to the DX-RT Windows SDK root directory (must contain `include/` and `lib/`).
- `-Configuration`: Optional build configuration (`Release` by default, `Debug` also supported).
- `-RuntimeVersion`: Optional OAAX runtime version. If omitted, the script reads the version from `dxrt_version.txt` in this directory.

 If PowerShell reports that running scripts is disabled on your system, you can temporarily relax the policy for the current session only:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

### Artifacts

The compiled runtime libraries are saved under the `artifacts/` directory.

- **Linux**: `libRuntimeLibrary.so`, structured by target architecture and Ubuntu version based on the build environment.
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