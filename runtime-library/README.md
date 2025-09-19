# OAAX DEEPX Runtime Library

This directory contains the DeepX Runtime Library build system that creates `libRuntimeLibrary.so` for loading and executing DXNN models on DeepX acceleration platforms.

---
## Overview

The Runtime Library is a C++ library built on top of DX-RT, providing core functionalities such as:
- Loading DXNN models
- Managing input/output tensors for inference
- Executing model inference on DeepX NPU

The library is built using **Docker** containers to ensure consistent build environments across different host systems.

---
## Prerequisites

### System Requirements
- **Docker**: Must be installed and running
- **Supported Host Architectures**: x86_64, aarch64
- **Supported Ubuntu Versions**: 20.04, 22.04, 24.04

### Installing DX-RT

Simply clone the DX-RT repository from GitHub:

```bash
git clone https://github.com/DEEPX-AI/dx_rt.git
```

The repository contains all necessary source code and build scripts for the DeepX Runtime (DX-RT).

#### Using Specific Versions

If you need to use a specific version of DX-RT, you can checkout a particular tag after cloning:

```bash
# List available tags
git tag

# Checkout a specific version (e.g., v3.0.0)
git checkout v3.0.0
```

Available versions can be found in the [DX-RT releases page](https://github.com/DEEPX-AI/dx_rt/releases).

---
## Usage

### Basic Usage

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

### Artifacts

The compiled `libRuntimeLibrary.so` file is saved under the `artifacts/` directory, structured by target architecture and Ubuntu version based on the build environment.

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
└── aarch64-ubuntu24.04/
    └── libRuntimeLibrary.so
```