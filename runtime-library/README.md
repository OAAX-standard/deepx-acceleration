# OAAX DEEPX Runtime Library

This directory contains the DeepX Runtime Library build system that creates `libRuntimeLibrary.so` for loading and executing DXNN models on DeepX acceleration platforms.

---
## Overview

The Runtime Library is a C++ library that provides the core functionality for:
- Loading DXNN models (converted from ONNX using the DeepX compiler)
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

Since the DX-RT GitHub repository (`https://github.com/DEEPX-AI/dx_rt.git`) is not yet public, you need to download it using the [`download_dxrt.sh`](download_dxrt.sh) script.

#### Prerequisites
- [DeepX Developer Portal](https://developer.deepx.ai/) credentials (same as DX-COM installation)
- Python 3.8+ with `requests` and `beautifulsoup4` packages

#### Installation
```bash
# Download and install DX-RT (default version: 2.9.5)
DX_USERNAME="your_username" DX_PASSWORD="your_password" ./download_dxrt.sh
```

> **Future Update**: When the DX-RT GitHub repository becomes public, the build system will be updated to use `git clone https://github.com/DEEPX-AI/dx_rt.git` for downloading the DX-RT source code.

---
## Usage

### Basic Usage

#### Native Build (Host Architecture)
```bash
# Build for Ubuntu 22.04 on the host architecture (x86_64 or aarch64)
./build-runtimes.sh --os 22.04
```

#### Cross-Compilation (x86_64 → aarch64)
```bash
# Cross-compile for aarch64 Ubuntu 22.04 (x86_64 host only)
./build-runtimes.sh --os 22.04 --cross-aarch64
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