# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**deepx-acceleration** is an OAAX-compliant implementation for DeepX NPU acceleration (DX-M1, DX-H1 chips). It has two independently buildable components:

1. **Conversion Toolchain** (`conversion-toolchain/`): Docker-based pipeline to compile ONNX models into DXNN format using the proprietary DeepX Compiler (DX-COM).
2. **Runtime Library** (`runtime-library/`): C++ shared library (`libRuntimeLibrary.so` / `RuntimeLibrary.dll`) that loads and runs DXNN models on DeepX hardware via the DX-RT API.

## Build Commands

### Runtime Library — Linux

```bash
# Build for host architecture (inside Docker)
./runtime-library/build-runtimes.sh --ubuntu_version 22.04

# Cross-compile aarch64 on x86_64 host
./runtime-library/build-runtimes.sh --ubuntu_version 22.04 --cross-aarch64
```

Supported `--ubuntu_version` values: `20.04`, `22.04`, `24.04`.  
Output: `runtime-library/artifacts/{arch}-ubuntu{version}/`.

### Runtime Library — Windows

```powershell
./runtime-library/build-windows.ps1 -DxrtRoot "path/to/dx_rt_sdk" -Configuration Release
```

Requires Visual Studio 2022. Output: `runtime-library/artifacts/x86_64-windows/RuntimeLibrary.dll`.

### Conversion Toolchain

```bash
# 1. Obtain DX-COM (requires DeepX Developer Portal credentials)
./conversion-toolchain/setup-dx_com.sh

# 2. Build the Docker image
./conversion-toolchain/build-toolchain.sh
```

Output: `conversion-toolchain/artifacts/oaax-deepx-toolchain.tar`.

## Architecture

```
ONNX Model
    ↓
[Conversion Toolchain]  ← Docker + DX-COM compiler
    ↓
DXNN Model
    ↓
[Runtime Library]  ← C++ / DX-RT API
    ↓
DeepX NPU (DX-M1 / DX-H1)
```

### Runtime Library internals

- **Entry point**: `runtime-library/src/runtime_core.cpp`
- **Public C API**: `runtime-library/include/runtime_core.h`
- **Lifecycle**: `runtime_initialization()` → `runtime_model_loading()` → `send_input()` / `receive_output()` (loop) → `runtime_destruction()`
- **Tensor types**: defined in `runtime-library/deps/include/tensors_struct.h`
- **Logging**: spdlog (header-only, vendored under `runtime-library/deps/`)
- **Build standard**: C++11, CMake 3.14+
- **DX-RT version pin**: `runtime-library/dxrt_version.txt` (Linux source) and `runtime-library/dxrt_windows_commit.txt` (Windows SDK)

### Conversion Toolchain internals

- Runs inside a Ubuntu 22.04 Docker container. DX-COM is **not** baked into the image — it is mounted at runtime from `conversion-toolchain/dx_com/` (populated by `setup-dx_com.sh`).
- Input to conversion: a ZIP containing the ONNX model, a JSON config, and (optionally) a calibration dataset.
- Main conversion script executed inside the container: `conversion-toolchain/scripts/convert.sh`.

## Testing

```bash
# Stage 1 — convert all models to DXNN (requires Docker + dx_com + ultralytics)
uv run python tests/stage1.py

# Stage 1 without YOLO models
uv run python tests/stage1.py --no-yolo

# Stage 2 — run inference on compiled models (requires DeepX hardware)
uv run python tests/stage2.py
```

- `DEEPX_TOOLCHAIN_IMAGE` — override the Docker image (default: `oaax-deepx-toolchain:latest`)
- `DX_COM_PATH` — override the dx_com directory (default: `conversion-toolchain/dx_com`)

## CI/CD

GitHub Actions workflows in `.github/workflows/`:

| Workflow | Trigger | What it does |
|---|---|---|
| `build.yml` | push/PR to `main` | Builds toolchain image, matrix runtime builds (Linux x86_64+aarch64 × Ubuntu 20/22/24, Windows x86_64), integration tests |
| `lint.yml` | push/PR to `main` | Runs pre-commit hooks (ruff, clang-format, shellcheck, hadolint) |
| `delete-temporary-artifacts.yml` | PR closed | Cleans up branch-scoped S3 artifacts |

Artifacts are uploaded to an S3-compatible bucket. On `main` the version comes from the `VERSION` file; on feature branches the branch name is used as the version.

## Version Management

The project version lives in `/VERSION`. Update this file when cutting a release. The CI reads it to determine the S3 upload path and artifact naming.
