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
# Build the Docker image (DX-COM is pip-installed from sdk.deepx.ai during the build)
./conversion-toolchain/build-toolchain.sh          # DX-COM version override: ./build-toolchain.sh 2.4.0
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
- **Public C API** (OAAX 2.0, also defines the tensor types): `runtime-library/include/runtime_core.h`
- **Lifecycle**: `runtime_init()` → `runtime_load_models()` → `runtime_enqueue_input()` / `runtime_retrieve_output()` (loop) → `runtime_cleanup()`
- **Design**: callback-based (`InferenceEngine::RegisterCallback`), zero-copy input, lock-free output queue, bounded output buffer pool. A full pool returns `RUNTIME_STATUS_TIMEOUT` (backpressure — the caller keeps the input and retries/drops). Details in the header comment of `runtime_core.cpp`.
- **Logging**: spdlog (vendored under `runtime-library/deps/`, see `THIRD_PARTY_NOTICES.md`); log file is `./runtime.log` unless `DEEPX_RUNTIME_LOG_FILE` is set
- **Build standard**: C++11, CMake 3.14+
- **DX-RT version pin**: `runtime-library/dxrt_version.txt` (Linux source) and `runtime-library/dxrt_windows_commit.txt` (Windows SDK)
- **Unit tests** (no hardware): `runtime-library/test/` builds `runtime_core.cpp` against a mock DX-RT (`test/mock_dxrt/`) — keep the mock in sync with the DX-RT calls the runtime makes

### Conversion Toolchain internals

- Docker image based on `python:3.11-slim`. DX-COM (`dxcom` CLI) is pip-installed from the `sdk.deepx.ai` wheel at image build time (`DX_COM_VERSION` build arg, default 2.3.0).
- Input to conversion: a ZIP containing the ONNX model, a JSON config, and (optionally) a calibration dataset.
- Main conversion script executed inside the container: `conversion-toolchain/scripts/convert.sh`.

## Testing

```bash
# Stage 1 — convert models to DXNN (requires Docker + the toolchain image)
uv run python tests/stage1.py

# Stage 2 — run inference on compiled models (requires DeepX hardware)
uv run python tests/stage2.py
```

```bash
# Unit tests — no hardware or Docker needed (also run in CI)
pytest tests/test_models.py
cmake -S runtime-library/test -B build-unit-tests -DCMAKE_BUILD_TYPE=Debug && cmake --build build-unit-tests && ./build-unit-tests/test_runtime
```

- `DEEPX_TOOLCHAIN_IMAGE` — override the Docker image (default: `oaax-deepx-toolchain:latest`)
- `DEEPX_RUNNER_DIR` / `DEEPX_RUNTIME_LIB_DIR` — directories holding `inference_runner` / `libRuntimeLibrary.so`
- Python 3.10+ (`uv sync`). Details: `tests/README.md`

## CI/CD

GitHub Actions workflows in `.github/workflows/`:

| Workflow | Trigger | What it does |
|---|---|---|
| `build.yml` | push/PR to `main`, `v*` tag | `unit-tests` (mock-DX-RT runtime tests incl. ASan/UBSan, `test_models.py`); builds toolchain image; matrix runtime builds (Linux x86_64+aarch64 × Ubuntu 20/22/24; the Windows x86_64 job is disabled with `if: false` until Windows is supported); `integration-tests` is **opt-in** (repo variable `RUN_INTEGRATION_TESTS=true`; stage 2 also needs `RUN_STAGE2=true` and DeepX hardware on the runner); then `publish` (see below) |
| `lint.yml` | push/PR to `main` | Runs pre-commit hooks (ruff, clang-format, shellcheck, hadolint) |
| `delete-temporary-artifacts.yml` | PR closed | Removes the PR's `<branch>/` artifacts from S3 (never a release-looking name) |

The self-hosted `hetzner` jobs are skipped for PRs from forks. Build jobs pass their output to the tests and to `publish` as GitHub artifacts (the toolchain image is ~6 GB). `publish` runs only after the builds and tests passed and uploads those bytes to the S3-compatible bucket with `scripts/publish-artifacts.sh`:

| Event | Mode | Writes |
|---|---|---|
| `v*` tag (must equal `VERSION`, commit on `main`) | release | `<VERSION>/`, **write once** (refuses if any key exists), plus `latest/` |
| pull request with secrets | branch | `<branch>/`, overwritten on each run, never `latest/`; names that look like a release (`latest`, leading digit, `v<digit>`) are refused |
| push to `main` | dry run | nothing (files checked, keys printed) |

A push to `main` never changes the bucket.

## Version Management

The project version lives in `/VERSION`. Release: bump `VERSION` on `main`, then push the tag `vX.Y.Z` (it must match `VERSION`) on that commit. The CI reads `VERSION` for the release upload path; a published version is never overwritten, so bump it for every new release.
