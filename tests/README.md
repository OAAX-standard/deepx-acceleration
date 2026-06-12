# DeepX Acceleration — Test Suite

A two-stage test pipeline for the DeepX OAAX runtime:

- **Stage 1** converts ONNX models to DXNN format via the DeepX toolchain Docker image.
- **Stage 2** loads compiled DXNN models through the OAAX runtime library on real DeepX hardware and measures inference latency and throughput.

Stage 1 must run before Stage 2 — its output (`tests/compiled_models/`) is the input to Stage 2.

---

## Quick start

```bash
# 1. Convert models (requires Docker + toolchain image)
uv run python tests/stage1.py

# 2. Run inference (requires DeepX hardware + built runtime library)
./tests/run_stage2.sh

# 3. Unit tests — no hardware or Docker needed
pytest tests/test_models.py -v
```

---

## Models

The suite tests three image classification models from the DeepX Model Zoo:

| Name | ONNX source | Input shape |
|---|---|---|
| MobileNetV1 | `sdk.deepx.ai/modelzoo/onnx/MobileNetV1-1.onnx` | `[1,224,224,3]` |
| MobileNetV2 | `sdk.deepx.ai/modelzoo/onnx/MobileNetV2-1.onnx` | `[1,224,224,3]` |
| SqueezeNet1_1 | `sdk.deepx.ai/modelzoo/onnx/SqueezeNet1_1-3.onnx` | `[1,224,224,3]` |

All input shapes are NHWC; all compiled models take `uint8` input.

---

## Requirements

### Stage 1 (conversion)

| Requirement | Notes |
|---|---|
| Docker | Must be running and accessible to the current user |
| DeepX toolchain image | Default: `oaax-deepx-toolchain:latest` — build with `conversion-toolchain/build-toolchain.sh` |
| Python 3.8+ with `pytest` | `pip install pytest` or use `uv` |

### Stage 2 (inference)

| Requirement | Notes |
|---|---|
| DeepX NPU hardware | DX-M1 or DX-H1 |
| `libRuntimeLibrary.so` | Built from `runtime-library/` (see below) |
| CMake 3.14+ and a C++11 compiler | To build `inference_runner` |
| Python 3.8+ | |

---

## Stage 1 — Conversion

```bash
uv run python tests/stage1.py
```

Extra arguments are forwarded to pytest:

```bash
# Convert only mobilenet models
uv run python tests/stage1.py -k mobilenet -v

# Or invoke pytest directly
pytest tests/test_conversion.py -v
```

### Using a different toolchain image

```bash
DEEPX_TOOLCHAIN_IMAGE=my-custom-image:latest uv run python tests/stage1.py
```

### Output

```
tests/compiled_models/
├── onnx/                        # Downloaded ONNX files and JSON configs
├── calibration_dataset/         # Calibration images (downloaded once)
├── mobilenetv1/
│   ├── mobilenetv1.dxnn
│   └── convert.log
├── mobilenetv2/  ...
└── squeezenet1_1/  ...
```

Conversion is cached — re-running skips models whose `.dxnn` already exists.

---

## Stage 2 — Inference

### 1. Build the runtime library

```bash
cd runtime-library
./build-runtimes.sh --ubuntu_version 22.04
```

This produces `runtime-library/artifacts/x86_64-ubuntu22.04/libRuntimeLibrary.so`.

### 2. Build `inference_runner`

`run_stage2.sh` does this automatically on first run. To build manually:

```bash
bash tests/runtime/build-tests.sh
```

To target a specific Ubuntu version:

```bash
bash tests/runtime/build-tests.sh \
    --runtime-lib-dir runtime-library/artifacts/x86_64-ubuntu24.04 \
    --build-dir tests/runtime/build-x86_64
```

### 3. Run

```bash
./tests/run_stage2.sh
```

### Configuration

| Variable | Default | Description |
|---|---|---|
| `MODELS` | all | Space-separated model names to run |
| `RUNS` | 100 | Timed inference runs per model |
| `WARMUP` | 10 | Warmup runs (excluded from metrics) |
| `CSV` | — | Path to append results in CSV format |
| `DEEPX_RUNTIME_LIB_DIR` | auto-detected | Directory containing `libRuntimeLibrary.so` |
| `DEEPX_RUNNER_DIR` | auto-detected | Directory containing `inference_runner` |

Example:

```bash
MODELS="mobilenetv1 mobilenetv2" RUNS=200 CSV=results.csv ./tests/run_stage2.sh
```

### Run via pytest

```bash
pytest tests/test_inference.py -v
```

Tests are skipped automatically if `inference_runner` is not built or no compiled models are found.

---

## Unit tests (no hardware or Docker)

`test_models.py` validates the model metadata in `models.py` — no external dependencies:

```bash
pytest tests/test_models.py -v
```

These checks run in CI on every push and are a good sanity check after adding a new model.

---

## Packaging Stage 2 for a remote machine

```bash
bash tests/package_stage2.sh
```

Builds `libRuntimeLibrary.so` and `inference_runner` for **x86_64 and aarch64**, then bundles everything with the compiled DXNN models into a single archive. Already-built artifacts are reused automatically.

| Flag | Default | Description |
|---|---|---|
| `--ubuntu-version <ver>` | `20.04` | Ubuntu version (20.04 gives widest glibc compatibility) |
| `--arch <arch>` | both | `x86_64` or `aarch64` (repeatable) |
| `--output <path>` | `deepx-stage2-<version>.tar.gz` | Output archive path |

### On the target machine

```bash
tar -xzf deepx-stage2-<version>.tar.gz
cd deepx-stage2-<version>
./run_stage2.sh
```

---

## Adding a new model

1. Add an entry to `TEST_MODELS` in `tests/models.py`:

```python
"my_model": {
    "url": "https://sdk.deepx.ai/modelzoo/onnx/MyModel-1.onnx",
    "json_url": "https://sdk.deepx.ai/modelzoo/json/MyModel-1.json",
    "filename": "my_model.onnx",
    "input_name": "input",
    "input_shape": [1, 224, 224, 3],   # NHWC
    "input_dtype": "uint8",
},
```

2. Re-run stage 1 — the new model is picked up automatically by all tests.

---

## Test file reference

| File | What it tests | Requires |
|---|---|---|
| `test_models.py` | `models.py` metadata (keys, shapes, dtypes, sizes) | Nothing |
| `test_conversion.py` | Stage 1 — DXNN files produced and convert.log shows success | Docker + toolchain image |
| `test_inference.py` | Stage 2 — inference_runner exits 0, metrics are valid | DeepX hardware + built runtime |
