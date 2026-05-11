# DeepX Acceleration — Test Suite

This directory contains a two-stage test suite for the DeepX OAAX runtime.

- **Stage 1** converts ONNX models to DXNN format using the DeepX toolchain Docker image and verifies the outputs with pytest.
- **Stage 2** loads compiled DXNN models through the OAAX runtime library and measures inference latency and throughput.

Stage 1 must be run first; its output (`tests/compiled_models/`) is consumed by Stage 2.

---

## Requirements

### Stage 1

| Requirement | Notes |
|---|---|
| Docker | Must be running and accessible to the current user |
| DeepX toolchain image | Default: `deepx-conversion-toolchain:22.04`. Override with `DEEPX_TOOLCHAIN_IMAGE=<image>` |
| Python 3.8+ | |
| pytest | `pip install pytest` |
| ultralytics | Optional — only needed for YOLO model export: `pip install ultralytics` |

### Stage 2

| Requirement | Notes |
|---|---|
| `libRuntimeLibrary.so` | Built from `runtime-library/` (see below) |
| CMake 3.14+ | To build the `inference_runner` C++ binary |
| C++11 compiler | gcc or clang |
| Python 3.8+ | |
| DeepX NPU hardware | DX-M1 or DX-H1 — stage 2 performs real inference |

---

## Running Stage 1 (conversion)

### Convert classification models (SqueezeNet, ResNet18, MobileNetV2)

```bash
python3 tests/stage1.py
```

### Also convert YOLO models (yolov8n, yolo11n, yolo11s)

```bash
python3 tests/stage1.py --yolo
```

Requires `ultralytics`. YOLO `.pt` weights are downloaded automatically on first run.

### Run a single model via pytest

```bash
DEEPX_TOOLCHAIN_IMAGE=deepx-conversion-toolchain:22.04 \
  pytest tests/test_conversion.py -k "squeezenet" -v
```

### Using a different toolchain image

```bash
DEEPX_TOOLCHAIN_IMAGE=oaax-deepx-toolchain:latest python3 tests/stage1.py
```

> **Note:** `oaax-deepx-toolchain` requires the DX-COM compiler to be mounted at `/app/dx_com` inside the container. `deepx-conversion-toolchain:22.04` has it built in.

### Output

Compiled models are written to `tests/compiled_models/`:

```
tests/compiled_models/
├── onnx/               # Downloaded / exported ONNX files
├── squeezenet/
│   ├── squeezenet.dxnn
│   └── convert.log
├── resnet18/
│   └── ...
└── yolo11n/
    └── ...
```

Conversion is cached — re-running stage 1 skips models whose `.dxnn` already exists.

---

## Running Stage 2 (inference)

### Prerequisites: build the runtime library

The runtime library must be built before stage 2 can run. On a machine with `dx_rt` available:

```bash
cd runtime-library
./build-runtimes.sh --ubuntu_version 22.04
```

This produces `runtime-library/artifacts/x86_64-ubuntu22.04/libRuntimeLibrary.so`. The stage 2 build script discovers it automatically.

If the library lives elsewhere, set:

```bash
export DEEPX_RUNTIME_LIB_DIR=/path/to/dir/containing/libRuntimeLibrary.so
```

### Run stage 2

```bash
./tests/run_stage2.sh
```

This builds `inference_runner` on first run, then executes it against every `.dxnn` in `tests/compiled_models/`.

### Configuration via environment variables

| Variable | Default | Description |
|---|---|---|
| `MODELS` | all | Space-separated list of model names to run |
| `RUNS` | 100 | Number of timed inference runs per model |
| `WARMUP` | 10 | Number of warmup runs (excluded from metrics) |
| `CSV` | — | Path to append results in CSV format |
| `DEEPX_RUNTIME_LIB_DIR` | auto-detected | Directory containing `libRuntimeLibrary.so` |

Example:

```bash
MODELS="yolo11n yolo11s" RUNS=200 CSV=results.csv ./tests/run_stage2.sh
```

### Output

```
Model                 Runs   Avg ms   P50 ms   P95 ms   Min ms   Max ms   Throughput
--------------------------------------------------------------------------------
squeezenet             100     3.21     3.18     4.02     2.91     5.14      311.5
resnet18               100     4.87     4.83     5.91     4.41     7.02      205.3
yolo11n                100    12.44    12.37    14.91    11.23    16.08       80.4
```

---

## Packaging Stage 2 for a remote machine

To run stage 2 on a machine that does not have the full repository:

```bash
./tests/package_stage2.sh
```

This creates `deepx-stage2-<version>.tar.gz` containing:
- `inference_runner` binary
- `libRuntimeLibrary.so`
- All compiled `.dxnn` models
- `stage2.py` and `run_stage2.sh`

Transfer the archive to the target machine and run:

```bash
tar -xzf deepx-stage2-<version>.tar.gz
cd deepx-stage2-<version>
./run_stage2.sh
```

---

## Adding new models

### Classification / regression models (direct ONNX download)

Add an entry to `TEST_MODELS` in `tests/models.py`:

```python
"my_model": {
    "url": "https://example.com/my_model.onnx",
    "filename": "my_model.onnx",
    "input_name": "input",        # tensor name as seen by dx_com
    "input_shape": [1, 3, 224, 224],
    "input_dtype": "float32",
    "task": "image_classification",
},
```

The `input_name` must match the model's actual ONNX input tensor name. If unsure, inspect with:

```python
import onnx
m = onnx.load("my_model.onnx")
for inp in m.graph.input:
    print(inp.name, [d.dim_value for d in inp.type.tensor_type.shape.dim])
```

Then add the model name to the `compiled_classification_models` fixture in `tests/conftest.py` and re-run stage 1.

### YOLO models (ultralytics export)

Add an entry to `TEST_MODELS` using `pt_name` instead of `url`:

```python
"yolo11m": {
    "pt_name": "yolo11m.pt",
    "filename": "yolo11m.onnx",
    "input_name": "images",
    "input_shape": [1, 3, 640, 640],
    "input_dtype": "float32",
    "task": "object_detection",
},
```

Then add the model name to the `compiled_yolo_models` fixture in `tests/conftest.py` and re-run:

```bash
python3 tests/stage1.py --yolo
```

---

## Known limitations

- **yolov8n** fails to compile with a `CODEGEN` error from DX-COM — the model uses operators not supported by the current hardware/compiler version. It is automatically skipped by the test suite.
- **Stage 2 without hardware**: if `libRuntimeLibrary.so` is built against the mock `dxrt` (as in CI), inference runs but latency numbers reflect a simulated 5–15ms delay rather than real NPU performance.
