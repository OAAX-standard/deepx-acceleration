"""
Model definitions and download helpers for the DeepX test suite.

All models are sourced from the DeepX Model Zoo (https://developer.deepx.ai/modelzoo/).

Each entry in TEST_MODELS provides:
  - url:          direct ONNX download URL from the DeepX model zoo
  - filename:     name used when saving the ONNX file locally
  - input_name:   name of the model's input tensor (needed for dx_com config.json)
  - input_shape:  [N, H, W, C] shape (NHWC — the compiled model's runtime format)
  - input_dtype:  element type of the compiled model's input (always uint8 after dx_com)
"""

import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

CALIBRATION_DATASET_URL = "https://sdk.deepx.ai/dataset/calibration_dataset.tar.gz"

TEST_MODELS = {
    # --- Classification ---
    "mobilenetv1": {
        "url": "https://sdk.deepx.ai/modelzoo/onnx/MobileNetV1-1.onnx",
        "json_url": "https://sdk.deepx.ai/modelzoo/json/MobileNetV1-1.json",
        "filename": "mobilenetv1.onnx",
        "input_name": "input.1",
        "input_shape": [1, 224, 224, 3],
        "input_dtype": "uint8",
    },
    "mobilenetv2": {
        "url": "https://sdk.deepx.ai/modelzoo/onnx/MobileNetV2-1.onnx",
        "json_url": "https://sdk.deepx.ai/modelzoo/json/MobileNetV2-1.json",
        "filename": "mobilenetv2.onnx",
        "input_name": "input.1",
        "input_shape": [1, 224, 224, 3],
        "input_dtype": "uint8",
    },
    "squeezenet1_1": {
        "url": "https://sdk.deepx.ai/modelzoo/onnx/SqueezeNet1_1-3.onnx",
        "json_url": "https://sdk.deepx.ai/modelzoo/json/SqueezeNet1_1-3.json",
        "filename": "squeezenet1_1.onnx",
        "input_name": "input.1",
        "input_shape": [1, 224, 224, 3],
        "input_dtype": "uint8",
    },
}

_ELEMENT_SIZES = {"float32": 4, "float16": 2, "uint8": 1, "int8": 1}


def input_data_size(model_name: str) -> int:
    """Return the input tensor size in bytes for the given model."""
    meta = TEST_MODELS[model_name]
    shape = meta["input_shape"]
    elem = _ELEMENT_SIZES[meta["input_dtype"]]
    size = elem
    for d in shape:
        size *= d
    return size


def download_model(model_name: str, dest_dir: str) -> str:
    """Download the ONNX model to dest_dir. Returns path to the .onnx file."""
    meta = TEST_MODELS[model_name]
    dest = Path(dest_dir) / meta["filename"]
    if dest.exists():
        return str(dest)
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    _download_url(meta["url"], dest)
    return str(dest)


def download_model_json(model_name: str, dest_dir: str) -> str:
    """
    Download the model zoo JSON config for model_name to dest_dir.
    Rewrites dataset_path to './calibration_dataset' so it resolves correctly
    inside the conversion ZIP bundle. Returns path to the .json file.
    """
    meta = TEST_MODELS[model_name]
    json_filename = meta["json_url"].rsplit("/", 1)[-1]
    dest = Path(dest_dir) / json_filename
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    if not dest.exists():
        _download_url(meta["json_url"], dest)

    # Patch dataset_path so it resolves inside the ZIP bundle
    data = json.loads(dest.read_text())
    if "default_loader" in data and "dataset_path" in data["default_loader"]:
        data["default_loader"]["dataset_path"] = "./calibration_dataset"
        dest.write_text(json.dumps(data, indent=2))

    return str(dest)


def download_calibration_dataset(dest_dir: str) -> str:
    """
    Download and extract the calibration dataset to dest_dir/calibration_dataset/.
    Returns the path to the calibration_dataset/ directory.
    """
    dest_dir_path = Path(dest_dir)
    calib_dir = dest_dir_path / "calibration_dataset"
    if calib_dir.exists():
        return str(calib_dir)

    dest_dir_path.mkdir(parents=True, exist_ok=True)
    archive = dest_dir_path / "calibration_dataset.tar.gz"
    _download_url(CALIBRATION_DATASET_URL, archive)

    print("Extracting calibration dataset ...")
    with tarfile.open(archive) as tf:
        tf.extractall(path=dest_dir_path)
    archive.unlink()

    # The archive may extract into a subdirectory; normalise to calibration_dataset/
    if not calib_dir.exists():
        extracted = [d for d in dest_dir_path.iterdir() if d.is_dir() and d.name != "onnx"]
        if extracted:
            extracted[0].rename(calib_dir)

    return str(calib_dir)


def _download_url(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} ...")
    tmp = dest.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(url, timeout=300) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
