"""
Model definitions and download helpers for the DeepX test suite.

Each entry in TEST_MODELS provides:
  - url / pt_name: source for downloading or exporting the ONNX file
  - filename:      name used when saving the ONNX file locally
  - input_name:    name of the model's input tensor (needed for dx_com config.json)
  - input_shape:   [N, C, H, W] shape
  - input_dtype:   element type of the compiled model's input (always uint8 after dx_com)
  - task:          "image_classification" or "object_detection"
"""

import shutil
import urllib.request
from pathlib import Path

# YOLO models are exported via ultralytics (pt_name → ONNX export)
_YOLO_EXPORTS = {
    "yolov8n": ("yolov8n.pt", 1, 640),
    "yolo11n": ("yolo11n.pt", 1, 640),
    "yolo11s": ("yolo11s.pt", 1, 640),
}

TEST_MODELS = {
    "squeezenet": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-7.onnx",
        "filename": "squeezenet.onnx",
        "input_name": "data_0",
        "input_shape": [1, 3, 224, 224],
        "input_dtype": "uint8",
        "task": "image_classification",
    },
    "resnet18": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx",
        "filename": "resnet18.onnx",
        "input_name": "data",
        "input_shape": [1, 3, 224, 224],
        "input_dtype": "uint8",
        "task": "image_classification",
    },
    "mobilenetv2": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
        "filename": "mobilenetv2.onnx",
        "input_name": "data",
        "input_shape": [1, 3, 224, 224],
        "input_dtype": "uint8",
        "task": "image_classification",
    },
    "yolov8n": {
        "pt_name": "yolov8n.pt",
        "filename": "yolov8n.onnx",
        "input_name": "images",
        "input_shape": [1, 3, 640, 640],
        "input_dtype": "uint8",
        "task": "object_detection",
    },
    "yolo11n": {
        "pt_name": "yolo11n.pt",
        "filename": "yolo11n.onnx",
        "input_name": "images",
        "input_shape": [1, 3, 640, 640],
        "input_dtype": "uint8",
        "task": "object_detection",
    },
    "yolo11s": {
        "pt_name": "yolo11s.pt",
        "filename": "yolo11s.onnx",
        "input_name": "images",
        "input_shape": [1, 3, 640, 640],
        "input_dtype": "uint8",
        "task": "object_detection",
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
    """
    Download (or export) the ONNX model to dest_dir.
    Returns the path to the .onnx file.
    Raises RuntimeError for YOLO models if ultralytics is not installed.
    """
    meta = TEST_MODELS[model_name]
    dest = Path(dest_dir) / meta["filename"]

    if dest.exists():
        return str(dest)

    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    if "url" in meta:
        _download_url(meta["url"], dest)
    elif "pt_name" in meta:
        _export_yolo(model_name, meta, dest)
    else:
        raise ValueError(f"No download source for model '{model_name}'")

    return str(dest)


def _download_url(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} ...")
    tmp = dest.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _export_yolo(model_name: str, meta: dict, dest: Path) -> None:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        raise RuntimeError(
            f"ultralytics is required to export {model_name}. Install with: pip install ultralytics"
        ) from None

    pt_name, batch, imgsz = (
        meta["pt_name"],
        meta["input_shape"][0],
        meta["input_shape"][2],
    )

    print(f"Exporting {pt_name} (batch={batch}, imgsz={imgsz}) to ONNX ...")
    model = YOLO(pt_name)
    exported = model.export(format="onnx", imgsz=imgsz, batch=batch, opset=11, simplify=True, dynamic=False)
    if not exported or not Path(str(exported)).exists():
        raise RuntimeError(f"YOLO export produced no .onnx file for {model_name}")

    shutil.move(str(exported), dest)
