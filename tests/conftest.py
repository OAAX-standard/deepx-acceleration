"""
Shared session fixtures for the DeepX OAAX test suite.

converted_models downloads all configured ONNX models and converts them to
DXNN format via the Docker toolchain image, caching to tests/compiled_models/.
Stage 1 populates this cache; Stage 2 reads from it without re-converting.
"""

import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pytest

from tests.models import TEST_MODELS, download_model

COMPILED_DIR = Path(__file__).parent / "compiled_models"
DOCKER_IMAGE = os.environ.get("DEEPX_TOOLCHAIN_IMAGE", "deepx-conversion-toolchain:22.04")


def _docker_image_available() -> bool:
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=30)
        if r.returncode != 0:
            return False
        r = subprocess.run(
            ["docker", "images", "-q", DOCKER_IMAGE],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


def _convert_with_docker(
    model_name: str,
    onnx_path: Path,
    out_dir: Path,
) -> Path:
    """
    Convert one ONNX model to DXNN using the Docker toolchain image.
    Returns the path to the produced .dxnn file.
    """
    meta = TEST_MODELS[model_name]
    config = {"input_shapes": {meta["input_name"]: meta["input_shape"]}}

    out_dir.mkdir(parents=True, exist_ok=True)
    dxnn_path = out_dir / f"{model_name}.dxnn"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bundle = tmp_path / "bundle.zip"
        docker_out = tmp_path / "output"
        docker_out.mkdir()

        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(onnx_path, arcname=f"{model_name}.onnx")
            z.writestr("config.json", json.dumps(config))

        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{bundle}:/input/bundle.zip",
                "-v",
                f"{docker_out}:/output",
                "--entrypoint",
                "bash",
                DOCKER_IMAGE,
                "-c",
                "cd /app && /app/scripts/convert.sh /input/bundle.zip /output",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        log_src = docker_out / "convert.log"
        if log_src.exists():
            import shutil

            shutil.copy(log_src, out_dir / "convert.log")

        if result.returncode != 0:
            raise RuntimeError(
                f"Conversion failed for {model_name} (exit {result.returncode}):\n" f"{result.stdout}\n{result.stderr}"
            )

        produced = list(docker_out.glob("*.dxnn"))
        if not produced:
            raise RuntimeError(f"Conversion for {model_name} succeeded but no .dxnn file was produced.")

        import shutil

        shutil.copy(produced[0], dxnn_path)

    return dxnn_path


@pytest.fixture(scope="session")
def compiled_classification_models() -> dict:
    """
    Download and convert classification models (SqueezeNet, ResNet18, MobileNetV2).
    Returns {model_name: Path-to-dxnn}.
    """
    if not _docker_image_available():
        pytest.skip(
            f"Docker image '{DOCKER_IMAGE}' not available. " f"Override with: DEEPX_TOOLCHAIN_IMAGE=<image> pytest ..."
        )

    onnx_dir = COMPILED_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for name in ("squeezenet", "resnet18", "mobilenetv2"):
        dxnn = COMPILED_DIR / name / f"{name}.dxnn"
        if dxnn.exists():
            result[name] = dxnn
            continue
        try:
            onnx = Path(download_model(name, str(onnx_dir)))
            result[name] = _convert_with_docker(name, onnx, COMPILED_DIR / name)
        except Exception as e:
            print(f"\nWarning: skipping {name}: {e}")

    return result


@pytest.fixture(scope="session")
def compiled_yolo_models() -> dict:
    """
    Export and convert YOLO models (yolov8n, yolo11n, yolo11s).
    Requires ultralytics. Returns {model_name: Path-to-dxnn}.
    """
    if not _docker_image_available():
        pytest.skip(f"Docker image '{DOCKER_IMAGE}' not available.")

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        pytest.skip("ultralytics not installed — run: pip install ultralytics")

    onnx_dir = COMPILED_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for name in ("yolov8n", "yolo11n", "yolo11s"):
        dxnn = COMPILED_DIR / name / f"{name}.dxnn"
        if dxnn.exists():
            result[name] = dxnn
            continue
        try:
            onnx = Path(download_model(name, str(onnx_dir)))
            result[name] = _convert_with_docker(name, onnx, COMPILED_DIR / name)
        except Exception as e:
            print(f"\nWarning: skipping {name}: {e}")

    return result
