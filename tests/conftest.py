"""
Shared session fixtures for the DeepX OAAX test suite.

converted_models downloads all configured ONNX models and converts them to
DXNN format via the Docker toolchain image, caching to tests/compiled_models/.
Stage 1 populates this cache; Stage 2 reads from it without re-converting.
"""

import json
import os
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from pathlib import Path

import pytest

from tests.models import TEST_MODELS, download_model

COMPILED_DIR = Path(__file__).parent / "compiled_models"
DOCKER_IMAGE = os.environ.get("DEEPX_TOOLCHAIN_IMAGE", "oaax-deepx-toolchain:latest")
_REPO_ROOT = Path(__file__).parent.parent


def _docker_image_available() -> bool:
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=30)
        if r.returncode != 0:
            print(f"\n[docker check] 'docker info' failed (rc={r.returncode}): {r.stderr.decode()[:300]}")
            return False
        r = subprocess.run(
            ["docker", "inspect", "--type=image", DOCKER_IMAGE],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode != 0:
            print(f"\n[docker check] image '{DOCKER_IMAGE}' not found (docker inspect rc={r.returncode})")
            print("\n[docker check] Available images:")
            subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], timeout=10)
        return r.returncode == 0
    except Exception as e:
        print(f"\n[docker check] exception: {type(e).__name__}: {e}")
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
    # input_shape is NHWC [N,H,W,C]; dx_com expects the ONNX NCHW shape [N,C,H,W]
    n, h, w, c = meta["input_shape"]
    config = {"input_shapes": {meta["input_name"]: [n, c, h, w]}}

    out_dir.mkdir(parents=True, exist_ok=True)
    dxnn_path = out_dir / f"{model_name}.dxnn"

    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        tmp_path = Path(tmp)
        bundle = tmp_path / "bundle.zip"
        docker_out = tmp_path / "output"
        docker_out.mkdir()

        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(onnx_path, arcname=f"{model_name}.onnx")
            z.writestr("config.json", json.dumps(config))

        # Use docker create + cp + start instead of bind mounts so this works
        # in DinD CI environments where the host Docker daemon can't resolve
        # paths from inside the CI container (/__w vs /home/github-runner/...).
        container = f"deepx-convert-{model_name}-{uuid.uuid4().hex[:8]}"
        result = None
        try:
            subprocess.run(
                ["docker", "create", "--name", container, DOCKER_IMAGE, "/tmp/bundle.zip", "/tmp/output"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["docker", "cp", str(bundle), f"{container}:/tmp/bundle.zip"],
                check=True,
                capture_output=True,
            )
            result = subprocess.run(
                ["docker", "start", "-a", container],
                capture_output=True,
                text=True,
                timeout=600,
            )
            subprocess.run(
                ["docker", "cp", f"{container}:/tmp/output/.", str(docker_out)],
                capture_output=True,
            )
        finally:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)

        log_src = docker_out / "convert.log"
        if log_src.exists():
            shutil.copy(log_src, out_dir / "convert.log")

        if result is None or result.returncode != 0:
            raise RuntimeError(
                f"Conversion failed for {model_name} "
                f"(exit {result.returncode if result else 'N/A'}):\n"
                f"{result.stdout if result else ''}\n{result.stderr if result else ''}"
            )

        produced = list(docker_out.glob("*.dxnn"))
        if not produced:
            raise RuntimeError(f"Conversion for {model_name} succeeded but no .dxnn file was produced.")

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
        onnx = Path(download_model(name, str(onnx_dir)))
        result[name] = _convert_with_docker(name, onnx, COMPILED_DIR / name)

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
        onnx = Path(download_model(name, str(onnx_dir)))
        result[name] = _convert_with_docker(name, onnx, COMPILED_DIR / name)

    return result
