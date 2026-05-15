"""
Shared session fixtures for the DeepX OAAX test suite.

compiled_models downloads classification ONNX models and converts them to
DXNN format via the Docker toolchain image, caching to tests/compiled_models/.
Stage 1 populates this cache; Stage 2 reads from it without re-converting.
"""

import os
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from pathlib import Path

import pytest

from tests.models import TEST_MODELS, download_calibration_dataset, download_model, download_model_json

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
    json_path: Path,
    calib_dir: Path,
    out_dir: Path,
) -> Path:
    """
    Convert one ONNX model to DXNN using the Docker toolchain image.
    Returns the path to the produced .dxnn file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dxnn_path = out_dir / f"{model_name}.dxnn"

    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        tmp_path = Path(tmp)
        bundle = tmp_path / "bundle.zip"
        docker_out = tmp_path / "output"
        docker_out.mkdir()

        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(onnx_path, arcname=f"{model_name}.onnx")
            z.write(json_path, arcname=json_path.name)
            for f in sorted(calib_dir.rglob("*")):
                if f.is_file():
                    z.write(f, arcname=Path("calibration_dataset") / f.relative_to(calib_dir))

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


class _LazyCompiledModels:
    """Converts models on first access so pytest reports results per model."""

    def __init__(self, onnx_dir: Path, calib_dir: Path):
        self._onnx_dir = onnx_dir
        self._calib_dir = calib_dir
        self._cache: dict = {}

    def __getitem__(self, name: str) -> Path:
        if name in self._cache:
            return self._cache[name]
        dxnn = COMPILED_DIR / name / f"{name}.dxnn"
        if not dxnn.exists():
            onnx = Path(download_model(name, str(self._onnx_dir)))
            json_path = Path(download_model_json(name, str(self._onnx_dir)))
            dxnn = _convert_with_docker(name, onnx, json_path, self._calib_dir, COMPILED_DIR / name)
        self._cache[name] = dxnn
        return dxnn


@pytest.fixture(scope="session")
def compiled_models() -> _LazyCompiledModels:
    """
    Returns a lazy dict-like object that downloads and converts each model on first access.
    Conversion is triggered per model as tests run, so results are reported incrementally.
    """
    if not _docker_image_available():
        pytest.skip(
            f"Docker image '{DOCKER_IMAGE}' not available. "
            f"Override with: DEEPX_TOOLCHAIN_IMAGE=<image> pytest ..."
        )

    onnx_dir = COMPILED_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = Path(download_calibration_dataset(str(COMPILED_DIR)))

    return _LazyCompiledModels(onnx_dir, calib_dir)
