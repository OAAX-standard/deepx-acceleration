"""
Stage 2 tests: run compiled DXNN models on DeepX hardware and validate metrics.

Requirements:
  - Compiled DXNN models in tests/compiled_models/  (run stage 1 first)
  - inference_runner binary  (run: bash tests/runtime/build-tests.sh)
  - DeepX NPU hardware (DX-M1 or DX-H1)

Both requirements are checked at collection time; all tests in this module are
skipped automatically when either is missing.

Override the binary location:
  DEEPX_RUNNER_DIR=/path/to/dir pytest tests/test_inference.py
"""

import os
import subprocess
from pathlib import Path

import pytest

from tests.models import TEST_MODELS, input_data_size

TESTS_DIR = Path(__file__).parent
COMPILED_DIR = TESTS_DIR / "compiled_models"

_RUNNER_CANDIDATES = [
    TESTS_DIR / "runtime" / "build" / "inference_runner",
    TESTS_DIR / "runtime" / "build-x86_64" / "inference_runner",
    TESTS_DIR / "runtime" / "build-aarch64" / "inference_runner",
]

# Counts kept low so hardware tests are fast in CI
_TEST_RUNS = 10
_TEST_WARMUP = 3


def _find_runner() -> Path | None:
    runner_dir = os.environ.get("DEEPX_RUNNER_DIR")
    if runner_dir:
        p = Path(runner_dir) / "inference_runner"
        if p.exists():
            return p
    for p in _RUNNER_CANDIDATES:
        if p.exists():
            return p
    return None


def _compiled_model_names() -> list[str]:
    return [dxnn.parent.name for dxnn in sorted(COMPILED_DIR.glob("*/*.dxnn")) if dxnn.parent.name in TEST_MODELS]


# ---------------------------------------------------------------------------
# Module-level skip guards
# ---------------------------------------------------------------------------

_runner = _find_runner()
_models = _compiled_model_names()

pytestmark = [
    pytest.mark.skipif(
        _runner is None,
        reason="inference_runner binary not found — build with: bash tests/runtime/build-tests.sh",
    ),
    pytest.mark.skipif(
        not _models,
        reason="No compiled DXNN models found — run stage 1 first: uv run python tests/stage1.py",
    ),
]


def _model_args(model_name: str) -> list[str]:
    """Per-model inference_runner arguments; repeat them to load several models."""
    return [
        "--model",
        str(COMPILED_DIR / model_name / f"{model_name}.dxnn"),
        "--input-size",
        str(input_data_size(model_name)),
        "--input-shape",
        ",".join(str(d) for d in TEST_MODELS[model_name]["input_shape"]),
    ]


def _run(runner: Path, *args: str) -> subprocess.CompletedProcess:
    cmd = [str(runner), *args, "--runs", str(_TEST_RUNS), "--warmup", str(_TEST_WARMUP), "--csv"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runner() -> Path:
    return _runner  # type: ignore[return-value]  # guarded by pytestmark


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", _models or ["_placeholder"])
def test_inference_succeeds(runner, model_name):
    """inference_runner exits 0 and produces valid CSV output."""
    result = _run(runner, *_model_args(model_name))
    assert result.returncode == 0, f"inference_runner failed for '{model_name}':\n{result.stderr.strip()}"

    line = result.stdout.strip()
    parts = line.split(",")
    assert len(parts) == 8, f"Expected 8 CSV fields, got: '{line}'"


@pytest.mark.parametrize("model_name", _models or ["_placeholder"])
def test_inference_metrics_valid(runner, model_name):
    """Reported latency and throughput are positive finite numbers."""
    result = _run(runner, *_model_args(model_name))
    if result.returncode != 0:
        pytest.skip("inference_runner failed — see test_inference_succeeds for details")

    parts = result.stdout.strip().split(",")
    # CSV: model,runs,avg_ms,p50_ms,p95_ms,min_ms,max_ms,throughput_fps
    runs = int(parts[1])
    avg_ms = float(parts[2])
    p50_ms = float(parts[3])
    p95_ms = float(parts[4])
    min_ms = float(parts[5])
    max_ms = float(parts[6])
    throughput = float(parts[7])

    assert runs == _TEST_RUNS
    assert avg_ms > 0, f"avg_ms={avg_ms} should be positive"
    assert throughput > 0, f"throughput={throughput} should be positive"
    assert min_ms <= avg_ms <= max_ms, f"min={min_ms} <= avg={avg_ms} <= max={max_ms} violated"
    assert p50_ms <= p95_ms, f"p50={p50_ms} <= p95={p95_ms} violated"


def test_multi_model_from_memory(runner):
    """Two models loaded from memory in one call each report one valid CSV row."""
    if len(_models) < 2:
        pytest.skip("needs at least two compiled models")
    names = _models[:2]

    result = _run(runner, *_model_args(names[0]), *_model_args(names[1]), "--from-memory")
    assert result.returncode == 0, f"inference_runner failed for {names}:\n{result.stderr.strip()}"

    rows = [line.split(",") for line in result.stdout.strip().splitlines()]
    assert [row[0] for row in rows] == names
    for row in rows:
        assert len(row) == 8, f"Expected 8 CSV fields, got: {row}"
        assert int(row[1]) == _TEST_RUNS
        assert float(row[7]) > 0, f"throughput={row[7]} should be positive"


def test_unpaired_model_args_are_rejected(runner):
    """Each --model needs its own --input-size; a count mismatch is a usage error (exit 2)."""
    name = _models[0]
    model = str(COMPILED_DIR / name / f"{name}.dxnn")
    result = _run(runner, "--model", model, "--model", model, "--input-size", str(input_data_size(name)))
    assert result.returncode == 2, result.stderr
