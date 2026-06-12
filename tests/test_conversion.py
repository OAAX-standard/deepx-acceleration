"""
Stage 1 tests: verify that ONNX models are correctly converted to DXNN format.
"""

import pytest

from tests.models import TEST_MODELS

ALL_MODELS = list(TEST_MODELS.keys())

MIN_DXNN_SIZE_BYTES = 1024


class TestModelConversion:
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_dxnn_exists(self, compiled_models, model_name):
        assert compiled_models[model_name].exists()

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_dxnn_nonempty(self, compiled_models, model_name):
        path = compiled_models[model_name]
        assert path.stat().st_size > 0, f"{path.name} is empty"

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_dxnn_min_size(self, compiled_models, model_name):
        path = compiled_models[model_name]
        size = path.stat().st_size
        assert size >= MIN_DXNN_SIZE_BYTES, (
            f"{path.name} is suspiciously small: {size} bytes (expected >= {MIN_DXNN_SIZE_BYTES})"
        )

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_convert_log_exists(self, compiled_models, model_name):
        log = compiled_models[model_name].parent / "convert.log"
        assert log.exists(), "convert.log not found"

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_convert_log_success(self, compiled_models, model_name):
        log = compiled_models[model_name].parent / "convert.log"
        if not log.exists():
            pytest.skip("convert.log not found")
        text = log.read_text()
        assert "Conversion finished" in text, (
            f"'Conversion finished' not found in convert.log — conversion may have failed:\n{text[-500:]}"
        )
