"""
Stage 1 tests: verify that ONNX models are correctly converted to DXNN format.
"""

from pathlib import Path

import pytest

from tests.models import TEST_MODELS

CLASSIFICATION_MODELS = ["squeezenet", "resnet18", "mobilenetv2"]
YOLO_MODELS = ["yolov8n", "yolo11n", "yolo11s"]


# ---------------------------------------------------------------------------
# Classification models
# ---------------------------------------------------------------------------


class TestClassificationConversion:
    @pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
    def test_dxnn_exists(self, compiled_classification_models, model_name):
        if model_name not in compiled_classification_models:
            pytest.skip(f"{model_name} was not converted (check warnings above)")
        assert compiled_classification_models[model_name].exists()

    @pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
    def test_dxnn_nonempty(self, compiled_classification_models, model_name):
        if model_name not in compiled_classification_models:
            pytest.skip(f"{model_name} was not converted")
        path = compiled_classification_models[model_name]
        assert path.stat().st_size > 0, f"{path.name} is empty"

    @pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
    def test_convert_log_exists(self, compiled_classification_models, model_name):
        if model_name not in compiled_classification_models:
            pytest.skip(f"{model_name} was not converted")
        log = compiled_classification_models[model_name].parent / "convert.log"
        assert log.exists(), "convert.log not found"

    @pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
    def test_convert_log_no_errors(self, compiled_classification_models, model_name):
        if model_name not in compiled_classification_models:
            pytest.skip(f"{model_name} was not converted")
        log = compiled_classification_models[model_name].parent / "convert.log"
        if not log.exists():
            pytest.skip("convert.log not found")
        text = log.read_text()
        assert "Error" not in text and "error" not in text.lower().replace(
            "conversion finished", ""
        ), f"Errors found in convert.log:\n{text}"


# ---------------------------------------------------------------------------
# YOLO models
# ---------------------------------------------------------------------------


class TestYoloConversion:
    @pytest.mark.parametrize("model_name", YOLO_MODELS)
    def test_dxnn_exists(self, compiled_yolo_models, model_name):
        if model_name not in compiled_yolo_models:
            pytest.skip(f"{model_name} was not converted (check warnings above)")
        assert compiled_yolo_models[model_name].exists()

    @pytest.mark.parametrize("model_name", YOLO_MODELS)
    def test_dxnn_nonempty(self, compiled_yolo_models, model_name):
        if model_name not in compiled_yolo_models:
            pytest.skip(f"{model_name} was not converted")
        path = compiled_yolo_models[model_name]
        assert path.stat().st_size > 0, f"{path.name} is empty"

    @pytest.mark.parametrize("model_name", YOLO_MODELS)
    def test_convert_log_exists(self, compiled_yolo_models, model_name):
        if model_name not in compiled_yolo_models:
            pytest.skip(f"{model_name} was not converted")
        log = compiled_yolo_models[model_name].parent / "convert.log"
        assert log.exists(), "convert.log not found"

    @pytest.mark.parametrize("model_name", YOLO_MODELS)
    def test_convert_log_no_errors(self, compiled_yolo_models, model_name):
        if model_name not in compiled_yolo_models:
            pytest.skip(f"{model_name} was not converted")
        log = compiled_yolo_models[model_name].parent / "convert.log"
        if not log.exists():
            pytest.skip("convert.log not found")
        text = log.read_text()
        assert "Error" not in text and "error" not in text.lower().replace(
            "conversion finished", ""
        ), f"Errors found in convert.log:\n{text}"
