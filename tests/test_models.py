"""
Unit tests for models.py metadata.

No Docker, hardware, or network access required — these run anywhere.
"""

import pytest

from tests.models import TEST_MODELS, input_data_size

REQUIRED_KEYS = {"url", "json_url", "filename", "input_name", "input_shape", "input_dtype"}
VALID_DTYPES = {"float32", "float16", "uint8", "int8"}

EXPECTED_SIZES = {
    "mobilenetv1": 1 * 224 * 224 * 3,
    "mobilenetv2": 1 * 224 * 224 * 3,
    "squeezenet1_1": 1 * 224 * 224 * 3,
}


class TestModelMetadata:
    @pytest.mark.parametrize("name", TEST_MODELS.keys())
    def test_required_keys_present(self, name):
        missing = REQUIRED_KEYS - TEST_MODELS[name].keys()
        assert not missing, f"Model '{name}' is missing keys: {missing}"

    @pytest.mark.parametrize("name", TEST_MODELS.keys())
    def test_input_shape_is_4d_nhwc(self, name):
        shape = TEST_MODELS[name]["input_shape"]
        assert len(shape) == 4, f"'{name}' input_shape should be 4D [N,H,W,C], got {shape}"
        assert shape[0] == 1, f"'{name}' batch dimension should be 1, got {shape[0]}"
        assert all(d > 0 for d in shape), f"'{name}' all shape dims must be positive, got {shape}"

    @pytest.mark.parametrize("name", TEST_MODELS.keys())
    def test_input_dtype_valid(self, name):
        dtype = TEST_MODELS[name]["input_dtype"]
        assert dtype in VALID_DTYPES, f"'{name}' input_dtype='{dtype}' is not one of {VALID_DTYPES}"

    @pytest.mark.parametrize("name", TEST_MODELS.keys())
    def test_filename_ends_with_onnx(self, name):
        fname = TEST_MODELS[name]["filename"]
        assert fname.endswith(".onnx"), f"'{name}' filename should end with .onnx, got '{fname}'"

    def test_no_duplicate_filenames(self):
        filenames = [m["filename"] for m in TEST_MODELS.values()]
        assert len(filenames) == len(set(filenames)), "Duplicate filenames in TEST_MODELS"

    def test_no_duplicate_urls(self):
        urls = [m["url"] for m in TEST_MODELS.values()]
        assert len(urls) == len(set(urls)), "Duplicate URLs in TEST_MODELS"

    @pytest.mark.parametrize("name,expected", EXPECTED_SIZES.items())
    def test_input_data_size(self, name, expected):
        assert input_data_size(name) == expected, (
            f"input_data_size('{name}') = {input_data_size(name)}, expected {expected}"
        )
