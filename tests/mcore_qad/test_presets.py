"""Ensure every YAML-exposed MCoreQAD format resolves to a quantization preset."""

import pytest

from angelslim.compressor.mcore_qad.quant.presets import FORMATS, get_format
from angelslim.utils.config_parser import MCORE_QAD_FORMATS


@pytest.mark.parametrize("format_name", MCORE_QAD_FORMATS)
def test_every_config_format_has_a_backend_preset(format_name):
    weight_spec, activation_spec = get_format(format_name)

    assert set(FORMATS) == set(MCORE_QAD_FORMATS)
    assert not weight_spec.is_identity()
    assert weight_spec.source == "learnable"
    if format_name in ("nvfp4a16", "w4a16"):
        assert activation_spec.is_identity()
    else:
        assert activation_spec.source == "dynamic"
