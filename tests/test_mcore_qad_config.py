"""CPU-only tests for the MCoreQAD YAML contract."""

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from angelslim import MCoreQADEngine as PublicMCoreQADEngine
from angelslim.compressor.mcore_qad.runner import validate_parallel_world_size
from angelslim.compressor.mcore_qad.train.config import ParallelConfig
from angelslim.engine import MCoreQADEngine
from angelslim.utils.config_parser import (
    MCORE_QAD_FORMATS,
    SlimConfigParser,
    parse_json_full_config,
)

PROJECT_ROOT = Path(__file__).parents[1]


def test_mcore_qad_engine_is_public():
    assert PublicMCoreQADEngine is MCoreQADEngine


def _write_config(tmp_path, mcore_qad, compression_name="MCoreQAD", dataset=None):
    payload = {
        "global": {"save_path": str(tmp_path / "output")},
        "model": {
            "name": "Qwen",
            "model_path": str(tmp_path / "model"),
        },
        "compression": {
            "name": compression_name,
            "MCoreQAD": mcore_qad,
        },
    }
    if dataset is not None:
        payload["dataset"] = dataset
    config_path = tmp_path / "mcore_qad.yaml"
    config_path.write_text(yaml.safe_dump(payload))
    return config_path


@pytest.mark.parametrize("format_name", MCORE_QAD_FORMATS)
def test_mcore_qad_accepts_every_migrated_format(tmp_path, format_name):
    config_path = _write_config(
        tmp_path,
        {
            "checkpoint_path": str(tmp_path / "mcore-checkpoint"),
            "format": format_name,
            "parallel": {
                "tensor_parallel": 2,
                "expert_parallel": 2,
                "sequence_parallel": True,
            },
            "optim": {"lr": 1e-4, "betas": [0.9, 0.95]},
        },
    )

    config = SlimConfigParser().parse(str(config_path))
    mcore_qad = config.compression_config.MCoreQAD

    assert config.compression_config.name == ["MCoreQAD"]
    assert mcore_qad.format == format_name
    assert mcore_qad.parallel.tensor_parallel == 2
    assert mcore_qad.parallel.sequence_parallel is True
    assert mcore_qad.optim.betas == (0.9, 0.95)


def test_mcore_qad_rejects_mixed_compression_lifecycle(tmp_path):
    config_path = _write_config(
        tmp_path,
        {"checkpoint_path": str(tmp_path / "mcore-checkpoint")},
        compression_name=["MCoreQAD", "PTQ"],
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        SlimConfigParser().parse(str(config_path))


def test_mcore_qad_rejects_unknown_format(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "checkpoint_path": str(tmp_path / "mcore-checkpoint"),
            "format": "unknown",
        },
    )

    with pytest.raises(ValueError, match="Unsupported MCoreQAD format"):
        SlimConfigParser().parse(str(config_path))


def test_mcore_qad_validates_parallel_constraints(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "checkpoint_path": str(tmp_path / "mcore-checkpoint"),
            "parallel": {
                "tensor_parallel": 1,
                "sequence_parallel": True,
            },
        },
    )

    with pytest.raises(ValueError, match="sequence_parallel"):
        SlimConfigParser().parse(str(config_path))


def test_mcore_qad_requires_sequence_parallel_with_tp(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "checkpoint_path": str(tmp_path / "mcore-checkpoint"),
            "parallel": {
                "tensor_parallel": 2,
                "sequence_parallel": False,
            },
        },
    )

    with pytest.raises(ValueError, match="requires sequence_parallel"):
        SlimConfigParser().parse(str(config_path))


def test_mcore_qad_saved_config_roundtrip(tmp_path):
    yaml_path = _write_config(
        tmp_path,
        {
            "checkpoint_path": str(tmp_path / "mcore-checkpoint"),
            "format": "w4afp8",
        },
    )
    parsed = SlimConfigParser().parse(str(yaml_path))
    json_path = tmp_path / "angelslim_config.json"
    json_path.write_text(json.dumps(asdict(parsed)))

    restored = parse_json_full_config(str(json_path))

    assert restored.compression_config.MCoreQAD.format == "w4afp8"
    assert restored.compression_config.MCoreQAD.checkpoint_path == str(
        tmp_path / "mcore-checkpoint"
    )


def test_mcore_qad_engine_redacts_paths_from_saved_config(tmp_path):
    yaml_path = _write_config(
        tmp_path,
        {
            "checkpoint_path": str(tmp_path / "mcore-checkpoint"),
            "init_scales_path": str(tmp_path / "initial-scales.pt"),
        },
        dataset={
            "name": "TextDataset",
            "data_path": str(tmp_path / "dataset.jsonl"),
        },
    )
    config = SlimConfigParser().parse(str(yaml_path))
    engine = MCoreQADEngine()
    engine.compressor = MagicMock()

    engine.save(str(tmp_path / "output"), config)

    saved = json.loads((tmp_path / "output" / "angelslim_config.json").read_text())
    assert saved["model_config"]["model_path"] == "Base Model Path"
    assert saved["global_config"]["save_path"] == "Save Model Path"
    assert saved["dataset_config"]["data_path"] == "Data Path"
    assert saved["compression_config"]["MCoreQAD"]["checkpoint_path"] == "MCore Checkpoint Path"
    assert saved["compression_config"]["MCoreQAD"]["init_scales_path"] == "Initial Scales Path"


def test_mcore_qad_engine_builds_backend_without_hf_model_preload(tmp_path):
    yaml_path = _write_config(
        tmp_path,
        {
            "checkpoint_path": str(tmp_path / "mcore-checkpoint"),
            "format": "fp8",
            "train_iters": 7,
        },
    )
    config = SlimConfigParser().parse(str(yaml_path))

    engine = MCoreQADEngine()
    compressor = engine.prepare_compressor(config)

    assert compressor.train_config.hf_path == str(tmp_path / "model")
    assert compressor.train_config.ckpt_path == str(tmp_path / "mcore-checkpoint")
    assert compressor.train_config.fmt == "fp8"
    assert compressor.train_config.train_iters == 7
    assert compressor.trainer is None


def test_programmatic_engine_prepares_checkpoint_before_training(tmp_path):
    yaml_path = _write_config(
        tmp_path,
        {"checkpoint_path": str(tmp_path / "mcore-checkpoint")},
    )
    config = SlimConfigParser().parse(str(yaml_path))
    engine = MCoreQADEngine()
    compressor = engine.prepare_compressor(config)

    with patch(
        "angelslim.compressor.mcore_qad.checkpoint.prepare.ensure_mcore_checkpoint"
    ) as ensure, patch.object(compressor, "run") as train:
        engine.run()

    ensure.assert_called_once_with(config)
    train.assert_called_once_with()


def test_parallel_folding_allows_cp2_ep32_on_world32():
    parallel = ParallelConfig(
        tensor_parallel=1,
        pipeline_parallel=1,
        expert_parallel=32,
        context_parallel=2,
        sequence_parallel=False,
    )

    validate_parallel_world_size(32, parallel)


@pytest.mark.parametrize(
    ("world_size", "parallel", "message"),
    [
        (10, ParallelConfig(tensor_parallel=4), "TP\\*PP\\*CP"),
        (10, ParallelConfig(expert_parallel=4), "ETP\\*EP\\*PP"),
    ],
)
def test_parallel_folding_rejects_incompatible_world_size(
    world_size,
    parallel,
    message,
):
    with pytest.raises(ValueError, match=message):
        validate_parallel_world_size(world_size, parallel)


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/qwen3/mcore_qad/qwen3_moe_nvfp4.yaml",
        "configs/Hy3/mcore_qad/hy_v3_nvfp4.yaml",
    ],
)
def test_mcore_qad_example_configs_parse(relative_path):
    config = SlimConfigParser().parse(str(PROJECT_ROOT / relative_path))

    assert config.compression_config.name == ["MCoreQAD"]
    assert config.compression_config.MCoreQAD.format == "nvfp4"
