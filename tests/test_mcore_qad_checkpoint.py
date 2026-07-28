"""CPU-only tests for automatic MCoreQAD checkpoint preparation."""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from angelslim.compressor.mcore_qad.checkpoint.prepare import ensure_mcore_checkpoint

_METADATA = {
    "sharded_backend": "torch_dist",
    "sharded_backend_version": 1,
    "common_backend": "torch",
    "common_backend_version": 1,
}


def _config(tmp_path, use_cpu=False):
    checkpoint_path = tmp_path / "mcore-checkpoint"
    return (
        SimpleNamespace(
            model_config=SimpleNamespace(model_path=str(tmp_path / "hf-model")),
            compression_config=SimpleNamespace(
                MCoreQAD=SimpleNamespace(
                    checkpoint_path=str(checkpoint_path),
                    checkpoint_conversion_cpu=use_cpu,
                )
            ),
        ),
        checkpoint_path,
    )


def _single_rank(monkeypatch):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)


def test_existing_mcore_checkpoint_is_reused(tmp_path, monkeypatch):
    _single_rank(monkeypatch)
    config, checkpoint_path = _config(tmp_path)
    checkpoint_path.mkdir()
    (checkpoint_path / "metadata.json").write_text(json.dumps(_METADATA))

    with patch("angelslim.compressor.mcore_qad.checkpoint.prepare.subprocess.run") as run:
        converted = ensure_mcore_checkpoint(config)

    assert converted is False
    run.assert_not_called()


def test_missing_checkpoint_is_converted_once(tmp_path, monkeypatch):
    _single_rank(monkeypatch)
    config, checkpoint_path = _config(tmp_path, use_cpu=True)

    def fake_converter(command, check, env):
        assert check is True
        assert env["WORLD_SIZE"] == "1"
        checkpoint_path.mkdir(parents=True)
        (checkpoint_path / "metadata.json").write_text(json.dumps(_METADATA))

    with patch(
        "angelslim.compressor.mcore_qad.checkpoint.prepare.subprocess.run",
        side_effect=fake_converter,
    ) as run:
        converted = ensure_mcore_checkpoint(config)

    assert converted is True
    command = run.call_args.args[0]
    assert command[1:3] == [
        "-m",
        "angelslim.compressor.mcore_qad.tools.hf_to_megatron",
    ]
    assert command[-1] == "--cpu"


def test_partial_checkpoint_is_reconverted(tmp_path, monkeypatch):
    _single_rank(monkeypatch)
    config, checkpoint_path = _config(tmp_path)
    checkpoint_path.mkdir()
    (checkpoint_path / "partial-shard").write_text("incomplete")

    def fake_converter(command, check, env):
        (checkpoint_path / "metadata.json").write_text(json.dumps(_METADATA))

    with patch(
        "angelslim.compressor.mcore_qad.checkpoint.prepare.subprocess.run",
        side_effect=fake_converter,
    ) as run:
        assert ensure_mcore_checkpoint(config) is True

    run.assert_called_once()


def test_conversion_failure_has_backend_context(tmp_path, monkeypatch):
    _single_rank(monkeypatch)
    config, _ = _config(tmp_path)

    with patch(
        "angelslim.compressor.mcore_qad.checkpoint.prepare.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["converter"]),
    ), pytest.raises(RuntimeError, match="checkpoint conversion failed"):
        ensure_mcore_checkpoint(config)
