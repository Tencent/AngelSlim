# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from ....utils.config_parser import default_run_dirs

__all__ = [
    "ConvertSpec",
    "fp8_blockwise_spec",
    "spec_from_config",
    "convert_checkpoint",
    "run_calibration",
]


@dataclass
class ConvertSpec:
    model_path: str
    save_path: str
    config_groups: dict
    model_name: str = ""  # YAML model.name; selects the legacy fp8_blockwise suffix table
    ignore: List[str] = field(default_factory=list)
    algorithm: str = "rtn"
    gptq: dict = field(default_factory=dict)
    legacy: bool = False  # fp8_blockwise: ``quant_method: fp8`` instead of compressed-tensors
    split_experts: bool = True
    hessian_dir: Optional[str] = None
    stats_dir: Optional[str] = None
    mtp_file: Optional[str] = None
    num_workers: int = 8
    skip_unquantized_source: bool = False  # FP8 source: keep its BF16 modules unquantized


def fp8_blockwise_spec(model_path, save_path, block_size=(128, 128), num_workers=8, ignore=(),
                       model_name=""):  # fmt: skip
    from .export import fp8_blockwise_targets

    group = {"format": "fp8_block", "block_size": list(block_size),
             "targets": fp8_blockwise_targets(model_name)}  # fmt: skip
    return ConvertSpec(
        model_path=model_path,
        save_path=save_path,
        model_name=model_name,
        config_groups={"fp8_blockwise": group},
        ignore=list(ignore),
        legacy=True,
        # Qwen3.5 stores experts batched and the official FP8 layout splits them per expert.
        split_experts=model_name.lower() == "qwen3_5",
        num_workers=num_workers,
    )


def needs_hessian(quant):
    """True when any group resolves to GPTQ. ``hessian_dir`` also selects the calibration
    entry point, so an RTN run must not get one."""
    if quant is None or quant.name != "mixed":
        return False
    from .convert import build_groups

    groups = build_groups(quant.config_groups, quant.algorithm, dict(quant.gptq or {}))
    return any(g.algorithm == "gptq" for g in groups)


def spec_from_config(config):
    quant = config.compression_config.quantization
    calib = config.compression_config.calibrate
    model_path = config.model_config.model_path
    # save_path is the run root; weights go to out/, calibration dirs default beside them.
    dirs = default_run_dirs(config.global_config.save_path)
    save_path = dirs["save_path"]
    if quant.name == "fp8_blockwise":
        qm = quant.quant_method if isinstance(quant.quant_method, dict) else {}
        return fp8_blockwise_spec(
            model_path, save_path, tuple(qm.get("block_size", [128, 128])),
            int(qm.get("num_workers", 8)), quant.ignore_layers, config.model_config.name,
        )  # fmt: skip
    spec = ConvertSpec(
        model_path=model_path,
        save_path=save_path,
        model_name=config.model_config.name,
        config_groups=quant.config_groups,
        ignore=list(quant.ignore_layers),
        algorithm=quant.algorithm,
        gptq=dict(quant.gptq or {}),
        hessian_dir=getattr(calib, "hessian_dir", None)
        or (dirs["hessian_dir"] if needs_hessian(quant) else None),
        stats_dir=getattr(calib, "stats_dir", None) or dirs["stats_dir"],
        mtp_file=quant.mtp_file,
        num_workers=quant.num_workers,
        skip_unquantized_source=quant.skip_unquantized_source,
    )
    return spec


def run_calibration(config, config_path):
    """Stage 1: one vLLM forward collecting MoE absmax, plus Hessians when GPTQ needs them.
    GPTQ runs vllm_calib.hessian_collect, RTN the lighter vllm_calib.calibrate."""
    calib = config.compression_config.calibrate
    if calib is None or calib.backend != "vllm":
        print("calib stage: no compression.calibrate with backend vllm, skipping")
        return
    dirs = default_run_dirs(config.global_config.save_path)
    stats_dir = calib.stats_dir or dirs["stats_dir"]
    hessian_dir = calib.hessian_dir or (
        dirs["hessian_dir"] if needs_hessian(config.compression_config.quantization) else None
    )
    embeds = (calib.args or {}).get("image_embeds_dir") or dirs["image_embeds_dir"]
    outputs = [os.path.join(hessian_dir, "hessian_meta.json")] if hessian_dir else []
    outputs.append(os.path.join(stats_dir, "moe_expert_stats.json"))
    if (calib.args or {}).get("enable_mtp"):
        outputs.append(os.path.join(stats_dir, "mtp_moe_expert_stats.json"))
    if all(os.path.isfile(p) for p in outputs):
        print(f"calib stage: reusing {', '.join(outputs)}")
        return
    if not config.dataset_config or not config.dataset_config.data_path:
        raise ValueError("calib stage needs dataset.data_path")
    module = "hessian_collect" if hessian_dir else "calibrate"
    cmd = [sys.executable, "-m", f"angelslim.compressor.quant.vllm_calib.{module}",
           "-c", config_path, "--model-path", config.model_config.model_path,
           "--image-embeds-dir", embeds]  # fmt: skip
    # Passed explicitly: the subprocess re-reads the YAML and would otherwise miss dirs
    # derived from a --save-path given on the command line.
    if hessian_dir:
        cmd += ["--hessian-dir", hessian_dir,
                "--moe-stats-out", os.path.join(stats_dir, "moe_expert_stats.json")]  # fmt: skip
        if (calib.args or {}).get("enable_mtp"):
            cmd += ["--mtp-moe-stats-out",
                    os.path.join(stats_dir, "mtp_moe_expert_stats.json")]  # fmt: skip
    else:
        cmd += ["--output-dir", stats_dir]
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([repo, os.environ.get("PYTHONPATH", "")])}
    print("calib stage: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    missing = [p for p in outputs if not os.path.isfile(p)]
    if missing:
        raise RuntimeError(f"calibration finished without {missing}")


def convert_checkpoint(spec, stage="all", node_rank=0, num_nodes=1, skip_existing=False):
    """``quant``: this node's shards; ``finalize``: index + config.json; ``all``: both."""
    from .export import finalize, run_node

    os.makedirs(spec.save_path, exist_ok=True)
    if stage in ("quant", "all"):
        run_node(spec, node_rank, num_nodes, skip_existing)
    if stage in ("finalize", "all"):
        if stage == "all" and num_nodes > 1:
            print("multi-node run: finalize once with --stage finalize after every node")
            return
        finalize(spec)
