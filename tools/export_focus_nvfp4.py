#!/usr/bin/env python3
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

"""Export a FOCUS NVFP4 fake checkpoint for vLLM compressed-tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from angelslim.compressor.qat.export import export_focus_nvfp4_checkpoint  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="FOCUS fake checkpoint (.pt)")
    parser.add_argument("--model-path", required=True, help="Frozen base model or Hub ID")
    parser.add_argument("--output-path", required=True, help="Empty output directory")
    parser.add_argument(
        "--ignore-layer",
        action="append",
        dest="ignored_layers",
        help="Linear layer name/pattern kept in BF16; repeat as needed (default: lm_head)",
    )
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = export_focus_nvfp4_checkpoint(
        checkpoint_path=args.checkpoint,
        model_path=args.model_path,
        output_path=args.output_path,
        ignored_layers=args.ignored_layers,
        max_shard_size=args.max_shard_size,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
