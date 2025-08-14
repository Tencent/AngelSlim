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

import argparse
import os

from angelslim.compressor.quant.core.fp8_analyse_tools import (
    draw_bf16_fp8_weight_fig,
    draw_fp8_scale_fig,
)


def quant_analyse(args):
    if args.analyse_type == "act":
        os.makedirs(args.save_path, exist_ok=True)
        assert os.path.exists(args.model_path), f"File {args.model_path} not exist"
        print(f"[AngelSlim] Save all quant scale graph to {args.save_path}")
        draw_fp8_scale_fig(args.model_path, args.save_path)
    elif args.analyse_type == "weight":
        print(f"[AngelSlim] Save weight analyse graph to {args.save_path}")
        os.makedirs(args.save_path, exist_ok=True)
        assert os.path.exists(args.bf16_path), f"File {args.bf16_path} not exist"
        assert os.path.exists(args.fp8_path), f"File {args.fp8_path} not exist"
        bf16_path = args.bf16_path
        fp8_path = args.fp8_path
        layer_index = args.layer_index
        draw_bf16_fp8_weight_fig(
            bf16_path=bf16_path,
            fp8_path=fp8_path,
            save_path=args.save_path,
            layer_index=layer_index,
        )


if __name__ == "__main__":

    global_parser = argparse.ArgumentParser(
        description="Quantization analyse", add_help=True
    )
    global_parser.add_argument(
        "--analyse-type",
        type=str,
        required=True,
        choices=["act", "weight"],
        help="choice 'activation', 'weight'",
    )
    global_args, remaining_args = global_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description=f"branch {global_args.analyse_type} args"
    )
    if global_args.analyse_type == "act":
        parser.add_argument("--model-path", type=str, help="Fp8 path", required=True)
        parser.add_argument("--save-path", type=str, default="./Quant_Scale_SavePath")
    elif global_args.analyse_type == "weight":
        parser.add_argument("--bf16-path", type=str, help="Bf16 path", required=True)
        parser.add_argument("--fp8-path", type=str, help="Fp8 path", required=True)
        parser.add_argument("--save-path", type=str, default="./Weight_analyse")
        parser.add_argument("--layer-index", type=int, required=True)

    args = parser.parse_args(remaining_args)
    args.analyse_type = global_args.analyse_type
    print(f"Args:{args}")
    quant_analyse(args)
