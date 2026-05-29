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

"""Shared YAML-config loader for standalone tool scripts.

The Hy3 tool entry points (``tools/run_vllm_calibrate.py``,
``tools/kvcache/run_kvcache_calibrate.py`` and
``tools/fp8_quant_with_vllm_activation.py``) all use ``argparse``.  To match
the style of ``scripts/ptq/run_vllm_quant_for_deepseek_v3.sh`` (shell only
exports environment variables and passes ``-c CONFIG`` to Python), this
helper merges a YAML config file into the argparse ``args`` namespace.

Precedence (highest to lowest):
    1. explicit command-line flags
    2. YAML config values
    3. argparse defaults

Usage in a tool script::

    parser.add_argument("-c", "--config", type=str, default=None,
                        help="YAML config file path. Values override "
                             "argparse defaults; explicit CLI flags still "
                             "take final precedence.")
    args = parser.parse_args()
    apply_yaml_config(parser, args)
"""

import os
import sys

import yaml


def _collect_explicit_cli_dests(parser):
    """Return the set of argparse ``dest`` names actually provided on argv.

    We scan ``sys.argv`` looking for any token that matches an option string
    declared on ``parser`` (full match for ``--foo``/``-f``, or prefix match
    for ``--foo=...``).  This avoids cloning the parser and works uniformly
    for ``store_true`` / ``store_false`` / ``store`` / ``store_const``
    actions which otherwise need different ``nargs`` handling.
    """
    # Map every option_string -> dest, e.g. {"--tp-size": "tp_size", ...}
    option_to_dest = {}
    for action in parser._actions:
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest

    explicit = set()
    for token in sys.argv[1:]:
        if not token.startswith("-"):
            continue
        # Split off ``=value`` form (``--foo=bar``).
        head = token.split("=", 1)[0]
        if head in option_to_dest:
            explicit.add(option_to_dest[head])
    return explicit


def apply_yaml_config(parser, args):
    """Merge ``args.config`` (a YAML file) into ``args`` in place.

    YAML values override argparse defaults; values that the user passed
    explicitly on the command line still win over the YAML file.

    The YAML file is expected to be a flat mapping whose keys match the
    argparse ``dest`` names (underscore form) of the target script.  Unknown
    keys are reported and ignored so that one shared YAML schema can be
    extended without breaking older scripts.
    """
    config_path = getattr(args, "config", None)
    if not config_path:
        return args

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"YAML config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Top level of YAML config '{config_path}' must be a mapping, "
            f"got {type(cfg).__name__}."
        )

    valid_dests = {
        action.dest
        for action in parser._actions
        if action.option_strings  # ignore positionals / help
    }
    explicit_cli = _collect_explicit_cli_dests(parser)

    overridden = []
    unknown = []
    for key, value in cfg.items():
        dest = key.replace("-", "_")
        if dest not in valid_dests:
            unknown.append(key)
            continue
        if dest in explicit_cli:
            # CLI wins – don't override.
            continue
        setattr(args, dest, value)
        overridden.append((dest, value))

    if overridden:
        print(f"[yaml-config] Loaded {config_path}; applied overrides:")
        for dest, value in overridden:
            print(f"  {dest} = {value!r}")
    if unknown:
        print(f"[yaml-config] WARNING: unknown keys in {config_path} " f"(ignored): {unknown}")

    return args
