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
import sys

import yaml


def _collect_explicit_cli_dests(parser):
    """argparse ``dest`` names given on ``sys.argv`` (``--foo`` or ``--foo=bar``)."""
    option_to_dest = {}
    for action in parser._actions:
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest

    explicit = set()
    for token in sys.argv[1:]:
        if not token.startswith("-"):
            continue
        head = token.split("=", 1)[0]
        if head in option_to_dest:
            explicit.add(option_to_dest[head])
    return explicit


def apply_yaml_config(parser, args):
    """Merge the flat YAML ``args.config`` (keys = argparse dests) into ``args`` in place.
    Precedence: explicit CLI flag > YAML > argparse default; unknown keys are reported."""
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
