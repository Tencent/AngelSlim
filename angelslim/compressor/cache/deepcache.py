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

__all__ = ["FluxDeepCacheHelper", "HunyuanDitDeepCacheHelper"]


class DeepCacheHelper(object):
    def __init__(
        self,
        pipe_model=None,
        timesteps=None,
        no_cache_steps=None,
        no_cache_block_id=None,
        no_cache_layer_id=None,
    ):
        if pipe_model is not None:
            self.pipe_model = pipe_model
        if timesteps is not None:
            self.timesteps = timesteps
        if no_cache_steps is not None:
            self.no_cache_steps = no_cache_steps
        if no_cache_block_id is not None:
            self.no_cache_block_id = no_cache_block_id
        if no_cache_layer_id is not None:
            self.no_cache_layer_id = no_cache_layer_id
        self.set_default_blocktypes()
        self.set_model_type()

    def set_default_blocktypes(self, default_blocktypes=None):
        self.default_blocktypes = default_blocktypes

    def set_model_type(self, model_type="flux"):
        self.model_type = model_type

    def enable(self):
        assert self.pipe_model is not None
        self.reset_states()
        self.wrap_modules()

    def disable(self):
        self.unwrap_modules()
        self.reset_states()

    def is_skip_step(self, block_i, layer_i, blocktype):
        self.start_timestep = (
            self.cur_timestep if self.start_timestep is None else self.start_timestep
        )  # For some pipeline that the first timestep != 0

        if self.cur_timestep - self.start_timestep in self.no_cache_steps:
            return False
        if blocktype in self.default_blocktypes:
            if block_i in self.no_cache_block_id[blocktype]:
                return False
            else:
                return True
        return True

    def wrap_model_forward(self):
        pass

    def wrap_block_forward(self, block, block_name, block_i, layer_i, blocktype):
        self.function_dict[(blocktype, block_name, block_i, layer_i)] = block.forward

        def wrapped_forward(*args, **kwargs):
            skip = self.is_skip_step(block_i, layer_i, blocktype)
            result = (
                self.cached_output[(blocktype, block_name, block_i, layer_i)]
                if skip
                else self.function_dict[(blocktype, block_name, block_i, layer_i)](
                    *args, **kwargs
                )
            )
            if not skip:
                self.cached_output[(blocktype, block_name, block_i, layer_i)] = result
            return result

        block.forward = wrapped_forward

    def wrap_modules(self):
        pass

    def unwrap_modules(self):
        pass

    def reset_states(self):
        self.cur_timestep = 0
        self.function_dict = {}
        self.cached_output = {}
        self.start_timestep = None


class FluxDeepCacheHelper(DeepCacheHelper):
    def set_default_blocktypes(self, default_blocktypes=None):
        self.default_blocktypes = ["single"]

    def wrap_model_forward(self):
        self.function_dict["model_forward"] = self.pipe_model.forward

        def wrapped_forward(*args, **kwargs):
            result = self.function_dict["model_forward"](*args, **kwargs)
            return result

        self.pipe_model.forward = wrapped_forward

    def wrap_modules(self):
        # 1. wrap flux forward
        self.wrap_model_forward()
        # 2. wrap double forward
        for block_i, block in enumerate(self.pipe_model.transformer_blocks):
            self.wrap_block_forward(block, "block", block_i, 0, blocktype="double")

        # 3. wrap single forward
        for block_i, block in enumerate(self.pipe_model.single_transformer_blocks):
            self.wrap_block_forward(block, "block", block_i, 0, blocktype="single")

    def unwrap_modules(self):
        # 1. model forward
        self.pipe_model.forward = self.function_dict["model_forward"]
        # 2. block forward
        for block_i, block in enumerate(self.pipe_model.transformer_blocks):
            block.forward = self.function_dict[("double", "block", block_i, 0)]

        # 3. single block forward
        block_num = len(self.pipe_model.single_transformer_blocks)
        for block_i, block in enumerate(self.pipe_model.single_transformer_blocks):
            block.forward = self.function_dict[
                ("single", "block", block_num - block_i - 1, 0)
            ]


class HunyuanDitDeepCacheHelper(DeepCacheHelper):
    def set_default_blocktypes(self, default_blocktypes=None):
        self.default_blocktypes = ["no_skip_block", "skip_block"]

    def set_model_type(self, model_type="hunyuandit"):
        self.model_type = model_type

    def wrap_model_forward(self):
        self.function_dict["model_forward"] = self.pipe_model.forward

        def wrapped_forward(*args, **kwargs):
            result = self.function_dict["model_forward"](*args, **kwargs)
            return result

        self.pipe_model.forward = wrapped_forward

    def wrap_modules(self):
        # 1. wrap model forward
        self.wrap_model_forward()
        # 2. wrap block forward
        block_num = len(self.pipe_model.blocks)
        for block_i, block in enumerate(self.pipe_model.blocks):
            if block_i <= block_num // 2:
                self.wrap_block_forward(
                    block, "block", block_i, 0, blocktype="no_skip_block"
                )
            else:
                self.wrap_block_forward(
                    block, "block", block_num - block_i - 1, 0, blocktype="skip_block"
                )

    def unwrap_modules(self):
        # 1. model forward
        self.pipe_model.forward = self.function_dict["model_forward"]
        # 2. block forward
        block_num = len(self.pipe_model.blocks)
        for block_i, block in enumerate(self.pipe_model.blocks):
            if block_i <= block_num // 2:
                block.forward = self.function_dict[
                    ("no_skip_block", "block", block_i, 0)
                ]
            else:
                block.forward = self.function_dict[
                    ("skip_block", "block", block_num - block_i - 1, 0)
                ]
