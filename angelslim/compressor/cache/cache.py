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

from ..compressor_factory import CompressorFactory
from .deepcache import FluxDeepCacheHelper
from .teacache import TeaCache

__all__ = ["Cache"]


@CompressorFactory.register
class Cache:
    def __init__(self, model, slim_config=None):
        """
        Diffusion Cache class for caching model layers during inference.
        Args:
            model(nn.Module, required): the model to be cached.
            slim_config(dict, required): the configuration for caching.
                - compress_config: the configuration for compression.
                - global_config: the global configuration for the model.
        """
        self.cache_model = model
        # init cache config of model
        self.init_cache(slim_config["compress_config"])

    def init_cache(self, slim_config):
        """Initialize the cache for the FLUX model.
        Args:
            slim_config (dict): Configuration for the cache.
        """
        cache_config = slim_config.cache
        if cache_config.name == "DeepCache":
            if self.cache_model.model_type == "flux":
                if cache_config.use_cache_helper:
                    self.cache_model.cache_helper = FluxDeepCacheHelper(
                        pipe_model=self.cache_model,
                        no_cache_steps=cache_config.no_cache_steps,
                        no_cache_block_id=cache_config.no_cache_block_id,
                    )
        elif cache_config.name == "TeaCache":
            if not cache_config.use_cache_helper:
                l1_distance = cache_config.accumulated_rel_l1_distance
                self.cache_module = TeaCache(
                    self.cache_model.model,
                    model_type=self.cache_model.model_type,
                    cnt=cache_config.cnt,
                    num_steps=cache_config.num_steps,
                    rel_l1_thresh=cache_config.rel_l1_thresh,
                    accumulated_rel_l1_distance=l1_distance,
                )
        else:
            raise ValueError(f"Unsupported cache method: {cache_config.name}. ")
