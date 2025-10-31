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


from transformers import AutoModelForCausalLM
from transformers.models.deepseek_v3 import DeepseekV3Config

from ...utils import print_info
from ..model_factory import SlimModelFactory
from .deepseek import DeepSeek
from .modeling_deepseek import DeepseekV3ForCausalLM
from .tiktoken_tokenizer import TikTokenTokenizer


@SlimModelFactory.register
class KimiK2(DeepSeek):
    def __init__(
        self,
        model=None,
        deploy_backend="vllm",
    ):
        super().__init__(
            model=model,
            deploy_backend=deploy_backend,
        )

    def from_pretrained(
        self,
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
        using_multi_nodes=False,
    ):
        if torch_dtype == "fp8":
            print_info("[Slim] Loading KimiK2 with fp8")
            config = DeepseekV3Config.from_pretrained(model_path)
            if hasattr(config, "quantization_config"):
                delattr(config, "quantization_config")
            if hasattr(config, "use_cache"):
                config.use_cache = use_cache
            self.model = DeepseekV3ForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype="auto",
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=low_cpu_mem_usage,
                using_multi_nodes=using_multi_nodes,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=low_cpu_mem_usage,
                use_cache=use_cache,
            )

        # Load tokenizer
        self.tokenizer = TikTokenTokenizer.from_pretrained(model_path)
