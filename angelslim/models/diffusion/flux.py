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

import torch
from diffusers import FluxPipeline

from ..base_model import BaseDiffusionModel
from ..model_factory import SlimModelFactory


@SlimModelFactory.register
class FLUX(BaseDiffusionModel):
    def __init__(
        self,
        model=None,
        deploy_backend="torch",
    ):
        super().__init__(
            model=model,
            deploy_backend=deploy_backend,
        )
        self.model_type = "flux"
        self.cache_helper = None

    def from_pretrained(
        self,
        model_path,
        torch_dtype="auto",
        cache_dir=None,
        use_cache_helper=False,
    ):
        """
        Load a pretrained FLUX model.
        Args:
            model_path (str): Path to the pretrained model.
            torch_dtype (str): Data type for the model weights.
            cache_dir (str): Directory to cache the model.
        """
        self.model = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
        if use_cache_helper:
            self.model.cache_helper = self.cache_helper

    def generate(
        self,
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        seed=42,
    ):
        """
        Generate images using the FLUX model.
        Args:
            prompt (list): List of text prompt for image generation.
            height (int): Height of the generated images.
            width (int): Width of the generated images.
            guidance_scale (float): Guidance scale for the generation.
            num_inference_steps (int): Number of inference steps.
            max_sequence_length (int): Maximum sequence length for the model.
            seed (int): Random number torch.Generator for reproducibility.
        Returns:
            Generated image tensor.
        """
        generator = torch.Generator().manual_seed(seed)
        return self.model(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator,
        ).images[0]

    def get_observer_layers(self):
        pass

    def get_save_func(self):
        pass
