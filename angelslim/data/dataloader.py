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
from typing import Dict, Union

from torch.utils.data import DataLoader
from transformers import ProcessorMixin

from .base_dataset import BaseDataset
from .multimodal_dataset import MultiModalDataset
from .omni_dataset import OmniDataset
from .text2image_dataset import Text2ImageDataset
from .text_dataset import TextDataset


class DataLoaderFactory:
    """Factory for creating PyTorch DataLoaders from various data sources"""

    @staticmethod
    def create_data_loader(
        processor: ProcessorMixin,
        device: str = "cpu",
        max_length: int = 4096,
        batch_size: int = 1,
        shuffle: bool = True,
        num_samples: int = -1,
        data_source: Union[str, Dict] = None,
        data_type: str = "auto",
        num_workers: int = 0,
        inference_settings: Dict = None,
        use_audio_in_video: bool = False,
        model_name: str = None,
    ) -> DataLoader:
        """
        Create appropriate DataLoader based on data source

        Args:
            processor: Text/vision processor
            device: Target device for tensors
            max_length: Maximum sequence length
            batch_size: DataLoader batch size
            shuffle: Whether to shuffle data
            num_samples: Limit number of samples (-1 for all)
            data_source: File path or HF dataset dict
            data_type: "text", "multimodal" or "auto"
            num_workers: Number of workers for DataLoader
            inference_settings: Settings for text-to-image inference
                - height: Image height
                - width: Image width
                - guidance_scale: Guidance scale for inference
                - num_inference_steps: Number of inference steps
                - max_sequence_length: Maximum sequence length for text inputs

        Returns:
            PyTorch DataLoader ready for use
        """
        # Auto detect data type if not specified
        if data_type == "auto":
            if isinstance(data_source, str) and (
                ".parquet" in data_source.lower() or ".json" in data_source
            ):
                data_type = "text"
            else:
                data_type = "multimodal"

        # Create appropriate dataset
        if data_type == "TextDataset":
            dataset = TextDataset(
                processor=processor,
                device=device,
                max_length=max_length,
                data_path=data_source,
                num_samples=num_samples,
            )
        elif data_type == "MultiModalDataset":
            dataset = MultiModalDataset(
                processor=processor,
                device=device,
                max_length=max_length,
                num_samples=num_samples,
                data_source=data_source,
                is_hf_dataset=not os.path.isfile(data_source),
                model_name=model_name,
            )
        elif data_type == "Text2ImageDataset":
            dataset = Text2ImageDataset(
                data_path=data_source,
                num_samples=num_samples,
                inference_settings=inference_settings,
            )
        elif data_type == "OmniDataset":
            dataset = OmniDataset(
                processor=processor,
                device=device,
                max_length=max_length,
                num_samples=num_samples,
                data_source=data_source,
                is_hf_dataset=not os.path.isfile(data_source),
                use_audio_in_video=use_audio_in_video,
            )
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        # Create DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=BaseDataset.collate_fn,
            num_workers=num_workers,
            # pin_memory = device.type == 'cuda'
        )
