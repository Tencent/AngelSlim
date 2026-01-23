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

"""
Unified dataset downloader for Eagle3 Colab training.

Downloads and prepares vision-language datasets (ShareGPT4V, InternVL, M3IT)
with image filtering and validation.
"""

import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Unified dataset downloader with image filtering and validation."""

    def __init__(
        self,
        output_dir: str,
        image_max_size_mb: float = 5.0,
        num_workers: int = 4,
        max_retries: int = 3,
    ):
        """
        Initialize the dataset downloader.

        Args:
            output_dir: Output directory for dataset
            image_max_size_mb: Maximum image size in MB
            num_workers: Number of parallel download workers
            max_retries: Maximum retry attempts for failed downloads
        """
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.image_max_size_bytes = int(image_max_size_mb * 1024 * 1024)
        self.num_workers = num_workers
        self.max_retries = max_retries

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Dataset downloader initialized at {self.output_dir}")
        logger.info(f"Image size limit: {image_max_size_mb} MB")

    def download_image_with_retry(
        self,
        url: str,
        output_path: str,
        timeout: int = 10,
    ) -> bool:
        """
        Download an image with retry logic.

        Args:
            url: Image URL
            output_path: Local path to save image
            timeout: Request timeout in seconds

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=timeout, stream=True)
                response.raise_for_status()

                # Check content length
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self.image_max_size_bytes:
                    logger.debug(f"Image too large: {url} ({int(content_length) / 1024 / 1024:.2f} MB)")
                    return False

                # Download and validate
                image_data = response.content

                # Check actual size
                if len(image_data) > self.image_max_size_bytes:
                    logger.debug(f"Image too large after download: {url}")
                    return False

                # Validate with PIL
                image = Image.open(io.BytesIO(image_data))
                image.verify()

                # Re-open and save (verify() closes the file)
                image = Image.open(io.BytesIO(image_data))
                image.save(output_path)

                return True

            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.debug(f"Failed to download {url} after {self.max_retries} attempts: {e}")
                    return False

                # Exponential backoff
                time.sleep(2 ** attempt)

        return False

    def download_sharegpt4v(
        self,
        num_samples: int = 50000,
        dataset_name: str = "Lin-Chen/ShareGPT4V",
        split: str = "train",
    ) -> Tuple[int, int]:
        """
        Download ShareGPT4V dataset (English).

        Args:
            num_samples: Number of samples to download
            dataset_name: HuggingFace dataset name
            split: Dataset split to use

        Returns:
            Tuple of (successful_downloads, failed_downloads)
        """
        logger.info(f"Downloading ShareGPT4V dataset: {num_samples} samples")

        try:
            # Load dataset from HuggingFace
            dataset = load_dataset(dataset_name, split=split, streaming=True)

            samples = []
            successful = 0
            failed = 0

            # Sample from dataset
            for i, example in enumerate(tqdm(dataset, total=num_samples, desc="Loading metadata")):
                if i >= num_samples:
                    break

                # Extract image URL and question
                if "image" in example:
                    image_url = example.get("image")
                elif "image_url" in example:
                    image_url = example["image_url"]
                else:
                    continue

                # Extract conversation
                conversations = example.get("conversations", [])
                if not conversations:
                    continue

                # Get first user message as question
                question = next(
                    (msg["value"] for msg in conversations if msg.get("from") == "human"),
                    "Describe this image"
                )

                samples.append({
                    "image_url": image_url,
                    "question": question,
                    "conversations": conversations,
                    "index": i,
                })

            # Download images in parallel
            logger.info(f"Downloading {len(samples)} images...")

            def download_sample(sample):
                img_path = self.images_dir / f"sharegpt4v_{sample['index']}.jpg"

                if self.download_image_with_retry(sample["image_url"], str(img_path)):
                    return {
                        "question": sample["question"],
                        "img_path": f"./images/{img_path.name}",
                        "index": sample["index"],
                    }, True
                return None, False

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(download_sample, s): s for s in samples}

                output_samples = []
                for future in tqdm(as_completed(futures), total=len(samples), desc="Downloading"):
                    result, success = future.result()
                    if success:
                        successful += 1
                        output_samples.append(result)
                    else:
                        failed += 1

            # Save to JSONL
            output_file = self.output_dir / "data_raw.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for sample in output_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            logger.info(f"ShareGPT4V download complete: {successful} successful, {failed} failed")
            logger.info(f"Saved to {output_file}")

            return successful, failed

        except Exception as e:
            logger.error(f"Failed to download ShareGPT4V: {e}")
            raise

    def download_internvl_chinese(
        self,
        num_samples: int = 50000,
        dataset_name: str = "OpenGVLab/InternVL-Chat-V1-5",
        split: str = "train",
    ) -> Tuple[int, int]:
        """
        Download InternVL Chinese dataset.

        Args:
            num_samples: Number of samples to download
            dataset_name: HuggingFace dataset name
            split: Dataset split

        Returns:
            Tuple of (successful_downloads, failed_downloads)
        """
        logger.info(f"Downloading InternVL Chinese dataset: {num_samples} samples")

        try:
            # Load dataset
            dataset = load_dataset(dataset_name, split=split, streaming=True)

            samples = []
            successful = 0
            failed = 0

            # Sample from dataset
            for i, example in enumerate(tqdm(dataset, total=num_samples, desc="Loading metadata")):
                if i >= num_samples:
                    break

                # Extract image and question
                if "image" in example:
                    image_url = example.get("image")
                elif "image_url" in example:
                    image_url = example["image_url"]
                else:
                    continue

                # Extract Chinese question
                question = example.get("question", "描述这张图片")

                samples.append({
                    "image_url": image_url,
                    "question": question,
                    "index": i,
                })

            # Download images
            logger.info(f"Downloading {len(samples)} images...")

            def download_sample(sample):
                img_path = self.images_dir / f"internvl_{sample['index']}.jpg"

                if self.download_image_with_retry(sample["image_url"], str(img_path)):
                    return {
                        "question": sample["question"],
                        "img_path": f"./images/{img_path.name}",
                        "index": sample["index"],
                    }, True
                return None, False

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(download_sample, s): s for s in samples}

                output_samples = []
                for future in tqdm(as_completed(futures), total=len(samples), desc="Downloading"):
                    result, success = future.result()
                    if success:
                        successful += 1
                        output_samples.append(result)
                    else:
                        failed += 1

            # Save to JSONL
            output_file = self.output_dir / "data_raw.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for sample in output_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            logger.info(f"InternVL download complete: {successful} successful, {failed} failed")
            logger.info(f"Saved to {output_file}")

            return successful, failed

        except Exception as e:
            logger.error(f"Failed to download InternVL: {e}")
            raise

    def download_m3it_chinese(
        self,
        num_samples: int = 50000,
        min_caption_length: int = 10,
    ) -> Tuple[int, int]:
        """
        Download M3IT Chinese subset.

        Args:
            num_samples: Number of samples to download
            min_caption_length: Minimum caption length for quality filtering

        Returns:
            Tuple of (successful_downloads, failed_downloads)
        """
        logger.info(f"Downloading M3IT Chinese dataset: {num_samples} samples")

        try:
            # Load M3IT Chinese subset
            dataset = load_dataset("MMInstruction/M3IT", "zh", split="train", streaming=True)

            samples = []
            successful = 0
            failed = 0

            # Sample dataset
            for i, example in enumerate(tqdm(dataset, total=num_samples, desc="Loading metadata")):
                if len(samples) >= num_samples:
                    break

                # Extract image and caption
                image_url = example.get("image_base_url") or example.get("image")
                caption = example.get("instruction", "")

                # Filter by caption length
                if len(caption) < min_caption_length:
                    continue

                samples.append({
                    "image_url": image_url,
                    "question": "描述这张图片",
                    "caption": caption,
                    "index": i,
                })

            # Download images
            logger.info(f"Downloading {len(samples)} images...")

            def download_sample(sample):
                img_path = self.images_dir / f"m3it_zh_{sample['index']}.jpg"

                if self.download_image_with_retry(sample["image_url"], str(img_path)):
                    return {
                        "question": sample["question"],
                        "img_path": f"./images/{img_path.name}",
                        "answer": sample["caption"],
                        "index": sample["index"],
                    }, True
                return None, False

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(download_sample, s): s for s in samples}

                output_samples = []
                for future in tqdm(as_completed(futures), total=len(samples), desc="Downloading"):
                    result, success = future.result()
                    if success:
                        successful += 1
                        output_samples.append(result)
                    else:
                        failed += 1

            # Save to JSONL
            output_file = self.output_dir / "data_raw.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for sample in output_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            logger.info(f"M3IT download complete: {successful} successful, {failed} failed")
            logger.info(f"Saved to {output_file}")

            return successful, failed

        except Exception as e:
            logger.error(f"Failed to download M3IT: {e}")
            raise

    def validate_dataset(self) -> Dict[str, int]:
        """
        Validate downloaded dataset.

        Returns:
            Dictionary with validation statistics
        """
        stats = {
            "total_samples": 0,
            "valid_images": 0,
            "invalid_images": 0,
            "missing_images": 0,
        }

        data_file = self.output_dir / "data_raw.jsonl"
        if not data_file.exists():
            logger.error(f"Dataset file not found: {data_file}")
            return stats

        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                stats["total_samples"] += 1
                sample = json.loads(line)

                img_path = self.output_dir / sample["img_path"].lstrip("./")

                if not img_path.exists():
                    stats["missing_images"] += 1
                    continue

                try:
                    Image.open(img_path).verify()
                    stats["valid_images"] += 1
                except Exception:
                    stats["invalid_images"] += 1

        logger.info("Dataset validation:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        return stats


# Convenience function
def download_dataset(
    dataset_type: str,
    output_dir: str,
    num_samples: int = 50000,
    image_max_size_mb: float = 5.0,
    num_workers: int = 4,
) -> Tuple[int, int]:
    """
    Download a dataset.

    Args:
        dataset_type: Type of dataset ("sharegpt4v", "internvl", "m3it")
        output_dir: Output directory
        num_samples: Number of samples
        image_max_size_mb: Max image size in MB
        num_workers: Number of download workers

    Returns:
        Tuple of (successful, failed)
    """
    downloader = DatasetDownloader(
        output_dir=output_dir,
        image_max_size_mb=image_max_size_mb,
        num_workers=num_workers,
    )

    if dataset_type == "sharegpt4v":
        return downloader.download_sharegpt4v(num_samples=num_samples)
    elif dataset_type == "internvl":
        return downloader.download_internvl_chinese(num_samples=num_samples)
    elif dataset_type == "m3it":
        return downloader.download_m3it_chinese(num_samples=num_samples)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
