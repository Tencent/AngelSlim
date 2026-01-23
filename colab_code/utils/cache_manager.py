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
Google Drive cache manager for Colab Eagle3 training.

Manages downloading, caching, and cleanup of models, datasets, and checkpoints
in Google Drive to avoid re-downloading between Colab sessions.
"""

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from huggingface_hub import snapshot_download
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DriveCache_Manager:
    """Manages model and data caching in Google Drive for Colab."""

    def __init__(self, drive_root: str = "/content/drive/MyDrive/Eagle3_Qwen3VL"):
        """
        Initialize the cache manager.

        Args:
            drive_root: Root directory in Google Drive for caching
        """
        self.drive_root = Path(drive_root)
        self.models_dir = self.drive_root / "models"
        self.datasets_dir = self.drive_root / "datasets"
        self.checkpoints_dir = self.drive_root / "checkpoints"
        self.hidden_states_dir = self.drive_root / "hidden_states"
        self.logs_dir = self.drive_root / "logs"

        # Create directories
        for dir_path in [
            self.models_dir,
            self.datasets_dir,
            self.checkpoints_dir,
            self.hidden_states_dir,
            self.logs_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Cache manager initialized at {self.drive_root}")

    def download_and_cache_model(
        self,
        model_name_or_path: str,
        cache_name: Optional[str] = None,
        force_download: bool = False,
    ) -> str:
        """
        Download a model from HuggingFace Hub and cache it in Drive.

        Args:
            model_name_or_path: HuggingFace model name (e.g., "Qwen/Qwen3-VL-30B-A3B")
            cache_name: Custom cache directory name (default: use model name)
            force_download: Force re-download even if cached

        Returns:
            Path to the cached model directory
        """
        if cache_name is None:
            cache_name = model_name_or_path.replace("/", "_")

        cache_path = self.models_dir / cache_name

        # Check if already cached
        if cache_path.exists() and not force_download:
            if self.verify_cache(str(cache_path)):
                logger.info(f"Model already cached at {cache_path}")
                return str(cache_path)
            else:
                logger.warning("Cached model is corrupted, re-downloading...")
                shutil.rmtree(cache_path)

        # Download model
        logger.info(f"Downloading {model_name_or_path} to {cache_path}...")
        try:
            downloaded_path = snapshot_download(
                repo_id=model_name_or_path,
                cache_dir=str(self.models_dir / ".cache"),
                local_dir=str(cache_path),
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            # Create metadata file
            self._save_metadata(cache_path, model_name_or_path)

            logger.info(f"Model cached successfully at {cache_path}")
            return str(cache_path)

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    def load_from_cache(self, cache_name: str) -> Optional[str]:
        """
        Load a model from cache.

        Args:
            cache_name: Name of the cached model directory

        Returns:
            Path to cached model or None if not found
        """
        cache_path = self.models_dir / cache_name

        if not cache_path.exists():
            logger.error(f"Model not found in cache: {cache_name}")
            return None

        if not self.verify_cache(str(cache_path)):
            logger.error(f"Cached model is corrupted: {cache_name}")
            return None

        logger.info(f"Loaded model from cache: {cache_path}")
        return str(cache_path)

    def verify_cache(self, cache_path: str) -> bool:
        """
        Verify the integrity of a cached model.

        Args:
            cache_path: Path to the cached model directory

        Returns:
            True if cache is valid, False otherwise
        """
        cache_path = Path(cache_path)

        # Check if config.json exists (basic validation)
        config_file = cache_path / "config.json"
        if not config_file.exists():
            logger.warning(f"Missing config.json in {cache_path}")
            return False

        # Check if model weights exist
        model_files = list(cache_path.glob("*.safetensors")) + list(
            cache_path.glob("*.bin")
        )
        if not model_files:
            logger.warning(f"No model weights found in {cache_path}")
            return False

        return True

    def _save_metadata(self, cache_path: Path, model_name: str) -> None:
        """Save metadata about the cached model."""
        metadata = {
            "model_name": model_name,
            "cache_path": str(cache_path),
        }

        metadata_file = cache_path / ".cache_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def cleanup_old_checkpoints(
        self,
        checkpoint_dir: str,
        keep_last: int = 3,
    ) -> List[str]:
        """
        Clean up old checkpoints, keeping only the most recent ones.

        Args:
            checkpoint_dir: Directory containing checkpoints
            keep_last: Number of most recent checkpoints to keep

        Returns:
            List of removed checkpoint directories
        """
        checkpoint_dir = Path(checkpoint_dir)

        if not checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            return []

        # Find all checkpoint directories
        checkpoints = sorted(
            [d for d in checkpoint_dir.glob("checkpoint-*") if d.is_dir()],
            key=lambda x: int(x.name.split("-")[1]),
        )

        if len(checkpoints) <= keep_last:
            logger.info(f"Only {len(checkpoints)} checkpoints, nothing to clean up")
            return []

        # Remove old checkpoints
        to_remove = checkpoints[:-keep_last]
        removed = []

        for ckpt in to_remove:
            try:
                shutil.rmtree(ckpt)
                removed.append(str(ckpt))
                logger.info(f"Removed old checkpoint: {ckpt}")
            except Exception as e:
                logger.error(f"Failed to remove {ckpt}: {e}")

        logger.info(f"Cleaned up {len(removed)} old checkpoints")
        return removed

    def estimate_storage(
        self,
        include_models: bool = True,
        include_datasets: bool = True,
        include_checkpoints: bool = True,
        include_hidden_states: bool = True,
    ) -> Dict[str, float]:
        """
        Estimate storage usage in Google Drive.

        Args:
            include_models: Include model storage
            include_datasets: Include dataset storage
            include_checkpoints: Include checkpoint storage
            include_hidden_states: Include hidden states storage

        Returns:
            Dictionary with storage estimates in GB
        """
        storage = {}

        def get_dir_size(path: Path) -> float:
            """Get directory size in GB."""
            if not path.exists():
                return 0.0

            total_size = sum(
                f.stat().st_size for f in path.rglob("*") if f.is_file()
            )
            return total_size / (1024**3)  # Convert to GB

        if include_models:
            storage["models"] = get_dir_size(self.models_dir)

        if include_datasets:
            storage["datasets"] = get_dir_size(self.datasets_dir)

        if include_checkpoints:
            storage["checkpoints"] = get_dir_size(self.checkpoints_dir)

        if include_hidden_states:
            storage["hidden_states"] = get_dir_size(self.hidden_states_dir)

        storage["total"] = sum(storage.values())

        # Log storage info
        logger.info("Storage usage:")
        for key, size in storage.items():
            logger.info(f"  {key}: {size:.2f} GB")

        return storage

    def copy_to_drive(
        self,
        source_path: str,
        dest_rel_path: str,
        category: str = "checkpoints",
    ) -> str:
        """
        Copy a file or directory to Google Drive cache.

        Args:
            source_path: Source file or directory path
            dest_rel_path: Relative path within category directory
            category: Category (models, datasets, checkpoints, hidden_states)

        Returns:
            Destination path in Drive
        """
        category_dir = getattr(self, f"{category}_dir")
        dest_path = category_dir / dest_rel_path

        # Create parent directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy
        source_path = Path(source_path)
        if source_path.is_dir():
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(source_path, dest_path)
        else:
            shutil.copy2(source_path, dest_path)

        logger.info(f"Copied {source_path} to {dest_path}")
        return str(dest_path)

    def get_dataset_path(self, dataset_name: str) -> Path:
        """Get the path to a dataset directory."""
        return self.datasets_dir / dataset_name

    def get_checkpoint_path(self, checkpoint_name: str) -> Path:
        """Get the path to a checkpoint directory."""
        return self.checkpoints_dir / checkpoint_name

    def get_hidden_states_path(self, split: str = "train") -> Path:
        """Get the path to hidden states directory."""
        return self.hidden_states_dir / split

    def sync_checkpoints_async(
        self,
        local_checkpoint_dir: str,
        drive_checkpoint_dir: Optional[str] = None,
    ) -> None:
        """
        Asynchronously sync checkpoints from local to Drive.

        Args:
            local_checkpoint_dir: Local checkpoint directory
            drive_checkpoint_dir: Drive checkpoint directory (default: same name)
        """
        local_checkpoint_dir = Path(local_checkpoint_dir)

        if drive_checkpoint_dir is None:
            drive_checkpoint_dir = local_checkpoint_dir.name

        drive_path = self.checkpoints_dir / drive_checkpoint_dir

        # Get all checkpoint directories
        checkpoints = sorted(
            [d for d in local_checkpoint_dir.glob("checkpoint-*") if d.is_dir()],
            key=lambda x: int(x.name.split("-")[1]),
        )

        for ckpt in checkpoints:
            dest_ckpt = drive_path / ckpt.name

            # Skip if already synced
            if dest_ckpt.exists():
                continue

            try:
                logger.info(f"Syncing {ckpt.name} to Drive...")
                shutil.copytree(ckpt, dest_ckpt)
                logger.info(f"Synced {ckpt.name} successfully")
            except Exception as e:
                logger.error(f"Failed to sync {ckpt.name}: {e}")


# Convenience functions for Colab
def ensure_drive_mounted(drive_path: str = "/content/drive") -> bool:
    """
    Ensure Google Drive is mounted in Colab.

    Args:
        drive_path: Expected mount point for Google Drive

    Returns:
        True if mounted, False otherwise
    """
    drive_path = Path(drive_path)

    if not drive_path.exists():
        logger.error("Google Drive is not mounted!")
        logger.info("Please run: from google.colab import drive; drive.mount('/content/drive')")
        return False

    logger.info("Google Drive is mounted")
    return True


def get_cache_manager(
    drive_root: str = "/content/drive/MyDrive/Eagle3_Qwen3VL",
) -> DriveCacheManager:
    """
    Get a cache manager instance, ensuring Drive is mounted.

    Args:
        drive_root: Root directory for caching

    Returns:
        DriveCacheManager instance

    Raises:
        RuntimeError: If Drive is not mounted
    """
    if not ensure_drive_mounted():
        raise RuntimeError("Google Drive must be mounted to use cache manager")

    return DriveCacheManager(drive_root)
