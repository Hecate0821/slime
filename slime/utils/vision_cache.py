"""
Vision Feature Cache for VLM Training (Memory-only)

This module provides an in-memory LRU cache for vision features to avoid
redundant preprocessing of images during GRPO training where the same image
is processed multiple times (n_samples_per_prompt > 1).

Key Features:
- Simple LRU cache in memory
- Thread-safe for async operations
- Hash-based deduplication
- No external dependencies
"""

import hashlib
import io
import logging
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Singleton instance of the cache
_vision_cache: Optional["VisionCache"] = None
_cache_lock = Lock()


def get_vision_cache(max_size: int = 2000) -> "VisionCache":
    """Returns the singleton instance of the VisionCache."""
    global _vision_cache
    if _vision_cache is None:
        with _cache_lock:
            if _vision_cache is None:
                _vision_cache = VisionCache(max_size=max_size)
    return _vision_cache


def image_to_hash(image: Image.Image) -> str:
    """
    Convert PIL Image to a consistent hash string for caching.
    
    Args:
        image: PIL Image object
        
    Returns:
        SHA256 hash string of the image content
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    return hashlib.sha256(img_bytes).hexdigest()


def image_list_to_hash(images: List[Image.Image]) -> str:
    """
    Generate hash for a list of images.
    
    Args:
        images: List of PIL Image objects
        
    Returns:
        Combined SHA256 hash string
    """
    if not images:
        return "empty_image_list"
    
    combined_hash = hashlib.sha256()
    for img in images:
        img_hash = image_to_hash(img)
        combined_hash.update(img_hash.encode("utf-8"))
    
    return combined_hash.hexdigest()


class VisionCache:
    """
    An in-memory LRU cache for storing preprocessed vision features.

    This cache is designed to store the output of a HuggingFace `processor`
    (e.g., `pixel_values`, `image_grid_thw`, etc.) for given input images.
    It uses an LRU (Least Recently Used) eviction policy.
    """

    def __init__(self, max_size: int = 2000):
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("max_size must be a positive integer.")
        
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        
        logger.info(f"VisionCache initialized with max_size={max_size}")

    def _convert_to_cpu_and_numpy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Converts torch.Tensor values to CPU numpy arrays for storage."""
        converted_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                converted_data[k] = v.detach().cpu().numpy()
            elif isinstance(v, np.ndarray):
                converted_data[k] = v
            elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                converted_data[k] = [item.detach().cpu().numpy() for item in v]
            else:
                converted_data[k] = v
        return converted_data

    def _convert_to_torch_tensor(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Converts numpy arrays back to torch.Tensor for retrieval."""
        converted_data = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                converted_data[k] = torch.from_numpy(v)
            elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                converted_data[k] = [torch.from_numpy(item) for item in v]
            else:
                converted_data[k] = v
        return converted_data

    def get(self, images: List[Image.Image], image_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves preprocessed features from the cache.

        Args:
            images: The list of PIL Images to look up.
            image_hash: Precomputed hash of the image list. If None, it will be computed.

        Returns:
            The cached multimodal inputs (dict) or None if not found.
        """
        if not images:
            return None

        if image_hash is None:
            image_hash = image_list_to_hash(images)

        with self._lock:
            if image_hash in self._cache:
                self._hits += 1
                # Move to end to mark as recently used
                value = self._cache.pop(image_hash)
                self._cache[image_hash] = value
                logger.debug(f"Cache hit for hash: {image_hash[:16]}...")
                return self._convert_to_torch_tensor(value)
            else:
                self._misses += 1
                logger.debug(f"Cache miss for hash: {image_hash[:16]}...")
                return None

    def put(self, images: List[Image.Image], features: Dict[str, Any], image_hash: Optional[str] = None):
        """
        Stores preprocessed features in the cache.

        Args:
            images: The list of PIL Images associated with the features.
            features: The preprocessed multimodal inputs (dict) to store.
            image_hash: Precomputed hash of the image list. If None, it will be computed.
        """
        if not images:
            return

        if image_hash is None:
            image_hash = image_list_to_hash(images)

        with self._lock:
            if image_hash in self._cache:
                # Remove old entry to update position
                self._cache.pop(image_hash)
            elif len(self._cache) >= self.max_size:
                # Evict LRU item
                lru_key, _ = self._cache.popitem(last=False)
                logger.debug(f"Cache evicted LRU item: {lru_key[:16]}...")

            # Store on CPU as numpy arrays to save GPU memory
            self._cache[image_hash] = self._convert_to_cpu_and_numpy(features)
            logger.debug(f"Cache put for hash: {image_hash[:16]}...")

    def clear(self):
        """Clears the entire cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
        logger.info("VisionCache cleared.")

    def log_stats(self):
        """Logs cache hit/miss statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) * 100 if total > 0 else 0.0
            logger.info(
                f"VisionCache Stats: Hits={self._hits}, Misses={self._misses}, "
                f"Total={total}, Hit Rate={hit_rate:.2f}%, Size={len(self._cache)}"
            )

    @property
    def current_size(self) -> int:
        """Returns the current number of items in the cache."""
        with self._lock:
            return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Returns the current hit rate."""
        with self._lock:
            total = self._hits + self._misses
            return (self._hits / total) if total > 0 else 0.0

