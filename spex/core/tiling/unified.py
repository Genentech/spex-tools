"""
Unified tiling interface for SPEX segmentation methods.

This module provides a single, consistent interface for tiling-based processing
of large images, implementing DRY, KISS, and SOLID principles.
"""

import numpy as np
import time
from typing import Tuple, Optional, Callable, Any, Dict, Union
from .core import compute_tiles, crop_core_safe, get_core_coordinates_safe


class TilingProcessor:
    """
    Unified processor for tiling-based segmentation of large images.

    This class implements the Single Responsibility Principle by focusing
    solely on tiling coordination and orchestration.
    """

    def __init__(
        self,
        memory_limit_mb: int = 30,
        overlap_percent: float = 20.0,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Initialize the tiling processor.

        Args:
            memory_limit_mb: Maximum memory per tile in MB
            overlap_percent: Overlap between tiles as percentage of tile size
            progress_callback: Optional callback for progress updates
        """
        self.memory_limit_mb = memory_limit_mb
        self.overlap_percent = overlap_percent
        self.progress_callback = progress_callback

    def should_use_tiling(
        self,
        img: np.ndarray,
        memory_threshold_mb: float = 500.0
    ) -> bool:
        """
        Determine if tiling is needed for the given image.

        Args:
            img: Input image array
            memory_threshold_mb: Threshold for automatic tiling

        Returns:
            True if tiling should be used
        """
        estimated_memory_mb = self._estimate_memory_usage(img)
        return estimated_memory_mb > memory_threshold_mb

    def calculate_optimal_tile_size(
        self,
        img_shape: Tuple[int, ...],
        target_memory_mb: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Calculate optimal tile size for the given image shape.

        Args:
            img_shape: Shape of the image (C, H, W)
            target_memory_mb: Target memory per tile (defaults to instance limit)

        Returns:
            Optimal tile size (height, width)
        """
        if target_memory_mb is None:
            target_memory_mb = self.memory_limit_mb

        channels = img_shape[0] if len(img_shape) == 3 else 1
        height, width = img_shape[-2:]

        # Assume uint8 data (1 byte per pixel)
        bytes_per_pixel = 1
        target_bytes = target_memory_mb * 1024 * 1024
        pixels_per_tile = target_bytes // (channels * bytes_per_pixel)

        # Calculate square tile dimension
        tile_dim = int(np.sqrt(pixels_per_tile))

        # Ensure tile doesn't exceed image dimensions
        tile_h = min(tile_dim, height)
        tile_w = min(tile_dim, width)

        return (tile_h, tile_w)

    def calculate_overlap(self, tile_size: Tuple[int, int]) -> int:
        """
        Calculate overlap in pixels based on tile size.

        Args:
            tile_size: Size of tiles (height, width)

        Returns:
            Overlap in pixels
        """
        min_tile_dim = min(tile_size)
        overlap = max(32, int(min_tile_dim * self.overlap_percent / 100))
        return overlap

    def process_with_tiling(
        self,
        segmentation_func: Callable,
        img: np.ndarray,
        *args,
        tile_size: Optional[Tuple[int, int]] = None,
        overlap: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Process image using tiling with the specified segmentation function.

        This method implements the Open/Closed Principle by accepting any
        segmentation function without modification.

        Args:
            segmentation_func: Function to apply to each tile
            img: Input image array
            *args: Positional arguments for segmentation function
            tile_size: Size of tiles (auto-calculated if None)
            overlap: Overlap in pixels (auto-calculated if None)
            **kwargs: Keyword arguments for segmentation function

        Returns:
            Segmented image as label array
        """
        # Auto-calculate parameters if not provided
        if tile_size is None:
            tile_size = self.calculate_optimal_tile_size(img.shape)

        if overlap is None:
            overlap = self.calculate_overlap(tile_size)

        # Validate configuration
        self._validate_tiling_config(img.shape, tile_size, overlap)

        # Compute tile coordinates
        image_shape = img.shape[-2:]  # Get spatial dimensions
        tiles = compute_tiles(image_shape, tile_size, overlap)

        print(f"🧩 Unified tiling: {tile_size} tiles, {overlap}px overlap")
        print(f"📊 Processing {len(tiles)} tiles...")

        # Initialize result array
        result_shape = image_shape
        labels_final = np.zeros(result_shape, dtype=np.int32)

        # Global label counter for unique labels across tiles
        global_label_counter = 1

        # Process each tile
        for tile_idx, tile in enumerate(tiles):
            if self.progress_callback:
                self.progress_callback(tile_idx, len(tiles))

            print(f"  🔸 Processing tile {tile_idx + 1}/{len(tiles)}")

            # Extract tile data
            y_slice, x_slice = tile
            tile_data = img[:, y_slice, x_slice] if len(img.shape) == 3 else img[y_slice, x_slice]

            # Apply segmentation function
            tile_labels = segmentation_func(tile_data, *args, **kwargs)

            # Relabel to avoid conflicts
            tile_labels_relabeled = self._relabel_tile(tile_labels, global_label_counter)
            global_label_counter = tile_labels_relabeled.max() + 1

            # Extract core region
            core = crop_core_safe(
                tile_labels_relabeled,
                overlap,
                tile,
                image_shape,
                tile_size
            )

            # Get placement coordinates
            y_start, x_start = get_core_coordinates_safe(
                tile,
                overlap,
                image_shape,
                tile_size
            )

            # Place core in result
            core_h, core_w = core.shape
            labels_final[y_start:y_start+core_h, x_start:x_start+core_w] = core

        total_labels = len(np.unique(labels_final)) - 1
        print(f"✅ Unified tiling completed. Labels found: {total_labels}")

        return labels_final

    def _estimate_memory_usage(self, img: np.ndarray) -> float:
        """Estimate memory usage for the image in MB."""
        return img.nbytes / (1024 * 1024)

    def _validate_tiling_config(
        self,
        img_shape: Tuple[int, ...],
        tile_size: Tuple[int, int],
        overlap: int
    ) -> None:
        """
        Validate tiling configuration parameters.

        Args:
            img_shape: Shape of the image
            tile_size: Size of tiles
            overlap: Overlap in pixels

        Raises:
            ValueError: If configuration is invalid
        """
        if overlap < 0:
            raise ValueError(f"Overlap must be non-negative, got {overlap}")

        if tile_size[0] <= overlap * 2 or tile_size[1] <= overlap * 2:
            raise ValueError(f"Tile size {tile_size} too small for overlap {overlap}")

        # Check if tile size is reasonable for image
        height, width = img_shape[-2:]
        if tile_size[0] > height or tile_size[1] > width:
            raise ValueError(f"Tile size {tile_size} larger than image {(height, width)}")

    def _relabel_tile(self, labels: np.ndarray, start_label: int) -> np.ndarray:
        """
        Relabel tile to avoid label conflicts across tiles.

        Args:
            labels: Label array from tile processing
            start_label: Starting label value for this tile

        Returns:
            Relabeled array with unique labels
        """
        if labels.max() == 0:
            return labels  # No labels found

        # Create mapping from old to new labels
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background

        relabeled = labels.copy()
        for i, old_label in enumerate(unique_labels):
            new_label = start_label + i
            relabeled[labels == old_label] = new_label

        return relabeled


def create_tiling_processor(
    memory_limit_mb: int = 30,
    overlap_percent: float = 20.0
) -> TilingProcessor:
    """
    Factory function to create a TilingProcessor instance.

    This function implements the Dependency Inversion Principle by providing
    a factory for creating processor instances.

    Args:
        memory_limit_mb: Maximum memory per tile in MB
        overlap_percent: Overlap between tiles as percentage

    Returns:
        Configured TilingProcessor instance
    """
    return TilingProcessor(memory_limit_mb, overlap_percent)


def should_use_tiling_for_image(
    img: np.ndarray,
    memory_threshold_mb: float = 500.0
) -> bool:
    """
    Convenience function to check if tiling should be used for an image.

    Args:
        img: Input image array
        memory_threshold_mb: Memory threshold for tiling decision

    Returns:
        True if tiling is recommended
    """
    processor = create_tiling_processor()
    return processor.should_use_tiling(img, memory_threshold_mb)