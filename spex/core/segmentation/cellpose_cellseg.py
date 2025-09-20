"""Cellpose segmentation helpers with unified tiling support."""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from ..tiling.core import _estimate_memory_usage
from ..tiling.unified import should_use_tiling_for_image, create_tiling_processor

DEFAULT_OVERLAP_RATIO = 0.125
DEFAULT_MIN_OVERLAP = 32


def cellpose_cellseg(
    img: np.ndarray,
    seg_channels: Sequence[int],
    diameter: int,
    scaling: int,
    num_tiles: Optional[int] = None,
    overlap: Optional[int] = None,
    auto_tile_memory_mb: Optional[float] = 500,
    tile_size: Optional[Tuple[int, int]] = None,
    auto_tiling: bool = True,
) -> np.ndarray:
    """Segment an image using Cellpose with automatic tiling for large images.

    This function automatically determines whether to use tiling based on image size
    and memory requirements, following the KISS principle for user simplicity.

    Args:
        img: Multichannel image as ``(C, H, W)`` numpy array.
        seg_channels: Channel indices used for segmentation.
        diameter: Typical object diameter provided to Cellpose.
        scaling: Integer scaling factor prior to inference.
        num_tiles: Deprecated tile-count hint kept for backwards compatibility.
        overlap: Optional overlap (pixels) to use between tiles.
        auto_tile_memory_mb: Trigger tiling when estimated memory exceeds this threshold.
        tile_size: Explicit tile size ``(height, width)`` for tiled processing.
        auto_tiling: Enable automatic tiling for large images (default: True).

    Returns:
        Segmentation mask as ``uint32`` array of shape ``(H, W)``.

    Examples:
        >>> import spex as sp
        >>> img, channels = sp.load_image("large_image.ome.tiff")
        >>> labels = sp.cellpose_cellseg(img, [0], diameter=30, scaling=1.0)
        >>> # Automatically uses tiling for large images

    Notes:
        - Uses proven Patient_test_1 methodology for large files
        - Automatically switches to Dask-based tiling when needed
        - Maintains backward compatibility with existing API
    """

    # Handle deprecated num_tiles parameter
    if num_tiles is not None:
        warnings.warn(
            "num_tiles parameter is deprecated. Use tile_size instead for explicit control.",
            DeprecationWarning,
            stacklevel=2
        )

    # Automatic tiling decision (new feature)
    if auto_tiling and should_use_tiling_for_image(img, auto_tile_memory_mb):
        from .dask_watershed import cellpose_cellseg_dask
        return cellpose_cellseg_dask(
            img, seg_channels, diameter, scaling,
            tile_size=tile_size, overlap=overlap
        )

    # Explicit tiling requested
    if tile_size is not None:
        from .dask_watershed import cellpose_cellseg_dask
        return cellpose_cellseg_dask(
            img, seg_channels, diameter, scaling,
            tile_size=tile_size, overlap=overlap
        )

    # Legacy logic for backward compatibility
    tile_size_override = None
    overlap_override = overlap
    should_tile = False

    if num_tiles is not None:
        tile_size_override = _tile_size_from_num_tiles(
            img.shape[1:],
            num_tiles,
            overlap_override,
        )
        should_tile = True

    if not should_tile and auto_tile_memory_mb is not None:
        estimated = _estimate_memory_usage(img)
        if estimated > auto_tile_memory_mb:
            should_tile = True

    if should_tile:
        from .dask_watershed import cellpose_cellseg_dask
        resolved_overlap = _resolve_overlap(tile_size_override, overlap_override)
        return cellpose_cellseg_dask(
            img,
            list(seg_channels),
            diameter,
            scaling,
            tile_size=tile_size_override,
            overlap=resolved_overlap,
        )

    return _cellpose_core(img, seg_channels, diameter, scaling)


def _cellpose_core(
    img: np.ndarray,
    seg_channels: Sequence[int],
    diameter: int,
    scaling: int,
) -> np.ndarray:
    """Core Cellpose invocation used by both monolithic and tiled paths."""
    from spex.core.utils import download_cellpose_models
    from cellpose import models

    download_cellpose_models()

    accumulator = np.zeros((img.shape[1], img.shape[2]), dtype=np.float32)
    for channel_index in seg_channels:
        accumulator += img[channel_index]

    seg_image = accumulator
    if scaling != 1:
        new_height = int(seg_image.shape[0] * scaling)
        new_width = int(seg_image.shape[1] * scaling)
        seg_image = cv2.resize(
            seg_image,
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST,
        )

    model = models.Cellpose(gpu=False, model_type="nuclei")
    labels, _, _, _ = model.eval(
        [seg_image],
        channels=[[0, 0]],
        diameter=diameter,
    )

    labels_array = np.float32(labels[0])

    if scaling != 1:
        original_height = img.shape[1]
        original_width = img.shape[2]
        labels_array = cv2.resize(
            labels_array,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )

    return np.uint32(labels_array)


def cellpose_cellseg_tiled(
    img: np.ndarray,
    seg_channels: Sequence[int],
    diameter: int,
    scaling: int,
    tile_size: Optional[Tuple[int, int]] = None,
    overlap: Optional[int] = None,
) -> np.ndarray:
    """Convenience wrapper that always performs tiled Cellpose segmentation."""

    resolved_overlap = _resolve_overlap(tile_size, overlap)
    return cellpose_cellseg_dask(
        img,
        list(seg_channels),
        diameter,
        scaling,
        tile_size=tile_size,
        overlap=resolved_overlap,
    )


def _resolve_overlap(
    tile_size: Optional[Tuple[int, int]],
    overlap: Optional[int],
) -> Optional[int]:
    if overlap is not None:
        return overlap

    if tile_size is None:
        return None

    return max(DEFAULT_MIN_OVERLAP, int(min(tile_size) * DEFAULT_OVERLAP_RATIO))


def _tile_size_from_num_tiles(
    image_shape: Tuple[int, int],
    num_tiles: int,
    overlap: Optional[int],
) -> Tuple[int, int]:
    if num_tiles <= 0:
        raise ValueError(f"Number of tiles must be positive, got {num_tiles}")

    height, width = image_shape
    grid = int(np.ceil(np.sqrt(num_tiles)))
    step_h = int(np.ceil(height / grid))
    step_w = int(np.ceil(width / grid))

    resolved_overlap = overlap if overlap is not None else 0
    min_tile = resolved_overlap * 2 + 1

    tile_h = max(min_tile, step_h + resolved_overlap)
    tile_w = max(min_tile, step_w + resolved_overlap)

    return min(height, tile_h), min(width, tile_w)


__all__ = ["cellpose_cellseg", "cellpose_cellseg_tiled"]
