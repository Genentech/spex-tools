"""
Dask-enhanced watershed segmentation for large images.

This module provides a Dask-based implementation of watershed segmentation
that can handle very large images without memory issues while preserving
segmentation quality.
"""

import numpy as np
from typing import Tuple, Optional

from ..tiling.dask_segmentation import dask_apply_tiling_to_segmentation


def watershed_classic_dask(
    img: np.ndarray,
    seg_channels: list,
    tile_size: Optional[Tuple[int, int]] = None,
    overlap: Optional[int] = None,
    parallel: Optional[bool] = None
) -> np.ndarray:
    """
    Watershed segmentation with Dask-based tiling for large images.

    This function provides memory-efficient processing of large images using
    a tiling approach with proper label preservation. Unlike the standard
    tiling implementation, this version maintains nearly all segmentation
    objects by using a global label counter and intelligent merging.

    Args:
        img: Input image as numpy array of shape (C, H, W)
        seg_channels: List of channel indices to use for segmentation
        tile_size: Size of tiles (height, width). If None, auto-calculated
        overlap: Overlap between tiles in pixels. If None, calculated as 20% of tile size
        parallel: Optional flag to control Dask parallel batches. ``None`` enables
            automatic selection based on tile count.

    Returns:
        labels: Segmentation result as numpy array of shape (H, W)

    Examples:
        >>> import spex as sp
        >>> img, channels = sp.load_image("large_image.ome.tiff")
        >>> labels = watershed_classic_dask(img, [0], tile_size=(2048, 2048))
        >>> print(f"Found {len(np.unique(labels)) - 1} cells")

    Notes:
        - Automatically determines optimal tile size if not specified
        - Uses 20% overlap by default for proper boundary handling
        - Preserves 95%+ of segmentation objects compared to non-tiled processing
        - Can process images larger than available RAM
    """
    # Import here to avoid circular dependencies
    from .watershed import _watershed_core

    return _dask_apply_tiling_to_segmentation(
        _watershed_core,
        img,
        seg_channels,
        tile_size=tile_size,
        overlap=overlap,
        parallel=parallel
    )


def cellpose_cellseg_dask(
    img: np.ndarray,
    seg_channels: list,
    diameter: int,
    scaling: int,
    tile_size: Optional[Tuple[int, int]] = None,
    overlap: Optional[int] = None,
    parallel: Optional[bool] = None
) -> np.ndarray:
    """
    Cellpose segmentation with Dask-based tiling for large images.

    Args:
        img: Input image as numpy array of shape (C, H, W)
        seg_channels: List of channel indices to use for segmentation
        diameter: Typical size of nucleus
        scaling: Integer value scaling
        tile_size: Size of tiles (height, width). If None, auto-calculated
        overlap: Overlap between tiles in pixels. If None, calculated as 20% of tile size
        parallel: Optional flag to control Dask parallel batches. ``None`` enables
            automatic selection based on tile count.

    Returns:
        labels: Segmentation result as numpy array of shape (H, W)
    """
    # Import here to avoid circular dependencies
    from .cellpose_cellseg import _cellpose_core

    return _dask_apply_tiling_to_segmentation(
        _cellpose_core,
        img,
        seg_channels,
        diameter,
        scaling,
        tile_size=tile_size,
        overlap=overlap,
        parallel=parallel
    )


def stardist_cellseg_dask(
    img: np.ndarray,
    seg_channels: list,
    scaling: int,
    threshold: float,
    _min: float,
    _max: float,
    tile_size: Optional[Tuple[int, int]] = None,
    overlap: Optional[int] = None,
    parallel: Optional[bool] = None
) -> np.ndarray:
    """
    StarDist segmentation with Dask-based tiling for large images.

    Args:
        img: Input image as numpy array of shape (C, H, W)
        seg_channels: List of channel indices to use for segmentation
        scaling: Integer value scaling
        threshold: Probability cutoff
        _min: Bottom percentile normalization
        _max: Top percentile normalization
        tile_size: Size of tiles (height, width). If None, auto-calculated
        overlap: Overlap between tiles in pixels. If None, calculated as 20% of tile size
        parallel: Optional flag to control Dask parallel batches. ``None`` enables
            automatic selection based on tile count.

    Returns:
        labels: Segmentation result as numpy array of shape (H, W)
    """
    # Import here to avoid circular dependencies
    from .stardist import _stardist_core

    return _dask_apply_tiling_to_segmentation(
        _stardist_core,
        img,
        seg_channels,
        scaling,
        threshold,
        _min,
        _max,
        tile_size=tile_size,
        overlap=overlap,
        parallel=parallel
    )


def _dask_apply_tiling_to_segmentation(
    seg_func,
    img,
    *args,
    tile_size=None,
    overlap=None,
    parallel=None,
    **kwargs,
):
    """Delegate tiled segmentation to the unified Dask implementation."""

    return dask_apply_tiling_to_segmentation(
        seg_func,
        img,
        *args,
        tile_size=tile_size,
        overlap=overlap,
        parallel=parallel,
        **kwargs,
    )
