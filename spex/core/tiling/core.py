"""
Core tiling utilities for memory-efficient image processing.

This module provides the fundamental functions for computing tile coordinates,
cropping core regions, and placing cores back into destination arrays.
"""

from typing import Tuple, List, Optional
import numpy as np


def compute_tiles(shape: Tuple[int, int], tile: Tuple[int, int], overlap: int) -> List[Tuple[slice, slice]]:
    """Compute tile coordinates with overlap.

    This function divides an image into overlapping tiles, returning the coordinates
    for each tile. The tiles are arranged in a grid pattern with specified overlap
    between adjacent tiles.

    Args:
        shape: Image shape (height, width)
        tile: Tile size (height, width)
        overlap: Overlap in pixels between adjacent tiles

    Returns:
        List of (y_slice, x_slice) tuples for each tile

    Raises:
        ValueError: If overlap is negative or >= min(tile)
        ValueError: If tile dimensions are not positive

    Examples:
        >>> tiles = compute_tiles((100, 100), (50, 50), 10)
        >>> len(tiles)
        9
        >>> tiles[0]
        (slice(0, 50), slice(0, 50))
    """
    if overlap < 0:
        raise ValueError(f"Overlap must be non-negative, got {overlap}")

    if min(tile) <= 0:
        raise ValueError(f"Tile dimensions must be positive, got {tile}")

    if overlap >= min(tile):
        raise ValueError(f"Overlap {overlap} must be < min(tile) {min(tile)}")

    height, width = shape
    tile_h, tile_w = tile

    # If image is smaller than tile, return single tile
    if height <= tile_h and width <= tile_w:
        return [(slice(0, height), slice(0, width))]

    tiles = []
    step_h = tile_h - overlap
    step_w = tile_w - overlap

    # Generate tile coordinates
    # Calculate how many tiles we need in each dimension
    n_tiles_h = max(1, (height - overlap + step_h - 1) // step_h)
    n_tiles_w = max(1, (width - overlap + step_w - 1) // step_w)

    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            y = i * step_h
            x = j * step_w
            y_end = min(y + tile_h, height)
            x_end = min(x + tile_w, width)
            tiles.append((slice(y, y_end), slice(x, x_end)))

    return tiles


def crop_core(view: np.ndarray, overlap: int) -> np.ndarray:
    """Crop core region from tile view, removing overlap.

    This function extracts the core (non-overlapping) region from a tile view.
    The core region is the central part of the tile that doesn't overlap with
    adjacent tiles.

    Args:
        view: Tile view as 2D numpy array
        overlap: Overlap in pixels to remove from each side

    Returns:
        Core region as 2D numpy array

    Raises:
        ValueError: If overlap is negative
        ValueError: If view is not 2D array

    Examples:
        >>> tile = np.random.rand(60, 60)  # 50 + 10 overlap
        >>> core = crop_core(tile, 10)
        >>> core.shape
        (40, 40)
    """
    if overlap < 0:
        raise ValueError(f"Overlap must be non-negative, got {overlap}")

    if view.ndim != 2:
        raise ValueError(f"Expected 2D array, got {view.ndim}D")

    if overlap == 0:
        return view.copy()

    h, w = view.shape

    # If tile is too small for overlap, return as is
    if h <= 2 * overlap or w <= 2 * overlap:
        return view.copy()

    # Crop core region (remove overlap from all sides)
    return view[overlap:h-overlap, overlap:w-overlap].copy()


def crop_core_safe(view: np.ndarray, overlap: int, tile: Tuple[slice, slice], image_shape: Tuple[int, int], tile_size: Tuple[int, int]) -> np.ndarray:
    """Crop core region safely, handling edge cases.

    This function extracts the core region while ensuring it fits within the
    original image bounds when placed back. The core region should cover
    the entire image without gaps.

    Args:
        view: Tile view as 2D numpy array
        overlap: Overlap in pixels to remove from each side
        tile: Tile coordinates (y_slice, x_slice)
        image_shape: Original image shape (height, width)
        tile_size: Tile size (height, width)

    Returns:
        Core region as 2D numpy array
    """
    if overlap < 0:
        raise ValueError(f"Overlap must be non-negative, got {overlap}")

    if view.ndim != 2:
        raise ValueError(f"Expected 2D array, got {view.ndim}D")

    if overlap == 0:
        return view.copy()

    h, w = view.shape
    img_h, img_w = image_shape
    tile_h, tile_w = tile_size

    # Calculate the step size (tile size - overlap)
    step_h = tile_h - overlap
    step_w = tile_w - overlap

    # Calculate which "core row" and "core column" this tile represents
    core_row = tile[0].start // step_h
    core_col = tile[1].start // step_w

    # Calculate the core bounds in image coordinates
    core_y_start = core_row * step_h
    core_y_end = min((core_row + 1) * step_h, img_h)
    core_x_start = core_col * step_w
    core_x_end = min((core_col + 1) * step_w, img_w)

    # Convert to tile coordinates
    tile_y_start = core_y_start - tile[0].start
    tile_y_end = core_y_end - tile[0].start
    tile_x_start = core_x_start - tile[1].start
    tile_x_end = core_x_end - tile[1].start

    # Ensure bounds are within tile
    tile_y_start = max(0, tile_y_start)
    tile_y_end = min(h, tile_y_end)
    tile_x_start = max(0, tile_x_start)
    tile_x_end = min(w, tile_x_end)

    # Extract core
    return view[tile_y_start:tile_y_end, tile_x_start:tile_x_end].copy()


def get_core_coordinates(tile: Tuple[slice, slice], overlap: int) -> Tuple[int, int]:
    """Get coordinates where core should be placed.

    Args:
        tile: Tile coordinates (y_slice, x_slice)
        overlap: Overlap in pixels

    Returns:
        (y, x) coordinates where core should be placed
    """
    y_start = tile[0].start + overlap
    x_start = tile[1].start + overlap
    return y_start, x_start


def get_core_coordinates_safe(tile: Tuple[slice, slice], overlap: int, image_shape: Tuple[int, int], tile_size: Tuple[int, int]) -> Tuple[int, int]:
    """Get coordinates where core should be placed, handling edge cases.

    Args:
        tile: Tile coordinates (y_slice, x_slice)
        overlap: Overlap in pixels
        image_shape: Original image shape (height, width)
        tile_size: Tile size (height, width)

    Returns:
        (y, x) coordinates where core should be placed
    """
    img_h, img_w = image_shape
    tile_h, tile_w = tile_size

    # Calculate the step size (tile size - overlap)
    step_h = tile_h - overlap
    step_w = tile_w - overlap

    # Calculate which "core row" and "core column" this tile represents
    core_row = tile[0].start // step_h
    core_col = tile[1].start // step_w

    # Calculate the core coordinates in image space
    y_start = core_row * step_h
    x_start = core_col * step_w

    return y_start, x_start


def place_core(dst: np.ndarray, core: np.ndarray, yx: Tuple[int, int]) -> None:
    """Place core region into destination array at specified coordinates.

    This function places a core region into a destination array at the specified
    coordinates. The core is copied into the destination array.

    Args:
        dst: Destination array (modified in place)
        core: Core region to place
        yx: Coordinates (y, x) where to place the core

    Raises:
        ValueError: If coordinates are negative
        ValueError: If core would extend beyond destination bounds

    Examples:
        >>> dst = np.zeros((100, 100))
        >>> core = np.ones((30, 30))
        >>> place_core(dst, core, (10, 20))
        >>> np.all(dst[10:40, 20:50] == 1)
        True
    """
    y, x = yx

    if y < 0 or x < 0:
        raise ValueError(f"Coordinates must be non-negative, got {yx}")

    core_h, core_w = core.shape
    dst_h, dst_w = dst.shape

    # Check bounds
    if y + core_h > dst_h or x + core_w > dst_w:
        raise ValueError(f"Core would extend beyond destination bounds: "
                        f"core {core.shape} at {yx} in dst {dst.shape}")

    # Place core
    dst[y:y+core_h, x:x+core_w] = core


def _estimate_memory_usage(img, dtype_bytes=4):
    """
    Estimate memory usage for image processing.

    Args:
        img: Input image array
        dtype_bytes: Bytes per element (default: 4 for float32)

    Returns:
        Estimated memory usage in MB
    """
    return (img.nbytes * 3) / (1024 * 1024)  # 3x for input + output + intermediate


def _calculate_auto_tiles(img, memory_threshold_mb=500, overlap=64):
    """
    Calculate optimal number of tiles based on memory threshold.

    Args:
        img: Input image array
        memory_threshold_mb: Memory threshold in MB (default: 500)
        overlap: Overlap between tiles in pixels (default: 64)

    Returns:
        Number of tiles to use
    """
    estimated_memory = _estimate_memory_usage(img)

    if estimated_memory <= memory_threshold_mb:
        return 1  # No tiling needed

    # Calculate how many tiles we need to stay under threshold
    memory_per_tile = memory_threshold_mb
    total_pixels = img.size
    pixels_per_tile = (memory_per_tile * 1024 * 1024) // (3 * 4)  # 3x for processing, 4 bytes per pixel

    # Calculate tiles needed
    tiles_needed = max(1, int(np.ceil(total_pixels / pixels_per_tile)))

    # Round up to nearest square number for better distribution
    sqrt_tiles = int(np.ceil(np.sqrt(tiles_needed)))
    return sqrt_tiles * sqrt_tiles


def _apply_tiling_to_segmentation(seg_func, img, *args, num_tiles=None, overlap=64, auto_tile_memory_mb=500, **kwargs):
    """Apply tiling to any segmentation function for memory efficiency.

    This function divides large images into overlapping tiles, processes each tile
    with the provided segmentation function, and reassembles the results.

    Args:
        seg_func: Segmentation function to apply to each tile
        img: Input image as numpy array of shape (C, H, W)
        *args: Positional arguments to pass to seg_func
        num_tiles: Number of tiles to divide image into (default: None for auto)
        overlap: Overlap between tiles in pixels
        auto_tile_memory_mb: Auto-tile if memory usage exceeds this threshold in MB (default: 500)
        **kwargs: Keyword arguments to pass to seg_func

    Returns:
        labels: Segmentation result as numpy array of shape (H, W)

    Raises:
        ValueError: If num_tiles is not positive
        ValueError: If overlap is negative
        ValueError: If img is not 3D array
    """
    # Input validation
    if img.ndim != 3:
        raise ValueError(f"Expected 3D array (C, H, W), got {img.ndim}D")

    if overlap < 0:
        raise ValueError(f"Overlap must be non-negative, got {overlap}")

    # Auto-calculate tiles if not specified
    if num_tiles is None:
        num_tiles = _calculate_auto_tiles(img, auto_tile_memory_mb, overlap)
    elif num_tiles <= 0:
        raise ValueError(f"Number of tiles must be positive, got {num_tiles}")

    # Get image dimensions
    channels, height, width = img.shape
    image_shape = (height, width)

    # If only 1 tile needed, process directly
    if num_tiles == 1:
        return seg_func(img, *args, **kwargs)

    # Calculate tile size based on number of tiles
    # Use square tiles for simplicity
    tile_size = int(np.sqrt((height * width) / num_tiles))
    tile_size = (tile_size, tile_size)

    # Ensure tile size is not larger than image
    tile_size = (min(tile_size[0], height), min(tile_size[1], width))

    # If image is smaller than tile, use regular segmentation
    if height <= tile_size[0] and width <= tile_size[1]:
        return seg_func(img, *args, **kwargs)

    # Compute tile coordinates
    tiles = compute_tiles(image_shape, tile_size, overlap)

    # Initialize result array
    labels_final = np.zeros(image_shape, dtype=np.uint32)

    # Process each tile
    for tile_idx, tile in enumerate(tiles):
        # Extract tile view
        tile_view = img[:, tile[0], tile[1]]

        # Segment tile using provided function
        try:
            tile_labels = seg_func(tile_view, *args, **kwargs)
        except Exception as e:
            # If segmentation fails on a tile, skip it
            print(f"Warning: Segmentation failed on tile {tile_idx}: {e}")
            continue

        # Crop core region from tile
        core = crop_core_safe(
            tile_labels,
            overlap,
            tile,
            image_shape,
            tile_size
        )

        # Get coordinates where core should be placed
        y_start, x_start = get_core_coordinates_safe(
            tile,
            overlap,
            image_shape,
            tile_size
        )

        # Place core in result array
        try:
            place_core(labels_final, core, (y_start, x_start))
        except ValueError as e:
            # Handle edge cases where core might not fit
            print(f"Warning: Could not place core for tile {tile_idx}: {e}")
            continue

    # Relabel to ensure continuous label sequence
    labels_final = _relabel_segmentation(labels_final)

    return labels_final


def _relabel_segmentation(labels: np.ndarray) -> np.ndarray:
    """Relabel segmentation to ensure continuous label sequence.

    Args:
        labels: Segmentation labels array

    Returns:
        Relabeled segmentation with continuous labels starting from 1
    """
    from skimage.measure import label

    # Create binary mask
    binary_mask = labels > 0

    # Relabel connected components
    relabeled = label(binary_mask)

    # Map old labels to new labels
    unique_labels = np.unique(labels[labels > 0])
    if len(unique_labels) == 0:
        return relabeled

    # Create mapping from old to new labels
    label_mapping = {}
    for old_label in unique_labels:
        # Find the new label for this old label
        mask = labels == old_label
        new_labels = relabeled[mask]
        if len(new_labels) > 0:
            new_label = new_labels[0]
            label_mapping[old_label] = new_label

    # Apply mapping
    result = np.zeros_like(labels)
    for old_label, new_label in label_mapping.items():
        result[labels == old_label] = new_label

    return result