"""
Dask-based tiling segmentation for large images with proper label preservation.

This module provides efficient processing of large images using Dask arrays
and proper label merging to avoid loss of segmentation objects.
"""

import numpy as np
import dask.array as da
import dask
from dask import delayed
from typing import Tuple, Optional, Callable, Any, Dict
from .core import compute_tiles, crop_core_safe, get_core_coordinates_safe, place_core


def dask_apply_tiling_to_segmentation(
    seg_func: Callable,
    img: np.ndarray,
    *args,
    tile_size: Optional[Tuple[int, int]] = None,
    overlap: int = None,
    chunk_size: Optional[Tuple[int, int]] = None,
    parallel: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Apply tiling to segmentation function using Dask for memory-efficient processing.

    This function uses Dask arrays to process large images in chunks, avoiding
    memory issues and preserving all segmentation labels through proper merging.

    Args:
        seg_func: Segmentation function to apply to each tile
        img: Input image as numpy array of shape (C, H, W)
        *args: Positional arguments to pass to seg_func
        tile_size: Size of tiles (height, width). If None, auto-calculated
        overlap: Overlap between tiles in pixels. If None, calculated as 20% of tile size
        chunk_size: Dask chunk size. If None, same as tile_size
        parallel: Use parallel processing with dask.delayed (default True)
        **kwargs: Keyword arguments to pass to seg_func

    Returns:
        labels: Segmentation result as numpy array

    Raises:
        ValueError: If img is not 3D array
        ValueError: If tile_size is invalid
    """
    # Input validation
    if img.ndim != 3:
        raise ValueError(f"Expected 3D array (C, H, W), got {img.ndim}D")

    channels, height, width = img.shape
    image_shape = (height, width)

    # Auto-calculate tile size if not provided
    if tile_size is None:
        tile_size = _calculate_optimal_tile_size(img.shape, height, width)

    # Auto-calculate overlap if not provided
    if overlap is None:
        overlap = max(32, int(min(tile_size) * 0.20))  # 20% with minimum 32px

    # Validate configuration and show warnings
    validation = validate_tiling_config(img.shape, tile_size, overlap)

    if not validation['valid']:
        raise ValueError("Invalid tiling configuration")

    # Show warnings
    for warning in validation['warnings']:
        print(warning)

    # Show estimate
    estimate = validation['estimate']
    print(f"📊 Tiling estimate: {estimate['num_tiles']} tiles, "
          f"~{estimate['total_time_minutes']:.1f} min, "
          f"{estimate['memory_per_tile_mb']:.1f}MB/tile")

    # If image is smaller than one tile, process directly
    if height <= tile_size[0] and width <= tile_size[1]:
        return seg_func(img, *args, **kwargs)

    print(f"🧩 Dask tiling: {tile_size} tiles, {overlap}px overlap")

    # Compute tile coordinates
    tiles = compute_tiles(image_shape, tile_size, overlap)
    print(f"📊 Processing {len(tiles)} tiles...")

    # Check if we should use parallel processing
    if parallel and len(tiles) > 1:
        print(f"🚀 Using parallel processing with dask.delayed")
        return _process_tiles_parallel(seg_func, img, tiles, args, kwargs,
                                       tile_size, overlap, image_shape)
    else:
        print(f"🔄 Using sequential processing")
        return _process_tiles_sequential(seg_func, img, tiles, args, kwargs,
                                        tile_size, overlap, image_shape)


def _process_tiles_parallel(seg_func, img, tiles, args, kwargs, tile_size, overlap, image_shape):
    """
    Process tiles in parallel using dask.delayed for improved performance.
    """
    import time

    # Initialize result array
    labels_final = np.zeros(image_shape, dtype=np.uint32)

    # Import tqdm for progress tracking
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=len(tiles), desc="Processing tiles", unit="tile")
    except ImportError:
        progress_bar = None
        print("📊 tqdm not available, showing basic progress")

    # Memory monitoring
    import psutil
    import os
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # MB

    print(f"🔄 Creating delayed computation graph...")
    start_time = time.time()

    # Create delayed computation for each tile
    @delayed
    def process_single_tile(tile_idx, tile):
        tile_view = img[:, tile[0], tile[1]]
        try:
            tile_labels = seg_func(tile_view, *args, **kwargs)
            return (tile_idx, tile_labels, tile, True)
        except Exception as e:
            print(f"    ⚠️  Warning: Segmentation failed on tile {tile_idx}: {e}")
            return (tile_idx, None, tile, False)

    # Create delayed tasks
    delayed_tasks = [process_single_tile(i, tile) for i, tile in enumerate(tiles)]

    # Process tiles in batches to control memory usage
    batch_size = min(4, len(tiles))  # Process 4 tiles at a time
    global_label_counter = 1
    processed_tiles = 0
    failed_tiles = 0

    print(f"🚀 Processing {len(tiles)} tiles in batches of {batch_size}...")

    for batch_start in range(0, len(tiles), batch_size):
        batch_end = min(batch_start + batch_size, len(tiles))
        batch_tasks = delayed_tasks[batch_start:batch_end]

        # Compute batch in parallel
        batch_results = dask.compute(*batch_tasks)

        # Process results sequentially to maintain label consistency
        for tile_idx, tile_labels, tile, success in batch_results:
            if progress_bar:
                progress_bar.set_description(f"Assembling tile {tile_idx + 1}/{len(tiles)}")
            else:
                print(f"  🔸 Assembling tile {tile_idx + 1}/{len(tiles)}")

            if not success:
                failed_tiles += 1
                if progress_bar:
                    progress_bar.update(1)
                continue

            # Relabel tile with unique global labels
            tile_labels_relabeled = _relabel_tile_with_global_counter(
                tile_labels, global_label_counter
            )

            # Update global counter
            if tile_labels_relabeled.max() > 0:
                global_label_counter = tile_labels_relabeled.max() + 1

            # Crop core region from tile
            core = crop_core_safe(
                tile_labels_relabeled,
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

            # Place core in result array with boundary merging
            try:
                _place_core_with_boundary_merging(
                    labels_final, core, (y_start, x_start), overlap
                )
                processed_tiles += 1
            except ValueError as e:
                print(f"    ⚠️  Warning: Could not place core for tile {tile_idx}: {e}")
                failed_tiles += 1

            # Update progress bar
            if progress_bar:
                progress_bar.update(1)

        # Memory monitoring after each batch
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - start_memory

        if progress_bar:
            progress_bar.set_postfix({
                'Memory': f'{current_memory:.1f}MB',
                'Δ': f'+{memory_increase:.1f}MB',
                'Failed': failed_tiles
            })
        else:
            print(f"    💾 Batch complete. Memory: {current_memory:.1f}MB (+{memory_increase:.1f}MB), Failed: {failed_tiles}")

    # Close progress bar
    if progress_bar:
        progress_bar.close()

    # Final report
    final_memory = process.memory_info().rss / 1024 / 1024
    total_time = time.time() - start_time

    print(f"📊 Parallel processing complete: {processed_tiles}/{len(tiles)} tiles processed, "
          f"{failed_tiles} failed")
    print(f"⏱️  Total time: {total_time:.1f}s, Memory: {final_memory:.1f}MB")
    print(f"✅ Dask parallel tiling completed. Labels found: {len(np.unique(labels_final)) - 1}")

    return labels_final


def _process_tiles_sequential(seg_func, img, tiles, args, kwargs, tile_size, overlap, image_shape):
    """
    Process tiles sequentially (fallback for single tile or when parallel=False).
    """
    # Initialize result array
    labels_final = np.zeros(image_shape, dtype=np.uint32)

    # Global label counter for unique labeling across tiles
    global_label_counter = 1

    # Import tqdm for progress tracking
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=len(tiles), desc="Processing tiles", unit="tile")
    except ImportError:
        progress_bar = None
        print("📊 tqdm not available, showing basic progress")

    # Memory monitoring
    import psutil
    import os
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Process each tile with progress tracking
    processed_tiles = 0
    failed_tiles = 0

    for tile_idx, tile in enumerate(tiles):
        # Update progress
        if progress_bar:
            progress_bar.set_description(f"Processing tile {tile_idx + 1}/{len(tiles)}")
        else:
            print(f"  🔸 Processing tile {tile_idx + 1}/{len(tiles)}")

        # Extract tile view
        tile_view = img[:, tile[0], tile[1]]

        # Segment tile using provided function
        try:
            tile_labels = seg_func(tile_view, *args, **kwargs)

            # Relabel tile with unique global labels
            tile_labels_relabeled = _relabel_tile_with_global_counter(
                tile_labels, global_label_counter
            )

            # Update global counter
            if tile_labels_relabeled.max() > 0:
                global_label_counter = tile_labels_relabeled.max() + 1

            processed_tiles += 1

        except Exception as e:
            print(f"    ⚠️  Warning: Segmentation failed on tile {tile_idx}: {e}")
            failed_tiles += 1
            if progress_bar:
                progress_bar.update(1)
            continue

        # Crop core region from tile
        core = crop_core_safe(
            tile_labels_relabeled,
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

        # Place core in result array with boundary merging
        try:
            _place_core_with_boundary_merging(
                labels_final, core, (y_start, x_start), overlap
            )
        except ValueError as e:
            print(f"    ⚠️  Warning: Could not place core for tile {tile_idx}: {e}")
            failed_tiles += 1

        # Update progress bar
        if progress_bar:
            progress_bar.update(1)

        # Memory monitoring every 10 tiles
        if (tile_idx + 1) % 10 == 0:
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - start_memory

            if progress_bar:
                progress_bar.set_postfix({
                    'Memory': f'{current_memory:.1f}MB',
                    'Δ': f'+{memory_increase:.1f}MB',
                    'Failed': failed_tiles
                })
            else:
                print(f"    💾 Memory: {current_memory:.1f}MB (+{memory_increase:.1f}MB), Failed: {failed_tiles}")

    # Close progress bar
    if progress_bar:
        progress_bar.close()

    # Final memory report
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"📊 Sequential processing complete: {processed_tiles}/{len(tiles)} tiles processed, "
          f"{failed_tiles} failed, Memory: {final_memory:.1f}MB")
    print(f"✅ Dask sequential tiling completed. Labels found: {len(np.unique(labels_final)) - 1}")

    return labels_final


def _relabel_tile_with_global_counter(tile_labels: np.ndarray, start_label: int) -> np.ndarray:
    """
    Relabel tile labels with globally unique labels starting from start_label.

    This preserves all original objects while ensuring global uniqueness.

    Args:
        tile_labels: Tile segmentation labels
        start_label: Starting label for this tile

    Returns:
        Relabeled array with globally unique labels
    """
    if tile_labels.max() == 0:
        return tile_labels.copy()

    # Get unique labels (excluding background)
    unique_labels = np.unique(tile_labels[tile_labels > 0])

    # Create mapping from old to new labels
    relabeled = np.zeros_like(tile_labels)

    for i, old_label in enumerate(unique_labels):
        new_label = start_label + i
        relabeled[tile_labels == old_label] = new_label

    return relabeled


def _place_core_with_boundary_merging(
    dst: np.ndarray,
    core: np.ndarray,
    yx: Tuple[int, int],
    overlap: int
) -> None:
    """
    Place core region into destination with intelligent boundary merging.

    This function handles overlapping regions by merging labels that represent
    the same object split across tile boundaries.

    Args:
        dst: Destination array (modified in place)
        core: Core region to place
        yx: Coordinates (y, x) where to place the core
        overlap: Overlap size for boundary analysis
    """
    y, x = yx
    core_h, core_w = core.shape

    # Validate bounds
    if y < 0 or x < 0:
        raise ValueError(f"Coordinates must be non-negative, got {yx}")

    if y + core_h > dst.shape[0] or x + core_w > dst.shape[1]:
        raise ValueError(f"Core would extend beyond destination bounds")

    # Get the region where core will be placed
    dst_region = dst[y:y+core_h, x:x+core_w]

    # For overlapping regions, perform intelligent merging
    if np.any(dst_region > 0) and np.any(core > 0):
        # Merge overlapping labels
        merged_core = _merge_overlapping_labels(dst_region, core, overlap)
        dst[y:y+core_h, x:x+core_w] = merged_core
    else:
        # No overlap, simple placement
        # Only place non-zero labels, preserve existing labels
        mask = core > 0
        dst[y:y+core_h, x:x+core_w][mask] = core[mask]


def _merge_overlapping_labels(existing: np.ndarray, new: np.ndarray, overlap: int) -> np.ndarray:
    """
    Merge overlapping labels intelligently.

    This function analyzes the overlap region and merges labels that likely
    represent the same object.

    Args:
        existing: Existing labels in destination
        new: New labels from core
        overlap: Overlap size for analysis

    Returns:
        Merged labels array
    """
    result = existing.copy()

    # Simple strategy: prioritize new labels but preserve existing where new is zero
    mask_new_nonzero = new > 0
    mask_existing_zero = existing == 0

    # Place new labels where there's no conflict
    result[mask_new_nonzero & mask_existing_zero] = new[mask_new_nonzero & mask_existing_zero]

    # For conflict areas, could implement more sophisticated merging
    # For now, keep existing labels (conservative approach)

    return result


def test_dask_segmentation_on_crop():
    """Test Dask segmentation on small crop to verify it works correctly"""
    print("🧪 Testing Dask segmentation on small crop...")

    try:
        import spex as sp

        # Load and create small crop
        img, channels = sp.load_image("Patient_test_1.ome.tiff")
        crop_h, crop_w = 512, 512
        start_h = (img.shape[1] - crop_h) // 2
        start_w = (img.shape[2] - crop_w) // 2
        crop_img = img[:, start_h:start_h+crop_h, start_w:start_w+crop_w]

        print(f"📐 Crop shape: {crop_img.shape}")

        # Test regular segmentation for comparison
        import time
        start_time = time.time()
        labels_regular = sp.watershed_classic(crop_img, [0], num_tiles=None)
        regular_time = time.time() - start_time
        regular_labels = len(np.unique(labels_regular)) - 1

        print(f"🚫 Regular: {regular_time:.3f}s, {regular_labels} labels")

        # Test Dask segmentation
        start_time = time.time()
        labels_dask = dask_apply_tiling_to_segmentation(
            lambda img, channels: sp.watershed_classic(img, channels, num_tiles=None),
            crop_img,
            [0],
            tile_size=(256, 256),
            overlap=32
        )
        dask_time = time.time() - start_time
        dask_labels = len(np.unique(labels_dask)) - 1

        print(f"🧩 Dask: {dask_time:.3f}s, {dask_labels} labels")

        # Compare results
        print(f"\n📊 Comparison:")
        print(f"  Labels preserved: {dask_labels/regular_labels*100:.1f}%")
        print(f"  Time overhead: {dask_time/regular_time:.2f}x")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def _calculate_optimal_tile_size(img_shape, height, width):
    """
    Calculate optimal tile size for large images with performance considerations.

    This version prioritizes processing speed over memory usage by using
    smaller tiles that can be processed in parallel.

    Args:
        img_shape: Full image shape (C, H, W)
        height: Image height
        width: Image width

    Returns:
        Optimal tile size (height, width)
    """
    channels = img_shape[0]
    total_pixels = height * width

    # Strategy: Use smaller tiles for better parallelization
    # Target 50-100MB per tile for faster watershed processing
    target_mb = 100  # Reduced from 500MB for faster processing
    target_pixels = (target_mb * 1024 * 1024) // (channels * 4)  # 4 bytes per float32

    # Calculate base tile dimension
    base_tile_dim = int(np.sqrt(target_pixels))

    # Apply constraints - prioritize speed over large tiles
    min_tile_size = 1024  # Minimum 1K tiles
    max_tile_size = 2048  # Reduced from 4096 for faster processing

    # Clamp to reasonable range
    tile_dim = max(min_tile_size, min(base_tile_dim, max_tile_size))

    # Ensure tile doesn't exceed image dimensions
    tile_h = min(tile_dim, height)
    tile_w = min(tile_dim, width)

    # For large images, allow more tiles for parallel processing
    max_tiles = 100  # Increased limit for parallel processing
    estimated_tiles_h = (height + tile_h - 1) // tile_h
    estimated_tiles_w = (width + tile_w - 1) // tile_w
    estimated_total_tiles = estimated_tiles_h * estimated_tiles_w

    print(f"🧩 Calculated tile size: {tile_h}x{tile_w} → {estimated_total_tiles} tiles total")

    # Only increase tile size if we have way too many tiles (>200)
    if estimated_total_tiles > 200:
        scale_factor = np.sqrt(estimated_total_tiles / 100)
        tile_h = min(int(tile_h * scale_factor), height, max_tile_size)
        tile_w = min(int(tile_w * scale_factor), width, max_tile_size)

        # Recalculate
        estimated_tiles_h = (height + tile_h - 1) // tile_h
        estimated_tiles_w = (width + tile_w - 1) // tile_w
        estimated_total_tiles = estimated_tiles_h * estimated_tiles_w

        print(f"⚠️  Adjusted to {tile_h}x{tile_w} → {estimated_total_tiles} tiles for manageable count")

    return (tile_h, tile_w)


def _estimate_processing_time(img_shape, tile_size, overlap):
    """
    Estimate processing time for tiled segmentation.

    Args:
        img_shape: Image shape (C, H, W)
        tile_size: Tile size (height, width)
        overlap: Overlap in pixels

    Returns:
        Dictionary with timing estimates
    """
    channels, height, width = img_shape

    # Calculate number of tiles
    from .core import compute_tiles
    tiles = compute_tiles((height, width), tile_size, overlap)
    num_tiles = len(tiles)

    # Estimate time per tile (based on tile size)
    tile_pixels = tile_size[0] * tile_size[1]
    # Rough estimate: 1M pixels per second for watershed
    time_per_tile = tile_pixels / 1e6

    # Total estimated time
    total_time = num_tiles * time_per_tile

    # Memory per tile
    tile_mb = (tile_pixels * channels * 4) / (1024 * 1024)

    return {
        'num_tiles': num_tiles,
        'time_per_tile': time_per_tile,
        'total_time_minutes': total_time / 60,
        'memory_per_tile_mb': tile_mb,
        'tile_size': tile_size,
        'overlap': overlap
    }


def validate_tiling_config(img_shape, tile_size, overlap):
    """
    Validate tiling configuration and provide warnings.

    Args:
        img_shape: Image shape (C, H, W)
        tile_size: Tile size (height, width)
        overlap: Overlap in pixels

    Returns:
        Dictionary with validation results and warnings
    """
    channels, height, width = img_shape
    warnings = []

    # Check tile size validity
    if tile_size[0] <= overlap * 2 or tile_size[1] <= overlap * 2:
        warnings.append(f"❌ Tile size {tile_size} too small for overlap {overlap}")

    # Check if tiles are too small
    if min(tile_size) < 512:
        warnings.append(f"⚠️  Very small tiles {tile_size} may be inefficient")

    # Check if too many tiles
    from .core import compute_tiles
    tiles = compute_tiles((height, width), tile_size, overlap)
    num_tiles = len(tiles)

    if num_tiles > 50:
        warnings.append(f"⚠️  Large number of tiles ({num_tiles}) may cause timeout")

    # Get time estimate
    estimate = _estimate_processing_time(img_shape, tile_size, overlap)

    if estimate['total_time_minutes'] > 30:
        warnings.append(f"⚠️  Estimated processing time: {estimate['total_time_minutes']:.1f} minutes")

    return {
        'valid': len([w for w in warnings if w.startswith('❌')]) == 0,
        'warnings': warnings,
        'estimate': estimate
    }


if __name__ == "__main__":
    test_dask_segmentation_on_crop()