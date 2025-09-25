from skimage.filters import median
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import dilation, erosion, disk
import skimage
from skimage.measure import label
import numpy as np


def watershed_classic(img, seg_channels, num_tiles=None, overlap=64, auto_tile_memory_mb=100):
    """Detect nuclei in image using classic watershed with automatic Dask tiling

    Parameters
    ----------
    img : Multichannel image as numpy array
    seg_channels: list of indices to use for nuclear segmentation
    num_tiles: Optional number of tiles for tiled processing
    overlap: Overlap between tiles in pixels
    auto_tile_memory_mb: Memory threshold for automatic tiling (default: 100MB)

    Returns
    -------
    dilated_labels : per cell segmentation as numpy array
    """
    from ..tiling.core import _estimate_memory_usage
    from ..tiling.dask_segmentation import dask_apply_tiling_to_segmentation

    # Legacy num_tiles support
    if num_tiles is not None:
        from ..tiling.core import _tile_size_from_num_tiles
        tile_size = _tile_size_from_num_tiles(img.shape[1:], num_tiles, overlap)
        return dask_apply_tiling_to_segmentation(
            _watershed_core, img, seg_channels,
            tile_size=tile_size, overlap=overlap
        )

    # Automatic tiling detection - works under the hood
    estimated = _estimate_memory_usage(img)
    if estimated > auto_tile_memory_mb:
        # Use default tile size for automatic tiling
        tile_size = (2000, 2000)
        return dask_apply_tiling_to_segmentation(
            _watershed_core, img, seg_channels,
            tile_size=tile_size, overlap=overlap
        )

    # Regular segmentation for small images
    return _watershed_core(img, seg_channels)


def _watershed_core(img, seg_channels):
    """Core watershed segmentation logic."""
    temp2 = np.zeros((img.shape[1], img.shape[2]))
    for i in seg_channels:
        temp = img[i]
        temp2 = temp + temp2

    seg_image = temp2 / len(seg_channels)
    med = median(seg_image, disk(3))

    coords = peak_local_max(med, min_distance=2, footprint=np.ones((3, 3)))
    local_max = np.zeros_like(med, dtype=bool)
    local_max[tuple(coords.T)] = True

    otsu = skimage.filters.threshold_otsu(med)
    otsu_mask = med > otsu

    otsu_mask = skimage.morphology.binary_dilation(otsu_mask, np.ones((2, 2)))
    masked_peaks = local_max * otsu_mask

    seed_label = label(masked_peaks)

    watershed_labels = watershed(
        image=-med,
        markers=seed_label,
        mask=otsu_mask,
        watershed_line=True,
        compactness=20,
    )

    selem = disk(1)
    dilated_labels = erosion(watershed_labels, selem)
    selem = disk(1)
    dilated_labels = dilation(dilated_labels, selem)

    return dilated_labels
