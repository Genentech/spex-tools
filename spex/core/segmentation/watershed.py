from skimage.filters import median
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import dilation, erosion, disk
import skimage
from skimage.measure import label
import numpy as np


def watershed_classic(img, seg_channels, num_tiles=None, overlap=64, auto_tile_memory_mb=500):
    """Detect nuclei in image using classic watershed with optional tiling

    Parameters
    ----------
    img : Multichannel image as numpy array
    seg_channels: list of indices to use for nuclear segmentation
    num_tiles: Optional number of tiles for tiled processing (default: 4)
    overlap: Overlap between tiles in pixels

    Returns
    -------
    dilated_labels : per cell segmentation as numpy array
    """
    from ..tiling.core import _apply_tiling_to_segmentation

    # If tiling is requested, use tiled segmentation
    if num_tiles is not None:
        return _apply_tiling_to_segmentation(
            _watershed_core, img, seg_channels,
            num_tiles=num_tiles, overlap=overlap, auto_tile_memory_mb=auto_tile_memory_mb
        )

    # Regular segmentation
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
