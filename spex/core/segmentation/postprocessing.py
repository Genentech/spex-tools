import numpy as np
from skimage.measure import label, regionprops_table, regionprops, find_contours
from skimage.filters import median
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, expand_labels
from skimage.morphology import dilation, erosion, disk, binary_dilation
import math
import pandas as pd
from anndata import AnnData
from spex.core.utils import to_uint8
import re


def simulate_cell(labels, dist):
    """Dilate labels by fixed amount to simulate cells

    Parameters
    ----------
    labels: numpy array of segmentation labels
    dist: number of pixels to dilate

    Returns
    -------
    out : 2D label numpy array with simulated cells

    """

    return expand_labels(labels, dist)


def rescue_cells(image, seg_channels, label_ling):
    """Rescue/Segment cells that deep learning approach may have missed

    Parameters
    ----------
    image : raw image 2d numpy array
    seg_channels: list of indices to use for nuclear segmentation
    label_ling: numpy array of segmentation labels

    Returns
    -------
    combine_label : 2D numpy array with added cells

    """
    temp2 = np.zeros((image.shape[1], image.shape[2]))
    for i in seg_channels:
        try:
            temp = image[i]
            temp2 = temp + temp2
        except IndexError:
            print("oops")

    seg_image = temp2 / len(seg_channels)

    props = regionprops_table(
        label_ling, intensity_image=seg_image, properties=["mean_intensity", "area"]
    )

    if len(props["mean_intensity"]) == 0:
        meanint_cell = 0
        meansize_cell = 0
    else:
        meanint_cell = np.mean(props["mean_intensity"])
        meansize_cell = np.mean(props["area"])

    radius = math.floor(math.sqrt(meansize_cell / 3.14) * 0.5)
    threshold = meanint_cell * 0.5

    med = median(seg_image, disk(max(1, radius)))

    local_max = peak_local_max(
        med, min_distance=max(1, math.floor(radius * 1.2)), indices=False
    )

    mask = med > threshold

    mask = binary_dilation(mask, np.ones((2, 2)))
    masked_peaks = local_max * mask

    seed_label = label(masked_peaks)

    watershed_labels = watershed(
        image=-med, markers=seed_label, mask=mask, watershed_line=True, compactness=20
    )

    selem = disk(1)
    dilated_labels = erosion(watershed_labels, selem)
    selem = disk(1)
    dilated_labels = dilation(dilated_labels, selem)

    labels2 = label_ling > 0

    props = regionprops(dilated_labels, intensity_image=labels2)

    labels_store = np.arange(np.max(dilated_labels) + 1)

    for cell in props:
        if cell.mean_intensity >= 0.03:
            labels_store[cell.label] = 0

    final_mask = labels_store[dilated_labels]

    combine_label = label_ling + final_mask
    combine_label = label(combine_label)

    return combine_label


def remove_large_objects(segments, maxsize):
    """Remove large segmented objects

    Parameters
    ----------
    segments: numpy array of segmentation labels
    maxsize: max pixel size

    Returns
    -------
    out : 2D label numpy array

    """

    out = np.copy(segments)
    component_sizes = np.bincount(segments.ravel())

    too_large = component_sizes > maxsize
    too_large_mask = too_large[segments]
    out[too_large_mask] = 0

    return out


def remove_small_objects(segments, minsize):
    """Remove small segmented objects

    Parameters
    ----------
    segments: numpy array of segmentation labels
    minsize: minimum pixel size

    Returns
    -------
    out : 2D label numpy array

    """

    out = np.copy(segments)
    component_sizes = np.bincount(segments.ravel())

    too_small = component_sizes < minsize
    too_small_mask = too_small[segments]
    out[too_small_mask] = 0

    return out


def feature_extraction(img, labels, all_channels):
    """Extract per cell expression for all channels

    Parameters
    ----------
    img : np.ndarray
        Multichannel image as numpy array with shape (C, X, Y).
    labels : np.ndarray
        Segmentation labels as numpy array with shape (X, Y).
    all_channels : list
        List of channel names.

    Returns
    -------
    pd.DataFrame
        DataFrame with per-cell expression data for all channels.
    """
    label_array = labels

    props = regionprops_table(
        label_array,
        intensity_image=np.transpose(img, (1, 2, 0)),
        properties=["label", "area", "centroid", "mean_intensity"],
    )
    perCellData = pd.DataFrame(props)

    perCellData.columns = [
        "cell_id",
        "area_pixels",
        "Y",
        "X",
    ] + all_channels  # rename columns

    return perCellData


def feature_extraction_adata(img, labels, all_channels):
    """Extract per cell expression for all channels

    Returns
    -------
    perCellanndata: anndata single-cell object with all data
    :param labels:
    :param img: Multichannel image as numpy array
    :param all_channels:

    """

    label_array = labels
    import re

    props = regionprops_table(
        label_array,
        intensity_image=np.transpose(img, (1, 2, 0)),
        properties=["label", "area", "centroid", "mean_intensity"],
    )
    perCellData = pd.DataFrame(props)

    perCellData.columns = [
        "cell_id",
        "area_pixels",
        "Y",
        "X",
    ] + all_channels  # rename columns

    coordinates = np.array([k for k in perCellData[["X", "Y"]].values.tolist()])

    props = regionprops(label_array)
    ordered_contours = []
    for region in props:
        # Find contours of the region
        contours = find_contours(label_array == region.label, 0.5)

        if len(contours) > 0:
            # Select the contour with the largest area
            contour = max(contours, key=lambda contour: contour.shape[0])
        else:
            contour = contours[0]

        centroid = np.mean(contour, axis=0)

        # Calculate angles of the points with respect to the centroid
        angles = np.arctan2(contour[:, 0] - centroid[0], contour[:, 1] - centroid[1])

        # Sort points based on angles
        sorted_indices = np.argsort(angles)
        sorted_contour = contour[sorted_indices]

        ordered_contours.append(sorted_contour)

    adata = AnnData(
        perCellData[all_channels], obsm={"spatial": coordinates}, dtype="float32"
    )
    adata.obsm["cell_polygon"] = np.array(ordered_contours, dtype=object)

    adata.obs["Cell_ID"] = perCellData[["cell_id"]].values
    adata.obs["Nucleus_area"] = perCellData[["area_pixels"]].values
    adata.obs["x_coordinate"] = perCellData[["X"]].values
    adata.obs["y_coordinate"] = perCellData[["Y"]].values

    adata.layers["X_uint8"] = to_uint8(
        adata.X, norm_along="global"
    )  # vitessce only supports 8bit expression
    channel_index_map = {
        re.sub("[^0-9a-zA-Z]", "", ch).lower().replace("target", ""): idx
        for idx, ch in enumerate(all_channels)
    }
    adata.uns["channel_index_map"] = channel_index_map

    return adata
