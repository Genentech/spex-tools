# Image Segmentation

The SPEX segmentation module provides a comprehensive set of tools for analyzing and segmenting multi-channel microscopy images. All functions are optimized for spatial transcriptomics data.

## Image Loading

### load_image

Main function for loading images in various formats.

::: spex.core.segmentation.io.load_image
    options:
      show_root_heading: true
      show_source: false

**Usage Example:**

```python
from spex import load_image

# Load OME-TIFF file
array, channels = load_image("path/to/image.ome.tiff")
print(f"Image shape: {array.shape}")
print(f"Channels: {channels}")

# Load OME-ZARR file
array, channels = load_image("path/to/image.zarr/0")
print(f"Number of channels: {array.shape[0]}")
```

## Filtering and Preprocessing

### median_denoise

Median filtering for noise removal.

::: spex.core.segmentation.filters.median_denoise
    options:
      show_root_heading: true
      show_source: false

### nlm_denoise

Non-local means for advanced noise filtering.

::: spex.core.segmentation.filters.nlm_denoise
    options:
      show_root_heading: true
      show_source: false

### background_subtract

Background subtraction for improved contrast.

::: spex.core.segmentation.background_subtract.background_subtract
    options:
      show_root_heading: true
      show_source: false

## Segmentation Algorithms

### cellpose_cellseg

Cell segmentation using deep learning with Cellpose.

::: spex.core.segmentation.cellpose_cellseg.cellpose_cellseg
    options:
      show_root_heading: true
      show_source: false

**Usage Example:**

```python
from spex import load_image, cellpose_cellseg

# Load an image
array, channels = load_image("path/to/image.ome.tiff")

# Perform segmentation
labels = cellpose_cellseg(
    array,
    seg_channels=[0],  # Use first channel for segmentation
    diameter=30,       # Typical nucleus size
    scaling=1          # Scaling factor
)

print(f"Detected {labels.max()} cells")
```

### stardist_cellseg

Star-convex object-based segmentation using StarDist.

::: spex.core.segmentation.stardist.stardist_cellseg
    options:
      show_root_heading: true
      show_source: false

**Usage Example:**

```python
from spex import load_image, stardist_cellseg

# Load an image
array, channels = load_image("path/to/image.ome.tiff")

# Perform segmentation
labels = stardist_cellseg(
    array,
    seg_channels=[0],     # Channels for segmentation
    scaling=1,            # Scaling factor
    threshold=0.479071,   # Probability threshold
    _min=1.0,            # Lower percentile for normalization
    _max=98.5            # Upper percentile for normalization
)
```

### watershed_classic

Classical watershed segmentation.

::: spex.core.segmentation.watershed.watershed_classic
    options:
      show_root_heading: true
      show_source: false

## Postprocessing

### rescue_cells

Recovery of cells that might have been lost during segmentation.

::: spex.core.segmentation.postprocessing.rescue_cells
    options:
      show_root_heading: true
      show_source: false

### simulate_cell

Cell simulation for testing and validation.

::: spex.core.segmentation.postprocessing.simulate_cell
    options:
      show_root_heading: true
      show_source: false

### remove_small_objects

Removal of small objects.

::: spex.core.segmentation.postprocessing.remove_small_objects
    options:
      show_root_heading: true
      show_source: false

### remove_large_objects

Removal of large objects.

::: spex.core.segmentation.postprocessing.remove_large_objects
    options:
      show_root_heading: true
      show_source: false

### feature_extraction_adata

Feature extraction from segmented images to AnnData format.

::: spex.core.segmentation.postprocessing.feature_extraction_adata
    options:
      show_root_heading: true
      show_source: false

## Typical Segmentation Pipeline

```python
from spex import (
    load_image,
    median_denoise,
    background_subtract,
    cellpose_cellseg,
    remove_small_objects,
    feature_extraction_adata
)

# 1. Load image
array, channels = load_image("path/to/image.ome.tiff")

# 2. Preprocessing
denoised = median_denoise(array, [0, 1])  # Denoising for channels 0 and 1
subtracted = background_subtract(denoised, [0])  # Background subtraction

# 3. Segmentation
labels = cellpose_cellseg(subtracted, seg_channels=[0], diameter=30, scaling=1)

# 4. Postprocessing
cleaned_labels = remove_small_objects(labels, min_size=50)

# 5. Feature extraction
adata = feature_extraction_adata(array, cleaned_labels, channels)

print(f"Successfully segmented {cleaned_labels.max()} cells")
```

## Supported Formats

- **OME-TIFF** (.ome.tiff, .ome.tif) - Primary format for multi-channel images
- **OME-ZARR** (.zarr) - Modern format for large datasets
- **TIFF** - Standard image format (with limited metadata support)

## Recommendations

1. **Preprocessing**: Always apply noise filtering before segmentation
2. **Channel Selection**: Use channels with clear cell boundaries for segmentation
3. **Parameters**: Adjust diameter and thresholds based on your data
4. **Validation**: Visually check segmentation results
5. **Postprocessing**: Remove artifacts and objects that are too small or large
