# 🔧 Segmentation API Reference

This document provides comprehensive API documentation for all segmentation functions in the SPEX library.

## Table of Contents

1. [Image Loading](#image-loading)
2. [Image Preprocessing](#image-preprocessing)
3. [Cell Segmentation](#cell-segmentation)
4. [Post-processing](#post-processing)
5. [Feature Extraction](#feature-extraction)
6. [Utilities](#utilities)

## Image Loading

### `load_image()`

Load and process image files for analysis.

```python
spex.load_image(image_path, channel_names=None)
```

**Parameters:**
- `image_path` (str): Path to the image file
- `channel_names` (list, optional): List of channel names. If None, auto-generated names will be used

**Returns:**
- `Image` (numpy.ndarray): Image array with shape (height, width, channels)
- `channel` (list): List of channel names

**Example:**
```python
import spex as sp

# Load multi-channel TIFF
Image, channel = sp.load_image('sample.tiff')
print(f"Image shape: {Image.shape}")
print(f"Channels: {channel}")

# Load with custom channel names
Image, channel = sp.load_image('sample.tiff', ['DAPI', 'CD3', 'CD8'])
```

**Supported Formats:**
- TIFF (multi-channel, recommended)
- PNG (single-channel)
- JPEG (single-channel)
- BMP (single-channel)

---

## Image Preprocessing

### `background_subtract()`

Subtract background from image channels.

```python
spex.background_subtract(Image, channels, method='rolling_ball', radius=50)
```

**Parameters:**
- `Image` (numpy.ndarray): Input image array
- `channels` (list): List of channel indices to process
- `method` (str): Background subtraction method ('rolling_ball' or 'gaussian')
- `radius` (int): Radius for background estimation

**Returns:**
- `numpy.ndarray`: Background-subtracted image

**Example:**
```python
# Subtract background from first channel
Image_bg = sp.background_subtract(Image, [0])

# Subtract from multiple channels
Image_bg = sp.background_subtract(Image, [0, 1, 2], radius=30)
```

---

### `median_denoise()`

Apply median filtering for noise reduction.

```python
spex.median_denoise(Image, channels, kernel_size=3)
```

**Parameters:**
- `Image` (numpy.ndarray): Input image array
- `channels` (list): List of channel indices to process
- `kernel_size` (int): Size of median filter kernel

**Returns:**
- `numpy.ndarray`: Denoised image

**Example:**
```python
# Denoise first channel with 3x3 kernel
Image_denoised = sp.median_denoise(Image, [0])

# Use larger kernel for more aggressive denoising
Image_denoised = sp.median_denoise(Image, [0], kernel_size=5)
```

---

### `nlm_denoise()`

Apply non-local means denoising.

```python
spex.nlm_denoise(Image, channels, h=10, template_window_size=7, search_window_size=21)
```

**Parameters:**
- `Image` (numpy.ndarray): Input image array
- `channels` (list): List of channel indices to process
- `h` (float): Filter strength parameter
- `template_window_size` (int): Size of template window
- `search_window_size` (int): Size of search window

**Returns:**
- `numpy.ndarray`: Denoised image

**Example:**
```python
# Apply non-local means denoising
Image_nlm = sp.nlm_denoise(Image, [0], h=15)

# Adjust parameters for different noise levels
Image_nlm = sp.nlm_denoise(Image, [0], h=20, template_window_size=5)
```

---

## Cell Segmentation

### `cellpose_cellseg()`

Perform cell segmentation using Cellpose.

```python
spex.cellpose_cellseg(
    Image,
    seg_channels=[0],
    diameter=30,
    scaling=1,
    auto_tile_memory_mb=500,
    tile_size=None,
)
```

**Parameters:**
- `Image` (numpy.ndarray): Input image array in (C, H, W) order.
- `seg_channels` (Sequence[int]): Channel indices to use for segmentation.
- `diameter` (int): Expected object diameter passed to Cellpose.
- `scaling` (int): Integer scaling factor applied before inference.
- `num_tiles` (int, optional): Deprecated tile-count hint.
- `overlap` (int, optional): Overlap in pixels when tiling is enabled.
- `auto_tile_memory_mb` (float, optional): Enable tiling automatically once the estimated memory footprint exceeds this threshold (default 500 MB).
- `tile_size` (tuple, optional): Explicit tile size (height, width) for tiled execution.

**Returns:**
- `numpy.ndarray`: Segmentation labels

**Example:**
```python
# Basic cell segmentation
labels = sp.cellpose_cellseg(Image, seg_channels=[0])

# Force auto-tiling when memory is limited
labels = sp.cellpose_cellseg(
    Image,
    seg_channels=[0],
    diameter=30,
    scaling=1,
    auto_tile_memory_mb=200,
)

# Explicit tile size
labels = sp.cellpose_cellseg(
    Image,
    seg_channels=[0],
    diameter=30,
    scaling=1,
    tile_size=(512, 512),
    overlap=64,
)
```

### `cellpose_cellseg_tiled()`

Perform cell segmentation using Cellpose with tiled processing for memory efficiency.

This function divides large images into overlapping tiles, processes each tile with Cellpose, and reassembles the results. This approach allows processing of very large images that would otherwise cause memory issues.

```python
spex.cellpose_cellseg_tiled(
    img,
    seg_channels,
    diameter,
    scaling,
    tile_size=(512, 512),
    overlap=64,
)
```

**Parameters:**
- `img` (numpy.ndarray): Multichannel image as numpy array of shape (C, H, W)
- `seg_channels` (Sequence[int]): Channel indices to use for segmentation
- `diameter` (int): Typical size of nucleus in pixels
- `scaling` (int): Integer scaling factor for processing
- `tile_size` (tuple, optional): Size of each tile (height, width). If ``None`` the tile size is auto-calculated.
- `overlap` (int, optional): Overlap between tiles in pixels. If ``None`` a 12.5% overlap is used.

**Returns:**
- `numpy.ndarray`: Per-cell segmentation as numpy array of shape (H, W)

**Example:**
```python
import spex as sp
import numpy as np

# Create large image
img = np.random.rand(1, 2048, 2048).astype(np.float32)

# Segment with tiled processing
labels = sp.cellpose_cellseg_tiled(
    img, 
    seg_channels=[0], 
    diameter=30, 
    scaling=1,
    tile_size=(512, 512),
    overlap=64
)

print(f"Segmentation shape: {labels.shape}")
print(f"Number of cells: {labels.max()}")

# For very large images, use smaller tiles
labels = sp.cellpose_cellseg_tiled(
    img, 
    seg_channels=[0], 
    diameter=30, 
    scaling=1,
    tile_size=(256, 256),
    overlap=32
)

# Preprocess noisy images (e.g., denoising) before running tiled segmentation
```

**Memory Efficiency:**
- Processes images in tiles to reduce memory usage
- Automatically falls back to regular segmentation for small images
- Handles edge cases and tile boundaries correctly

**Performance Tips:**
- Use larger tiles (512x512) for better segmentation quality
- Use smaller tiles (256x256) for very large images or limited memory
- Increase overlap for better boundary handling
- Preprocess noisy images (denoising/background subtraction) before running tiled segmentation

---

### `stardist_cellseg()`

Perform cell segmentation using StarDist.

```python
spex.stardist_cellseg(Image, seg_channels=[0], model_type='2D_versatile_fluo', 
                     nms_thresh=0.4, prob_thresh=0.5, scale=(1, 1))
```

**Parameters:**
- `Image` (numpy.ndarray): Input image array
- `seg_channels` (list): List of channel indices to use for segmentation
- `model_type` (str): StarDist model type
- `nms_thresh` (float): Non-maximum suppression threshold
- `prob_thresh` (float): Probability threshold
- `scale` (tuple): Scale factors for x and y dimensions

**Returns:**
- `numpy.ndarray`: Segmentation labels

**Example:**
```python
# Basic StarDist segmentation
labels = sp.stardist_cellseg(Image, seg_channels=[0])

# Adjust thresholds for better results
labels = sp.stardist_cellseg(Image, seg_channels=[0], nms_thresh=0.3, prob_thresh=0.6)
```

---

### `watershed_classic()`

Perform classical watershed segmentation.

```python
spex.watershed_classic(Image, seg_channels=[0], threshold_method='otsu', 
                      min_distance=10, min_size=50)
```

**Parameters:**
- `Image` (numpy.ndarray): Input image array
- `seg_channels` (list): List of channel indices to use for segmentation
- `threshold_method` (str): Thresholding method ('otsu', 'adaptive', 'manual')
- `min_distance` (int): Minimum distance between markers
- `min_size` (int): Minimum cell size in pixels

**Returns:**
- `numpy.ndarray`: Segmentation labels

**Example:**
```python
# Basic watershed segmentation
labels = sp.watershed_classic(Image, seg_channels=[0])

# Use adaptive thresholding
labels = sp.watershed_classic(Image, seg_channels=[0], threshold_method='adaptive')
```

---

## Post-processing

### `rescue_cells()`

Rescue cells that were missed during initial segmentation.

```python
spex.rescue_cells(Image, labels, seg_channels=[0], min_size=15, max_size=1000)
```

**Parameters:**
- `Image` (numpy.ndarray): Input image array
- `labels` (numpy.ndarray): Initial segmentation labels
- `seg_channels` (list): List of channel indices to use
- `min_size` (int): Minimum cell size to rescue
- `max_size` (int): Maximum cell size to rescue

**Returns:**
- `numpy.ndarray`: Updated segmentation labels

**Example:**
```python
# Rescue small cells
labels_rescued = sp.rescue_cells(Image, labels, min_size=10)

# Rescue cells within size range
labels_rescued = sp.rescue_cells(Image, labels, min_size=20, max_size=500)
```

---

### `simulate_cell()`

Simulate cell shapes for testing and validation.

```python
spex.simulate_cell(shape, n_cells=100, min_radius=5, max_radius=15, noise_level=0.1)
```

**Parameters:**
- `shape` (tuple): Image shape (height, width)
- `n_cells` (int): Number of cells to simulate
- `min_radius` (int): Minimum cell radius
- `max_radius` (int): Maximum cell radius
- `noise_level` (float): Noise level in the image

**Returns:**
- `numpy.ndarray`: Simulated cell image
- `numpy.ndarray`: Ground truth labels

**Example:**
```python
# Simulate cells for testing
sim_image, sim_labels = sp.simulate_cell((512, 512), n_cells=50)

# Create more realistic simulation
sim_image, sim_labels = sp.simulate_cell((1024, 1024), n_cells=200, noise_level=0.05)
```

---

### `remove_large_objects()`

Remove objects larger than specified size.

```python
spex.remove_large_objects(labels, max_size=1000)
```

**Parameters:**
- `labels` (numpy.ndarray): Segmentation labels
- `max_size` (int): Maximum object size in pixels

**Returns:**
- `numpy.ndarray`: Filtered segmentation labels

**Example:**
```python
# Remove large objects
labels_filtered = sp.remove_large_objects(labels, max_size=800)

# Very aggressive filtering
labels_filtered = sp.remove_large_objects(labels, max_size=500)
```

---

### `remove_small_objects()`

Remove objects smaller than specified size.

```python
spex.remove_small_objects(labels, min_size=15)
```

**Parameters:**
- `labels` (numpy.ndarray): Segmentation labels
- `min_size` (int): Minimum object size in pixels

**Returns:**
- `numpy.ndarray`: Filtered segmentation labels

**Example:**
```python
# Remove small objects
labels_filtered = sp.remove_small_objects(labels, min_size=20)

# Very strict filtering
labels_filtered = sp.remove_small_objects(labels, min_size=50)
```

---

## Feature Extraction

### `feature_extraction()`

Extract features from segmented cells.

```python
spex.feature_extraction(Image, labels, channels=None, features=['area', 'perimeter', 'intensity'])
```

**Parameters:**
- `Image` (numpy.ndarray): Input image array
- `labels` (numpy.ndarray): Segmentation labels
- `channels` (list, optional): List of channel indices to analyze
- `features` (list): List of features to extract

**Returns:**
- `pandas.DataFrame`: Feature table

**Available Features:**
- `area`: Cell area in pixels
- `perimeter`: Cell perimeter in pixels
- `intensity`: Mean intensity per channel
- `eccentricity`: Cell eccentricity
- `solidity`: Cell solidity
- `extent`: Cell extent ratio

**Example:**
```python
# Extract basic features
features_df = sp.feature_extraction(Image, labels)

# Extract specific features
features_df = sp.feature_extraction(Image, labels, 
                                   features=['area', 'intensity', 'eccentricity'])

# Analyze specific channels
features_df = sp.feature_extraction(Image, labels, channels=[0, 1])
```

---

### `feature_extraction_adata()`

Extract features and create AnnData object.

```python
spex.feature_extraction_adata(Image, labels, channels=None, features=['area', 'perimeter', 'intensity'])
```

**Parameters:**
- `Image` (numpy.ndarray): Input image array
- `labels` (numpy.ndarray): Segmentation labels
- `channels` (list, optional): List of channel indices to analyze
- `features` (list): List of features to extract

**Returns:**
- `anndata.AnnData`: AnnData object with features

**Example:**
```python
# Create AnnData object with features
adata = sp.feature_extraction_adata(Image, labels)

# Extract specific features
adata = sp.feature_extraction_adata(Image, labels, 
                                   features=['area', 'intensity', 'eccentricity'])
```

---

## Utilities

### `download_cellpose_models()`

Download Cellpose models for offline use.

```python
spex.download_cellpose_models()
```

**Returns:**
- `None`: Downloads models to default location

**Example:**
```python
# Download all Cellpose models
sp.download_cellpose_models()
```

---

### `delete_cellpose_models()`

Delete downloaded Cellpose models.

```python
spex.delete_cellpose_models()
```

**Returns:**
- `None`: Removes models from default location

**Example:**
```python
# Clean up downloaded models
sp.delete_cellpose_models()
```

---

### `to_uint8()`

Convert image to uint8 format.

```python
spex.to_uint8(Image, normalize=True)
```

**Parameters:**
- `Image` (numpy.ndarray): Input image array
- `normalize` (bool): Whether to normalize values to 0-255 range

**Returns:**
- `numpy.ndarray`: uint8 image array

**Example:**
```python
# Convert to uint8
Image_uint8 = sp.to_uint8(Image)

# Convert without normalization
Image_uint8 = sp.to_uint8(Image, normalize=False)
```

---

## Complete Segmentation Pipeline

Here's a complete example of a typical segmentation workflow:

```python
import spex as sp
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image
Image, channels = sp.load_image('sample.tiff', ['DAPI', 'CD3', 'CD8'])
print(f"Loaded image with shape: {Image.shape}")

# 2. Preprocess
Image_bg = sp.background_subtract(Image, [0], radius=30)
Image_denoised = sp.median_denoise(Image_bg, [0], kernel_size=3)

# 3. Segment cells
labels = sp.cellpose_cellseg(Image_denoised, seg_channels=[0], diameter=20)

# 4. Post-process
labels_filtered = sp.remove_small_objects(labels, min_size=15)
labels_filtered = sp.remove_large_objects(labels_filtered, max_size=1000)

# 5. Extract features
features_df = sp.feature_extraction(Image, labels_filtered, 
                                   features=['area', 'perimeter', 'intensity'])

# 6. Create AnnData object
adata = sp.feature_extraction_adata(Image, labels_filtered)

print(f"Segmentation complete! Found {len(np.unique(labels_filtered))-1} cells")
print(f"Features extracted: {features_df.shape}")
```

## Troubleshooting

### Common Issues and Solutions

**1. Segmentation fails or produces poor results**
- Try different `diameter` values
- Adjust `flow_threshold` and `cellprob_threshold`
- Use different `model_type` (cyto, nuclei, cyto2)
- Preprocess image with background subtraction and denoising

**2. Too many small objects**
- Increase `min_size` parameter
- Use `remove_small_objects()` function
- Adjust segmentation parameters

**3. Missing cells**
- Decrease `flow_threshold`
- Use `rescue_cells()` function
- Try different preprocessing methods

**4. Memory issues with large images**
- Process images in tiles
- Reduce image resolution
- Use uint8 format with `to_uint8()`

**5. Model download issues**
- Check internet connection
- Use `download_cellpose_models()` manually
- Verify disk space availability

---

**Next Steps:**
- [Clustering Guide](../tutorials/clustering-guide.md)
- [Spatial Analysis](../tutorials/spatial-analysis.md)
- [Complete Pipeline Example](../examples/complete-pipeline.md)
