# 🚀 Quick Start Guide

## Overview

SPEX is a spatial omics analysis library that implements state-of-the-art tissue segmentation techniques. This guide will help you get started with basic image processing and segmentation workflows.

## Basic Workflow

### 1. Import the Library

```python
import spex as sp
import numpy as np
import matplotlib.pyplot as plt
```

### 2. Load an Image

```python
# Load a multi-channel image
Image, channel = sp.load_image('your_image.tiff')

print(f"Image shape: {Image.shape}")
print(f"Channels: {channel}")
```

### 3. Perform Segmentation

Choose one of the available segmentation methods:

#### Watershed Segmentation (Classic)
```python
# Simple watershed segmentation
labels = sp.watershed_classic(Image, [0])  # Use first channel
```

#### Cellpose Segmentation (AI-based)
```python
# AI-powered cell segmentation
labels = sp.cellpose_cellseg(Image, [0], diameter=30)
```

#### StarDist Segmentation (Star-convex)
```python
# Star-convex object detection
labels = sp.stardist_cellseg(Image, [0])
```

### 4. Extract Features

```python
# Extract per-cell expression data
features = sp.feature_extraction_adata(Image, labels, channel)

print(f"Extracted features for {len(features)} cells")
```

### 5. Visualize Results

```python
# Plot segmentation results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(Image[:, :, 0], cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# Segmentation labels
axes[1].imshow(labels, cmap='tab20')
axes[1].set_title('Segmentation Labels')
axes[1].axis('off')

# Overlay
axes[2].imshow(Image[:, :, 0], cmap='gray')
axes[2].imshow(labels, cmap='tab20', alpha=0.5)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

## Complete Example

Here's a complete workflow combining all steps:

```python
import spex as sp
import matplotlib.pyplot as plt

# Load image
Image, channel = sp.load_image('sample_image.tiff')

# Preprocess (optional)
Image_processed = sp.background_subtract(Image, [0])
Image_processed = sp.median_denoise(Image_processed, [0])

# Segment
labels = sp.watershed_classic(Image_processed, [0])

# Post-process
labels = sp.remove_small_objects(labels, min_size=50)
labels = sp.remove_large_objects(labels, max_size=1000)

# Extract features
features = sp.feature_extraction_adata(Image, labels, channel)

print(f"Successfully processed {len(features)} cells!")
```

## Next Steps

- 📖 [Segmentation Tutorial](../tutorials/segmentation-basics.md)
- 🎯 [Clustering Guide](../tutorials/clustering-guide.md)
- 🧬 [Spatial Analysis](../tutorials/spatial-analysis.md)
- 📊 [Complete Examples](../examples/)

## Data Formats

SPEX supports various image formats:
- **TIFF**: Multi-channel images (recommended)
- **PNG**: Single-channel images
- **JPEG**: Single-channel images (lossy)

For best results, use multi-channel TIFF files with proper channel information.

## Performance Tips

1. **Memory Management**: Large images may require significant RAM
2. **Batch Processing**: Use loops for multiple images
3. **GPU Acceleration**: Some methods support GPU acceleration
4. **Image Tiling**: Process large images in tiles if needed

## Getting Help

- 📚 [API Reference](../api-reference/)
- 🐛 [Troubleshooting](../reference/troubleshooting.md)
- 💬 [GitHub Issues](https://github.com/genentech/spex-tools/issues)
