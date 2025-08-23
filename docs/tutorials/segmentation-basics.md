# 🖼️ Segmentation Basics Tutorial

## Overview

This tutorial covers the fundamental segmentation methods available in SPEX. You'll learn how to segment cells and objects from microscopy images using different approaches.

## Prerequisites

```python
import spex as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
```

## Image Loading and Preprocessing

### Loading Images

```python
# Load a multi-channel image
Image, channel = sp.load_image('sample_image.tiff')

print(f"Image shape: {Image.shape}")
print(f"Channels: {channel}")

# Display image
fig, axes = plt.subplots(1, len(channel), figsize=(4*len(channel), 4))
if len(channel) == 1:
    axes = [axes]
    
for i, ch in enumerate(channel):
    axes[i].imshow(Image[:, :, i], cmap='gray')
    axes[i].set_title(f'Channel: {ch}')
    axes[i].axis('off')
plt.tight_layout()
plt.show()
```

### Preprocessing

```python
# Background subtraction
Image_bg = sp.background_subtract(Image, [0])

# Denoising options
Image_median = sp.median_denoise(Image_bg, [0])
Image_nlm = sp.nlm_denoise(Image_bg, [0])

# Compare preprocessing results
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(Image[:, :, 0], cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(Image_bg[:, :, 0], cmap='gray')
axes[1].set_title('Background Subtracted')
axes[1].axis('off')

axes[2].imshow(Image_median[:, :, 0], cmap='gray')
axes[2].set_title('Median Denoised')
axes[2].axis('off')

axes[3].imshow(Image_nlm[:, :, 0], cmap='gray')
axes[3].set_title('NLM Denoised')
axes[3].axis('off')

plt.tight_layout()
plt.show()
```

## Segmentation Methods

### 1. Watershed Segmentation (Classic)

Watershed is a traditional image processing method that treats pixel values as elevation and finds watershed lines.

```python
# Basic watershed segmentation
labels_watershed = sp.watershed_classic(Image, [0])

print(f"Number of cells detected: {labels_watershed.max()}")

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(Image[:, :, 0], cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(labels_watershed, cmap='tab20')
axes[1].set_title(f'Watershed Labels ({labels_watershed.max()} cells)')
axes[1].axis('off')

axes[2].imshow(Image[:, :, 0], cmap='gray')
axes[2].imshow(labels_watershed, cmap='tab20', alpha=0.5)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

**Advantages:**
- Fast and computationally efficient
- Works well with clear cell boundaries
- No external dependencies

**Disadvantages:**
- Sensitive to noise
- May over-segment or under-segment
- Requires good preprocessing

### 2. Cellpose Segmentation (AI-based)

Cellpose uses deep learning to segment cells with high accuracy.

```python
# Cellpose segmentation
labels_cellpose = sp.cellpose_cellseg(Image, [0], diameter=30)

print(f"Number of cells detected: {labels_cellpose.max()}")

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(Image[:, :, 0], cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(labels_cellpose, cmap='tab20')
axes[1].set_title(f'Cellpose Labels ({labels_cellpose.max()} cells)')
axes[1].axis('off')

axes[2].imshow(Image[:, :, 0], cmap='gray')
axes[2].imshow(labels_cellpose, cmap='tab20', alpha=0.5)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

**Parameters:**
- `diameter`: Expected cell diameter in pixels
- `channels`: List of channel indices to use
- `flow_threshold`: Flow field threshold (default: 0.4)
- `cellprob_threshold`: Cell probability threshold (default: 0)

**Advantages:**
- High accuracy for cell segmentation
- Robust to noise and variations
- Works with various cell types

**Disadvantages:**
- Requires GPU for optimal performance
- Model download required on first use
- Slower than traditional methods

### 3. StarDist Segmentation (Star-convex)

StarDist detects star-convex objects using deep learning.

```python
# StarDist segmentation
labels_stardist = sp.stardist_cellseg(Image, [0])

print(f"Number of cells detected: {labels_stardist.max()}")

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(Image[:, :, 0], cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(labels_stardist, cmap='tab20')
axes[1].set_title(f'StarDist Labels ({labels_stardist.max()} cells)')
axes[1].axis('off')

axes[2].imshow(Image[:, :, 0], cmap='gray')
axes[2].imshow(labels_stardist, cmap='tab20', alpha=0.5)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

**Advantages:**
- Excellent for star-convex objects
- Good boundary accuracy
- Works well with nuclear staining

**Disadvantages:**
- Requires star-convex assumption
- May not work for irregular shapes
- Model download required

## Post-processing

### Cleaning Segmentation Results

```python
# Remove small objects (likely noise)
labels_clean = sp.remove_small_objects(labels_cellpose, min_size=50)

# Remove large objects (likely artifacts)
labels_clean = sp.remove_large_objects(labels_clean, max_size=1000)

print(f"Cells after cleaning: {labels_clean.max()}")

# Visualize cleaning results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(labels_cellpose, cmap='tab20')
axes[0].set_title(f'Before Cleaning ({labels_cellpose.max()} cells)')
axes[0].axis('off')

axes[1].imshow(labels_clean, cmap='tab20')
axes[1].set_title(f'After Cleaning ({labels_clean.max()} cells)')
axes[1].axis('off')

axes[2].imshow(Image[:, :, 0], cmap='gray')
axes[2].imshow(labels_clean, cmap='tab20', alpha=0.5)
axes[2].set_title('Final Result')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

### Cell Rescue

```python
# Rescue missing cells
labels_rescued = sp.rescue_cells(Image, labels_clean, [0])

print(f"Cells after rescue: {labels_rescued.max()}")

# Visualize rescue results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(labels_clean, cmap='tab20')
axes[0].set_title(f'Before Rescue ({labels_clean.max()} cells)')
axes[0].axis('off')

axes[1].imshow(labels_rescued, cmap='tab20')
axes[1].set_title(f'After Rescue ({labels_rescued.max()} cells)')
axes[1].axis('off')

axes[2].imshow(Image[:, :, 0], cmap='gray')
axes[2].imshow(labels_rescued, cmap='tab20', alpha=0.5)
axes[2].set_title('Final Result')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

## Comparison of Methods

```python
# Compare all methods
methods = {
    'Watershed': labels_watershed,
    'Cellpose': labels_cellpose,
    'StarDist': labels_stardist,
    'Cleaned': labels_clean,
    'Rescued': labels_rescued
}

# Create comparison plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i, (name, labels) in enumerate(methods.items()):
    row = i // 3
    col = i % 3
    
    axes[row, col].imshow(Image[:, :, 0], cmap='gray')
    axes[row, col].imshow(labels, cmap='tab20', alpha=0.5)
    axes[row, col].set_title(f'{name} ({labels.max()} cells)')
    axes[row, col].axis('off')

# Remove empty subplot
axes[1, 2].remove()

plt.tight_layout()
plt.show()

# Print statistics
print("Method Comparison:")
print("-" * 40)
for name, labels in methods.items():
    n_cells = labels.max()
    areas = np.bincount(labels.ravel())[1:]
    mean_area = areas.mean() if len(areas) > 0 else 0
    print(f"{name:10}: {n_cells:4d} cells, mean area: {mean_area:6.1f} pixels")
```

## Quality Assessment

### Cell Size Distribution

```python
# Analyze cell size distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of cell areas
areas = np.bincount(labels_rescued.ravel())[1:]
axes[0].hist(areas, bins=30, alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Cell Area (pixels)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Cell Size Distribution')

# Box plot comparison
area_data = []
method_names = []
for name, labels in methods.items():
    areas = np.bincount(labels.ravel())[1:]
    if len(areas) > 0:
        area_data.extend(areas)
        method_names.extend([name] * len(areas))

import pandas as pd
df = pd.DataFrame({'Method': method_names, 'Area': area_data})
sns.boxplot(data=df, x='Method', y='Area', ax=axes[1])
axes[1].set_title('Cell Size by Method')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

### Segmentation Quality Metrics

```python
# Calculate quality metrics
def calculate_metrics(labels):
    areas = np.bincount(labels.ravel())[1:]
    if len(areas) == 0:
        return {'n_cells': 0, 'mean_area': 0, 'std_area': 0, 'coverage': 0}
    
    coverage = (labels > 0).sum() / labels.size * 100
    return {
        'n_cells': len(areas),
        'mean_area': areas.mean(),
        'std_area': areas.std(),
        'coverage': coverage
    }

# Compare metrics
print("Quality Metrics:")
print("-" * 60)
print(f"{'Method':<12} {'Cells':<6} {'Mean Area':<10} {'Std Area':<10} {'Coverage':<10}")
print("-" * 60)

for name, labels in methods.items():
    metrics = calculate_metrics(labels)
    print(f"{name:<12} {metrics['n_cells']:<6} {metrics['mean_area']:<10.1f} "
          f"{metrics['std_area']:<10.1f} {metrics['coverage']:<10.1f}%")
```

## Best Practices

### Method Selection

1. **Watershed**: Use for simple, well-separated cells with clear boundaries
2. **Cellpose**: Use for complex cell shapes and noisy images
3. **StarDist**: Use for nuclear staining and star-convex objects

### Parameter Tuning

```python
# Watershed parameters
labels_ws1 = sp.watershed_classic(Image, [0])  # Default
labels_ws2 = sp.watershed_classic(Image, [0], min_distance=10)  # Adjust sensitivity

# Cellpose parameters
labels_cp1 = sp.cellpose_cellseg(Image, [0], diameter=20)  # Small cells
labels_cp2 = sp.cellpose_cellseg(Image, [0], diameter=40)  # Large cells

# Compare parameter effects
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

results = [
    ('Watershed Default', labels_ws1),
    ('Watershed Adjusted', labels_ws2),
    ('Cellpose Small', labels_cp1),
    ('Cellpose Large', labels_cp2),
    ('StarDist', labels_stardist),
    ('Final Result', labels_rescued)
]

for i, (name, labels) in enumerate(results):
    row = i // 3
    col = i % 3
    
    axes[row, col].imshow(Image[:, :, 0], cmap='gray')
    axes[row, col].imshow(labels, cmap='tab20', alpha=0.5)
    axes[row, col].set_title(f'{name}\n({labels.max()} cells)')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

## Next Steps

- 🎯 [Clustering Guide](clustering-guide.md) - Analyze cell types
- 🧬 [Spatial Analysis](spatial-analysis.md) - Study spatial relationships
- 📊 [Complete Examples](../examples/) - End-to-end workflows
- 🔧 [API Reference](../api-reference/) - Detailed function documentation
