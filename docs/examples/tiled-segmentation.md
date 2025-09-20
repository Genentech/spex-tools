# 🧩 Tiled Segmentation Examples

This document provides practical examples of using tiled segmentation for memory-efficient processing of large images.

## Table of Contents

1. [Basic Tiled Segmentation](#basic-tiled-segmentation)
2. [Large Image Processing](#large-image-processing)
3. [Memory Optimization](#memory-optimization)
4. [Performance Comparison](#performance-comparison)
5. [Troubleshooting](#troubleshooting)

## Basic Tiled Segmentation

### Simple Example

```python
import spex as sp
import numpy as np
import matplotlib.pyplot as plt

# Create a test image
img = np.random.rand(1, 512, 512).astype(np.float32)
img[0, 100:200, 100:200] = 255  # Add bright region

# Segment with tiled processing
labels = sp.cellpose_cellseg_tiled(
    img,
    seg_channels=[0],
    diameter=30,
    scaling=1,
    tile_size=(256, 256),
    overlap=32
)

print(f"Segmentation shape: {labels.shape}")
print(f"Number of cells detected: {labels.max()}")

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img[0], cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(labels, cmap='nipy_spectral')
axes[1].set_title('Segmentation Labels')
plt.show()
```

### Multi-channel Processing

```python
# Create multi-channel image
img = np.random.rand(3, 1024, 1024).astype(np.float32)
img[0, 200:400, 200:400] = 255  # Channel 0: nuclei
img[1, 300:500, 300:500] = 255  # Channel 1: cytoplasm

# Segment using multiple channels
labels = sp.cellpose_cellseg_tiled(
    img,
    seg_channels=[0, 1],  # Use both channels
    diameter=40,
    scaling=1,
    tile_size=(512, 512),
    overlap=64
)

print(f"Multi-channel segmentation completed: {labels.max()} cells")
```

## Large Image Processing

### Very Large Images (4K+)

```python
# Create very large image
img = np.random.rand(1, 4096, 4096).astype(np.float32)

# Add some structure
for i in range(0, 4096, 512):
    for j in range(0, 4096, 512):
        img[0, i:i+100, j:j+100] = 255

# Process with small tiles for memory efficiency
labels = sp.cellpose_cellseg_tiled(
    img,
    seg_channels=[0],
    diameter=50,
    scaling=1,
    tile_size=(256, 256),  # Small tiles
    overlap=32             # Small overlap
)

print(f"Large image processed: {labels.shape}, {labels.max()} cells")
```

### Memory-Efficient Processing

```python
import psutil
import gc

def process_large_image_memory_efficient(image_path):
    """Process large image with memory monitoring."""
    
    # Load image
    img, channels = sp.load_image(image_path)
    print(f"Loaded image: {img.shape}")
    
    # Monitor memory before processing
    memory_before = psutil.virtual_memory().used / 1024**3
    print(f"Memory before: {memory_before:.2f} GB")
    
    # Process with tiled segmentation
    labels = sp.cellpose_cellseg_tiled(
        img,
        seg_channels=[0],
        diameter=30,
        scaling=1,
        tile_size=(512, 512),
        overlap=64
    )
    
    # Monitor memory after processing
    memory_after = psutil.virtual_memory().used / 1024**3
    print(f"Memory after: {memory_after:.2f} GB")
    print(f"Memory increase: {memory_after - memory_before:.2f} GB")
    
    # Clean up
    del img
    gc.collect()
    
    return labels

# Usage
labels = process_large_image_memory_efficient('large_image.tiff')
```

## Memory Optimization

### Adaptive Tile Sizing

```python
def adaptive_tiled_segmentation(img, target_memory_gb=2.0):
    """Automatically choose tile size based on available memory."""
    
    # Estimate memory requirements
    img_size_gb = img.nbytes / 1024**3
    available_memory = psutil.virtual_memory().available / 1024**3
    
    # Calculate optimal tile size
    if img_size_gb < target_memory_gb:
        # Image is small enough, use regular segmentation
        return sp.cellpose_cellseg(img, seg_channels=[0], diameter=30, scaling=1)
    
    # Calculate tile size based on available memory
    max_tile_size = int(np.sqrt(available_memory * 0.5 * 1024**3 / img.nbytes * img.shape[1] * img.shape[2]))
    tile_size = min(max_tile_size, 512)  # Cap at 512
    overlap = max(32, tile_size // 8)    # Adaptive overlap
    
    print(f"Using tile size: {tile_size}x{tile_size}, overlap: {overlap}")
    
    return sp.cellpose_cellseg_tiled(
        img,
        seg_channels=[0],
        diameter=30,
        scaling=1,
        tile_size=(tile_size, tile_size),
        overlap=overlap
    )

# Usage
labels = adaptive_tiled_segmentation(img)
```

### Batch Processing with Tiling

```python
def batch_tiled_segmentation(image_paths, output_dir):
    """Process multiple images with tiled segmentation."""
    
    results = []
    
    for i, path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {path}")
        
        # Load image
        img, channels = sp.load_image(path)
        
        # Process with tiled segmentation
        labels = sp.cellpose_cellseg_tiled(
            img,
            seg_channels=[0],
            diameter=30,
            scaling=1,
            tile_size=(512, 512),
            overlap=64
        )
        
        # Save results
        output_path = os.path.join(output_dir, f"labels_{i:03d}.npy")
        np.save(output_path, labels)
        
        results.append({
            'input_path': path,
            'output_path': output_path,
            'n_cells': labels.max(),
            'shape': labels.shape
        })
        
        # Clean up
        del img, labels
        gc.collect()
    
    return results

# Usage
image_paths = ['image1.tiff', 'image2.tiff', 'image3.tiff']
results = batch_tiled_segmentation(image_paths, 'output_labels/')
```

## Performance Comparison

### Regular vs Tiled Segmentation

```python
import time

def compare_segmentation_methods(img):
    """Compare regular and tiled segmentation performance."""
    
    # Regular segmentation
    start_time = time.time()
    labels_regular = sp.cellpose_cellseg(img, seg_channels=[0], diameter=30, scaling=1)
    regular_time = time.time() - start_time
    
    # Tiled segmentation
    start_time = time.time()
    labels_tiled = sp.cellpose_cellseg_tiled(
        img,
        seg_channels=[0],
        diameter=30,
        scaling=1,
        tile_size=(512, 512),
        overlap=64
    )
    tiled_time = time.time() - start_time
    
    # Compare results
    print(f"Regular segmentation: {regular_time:.2f}s, {labels_regular.max()} cells")
    print(f"Tiled segmentation: {tiled_time:.2f}s, {labels_tiled.max()} cells")
    print(f"Speed ratio: {regular_time/tiled_time:.2f}x")
    
    # Check similarity
    similarity = np.sum(labels_regular == labels_tiled) / labels_regular.size
    print(f"Result similarity: {similarity:.3f}")
    
    return labels_regular, labels_tiled

# Usage
labels_regular, labels_tiled = compare_segmentation_methods(img)
```

### Memory Usage Comparison

```python
import tracemalloc

def compare_memory_usage(img):
    """Compare memory usage between regular and tiled segmentation."""
    
    # Regular segmentation
    tracemalloc.start()
    labels_regular = sp.cellpose_cellseg(img, seg_channels=[0], diameter=30, scaling=1)
    regular_memory = tracemalloc.get_traced_memory()[1] / 1024**2  # MB
    tracemalloc.stop()
    
    # Tiled segmentation
    tracemalloc.start()
    labels_tiled = sp.cellpose_cellseg_tiled(
        img,
        seg_channels=[0],
        diameter=30,
        scaling=1,
        tile_size=(512, 512),
        overlap=64
    )
    tiled_memory = tracemalloc.get_traced_memory()[1] / 1024**2  # MB
    tracemalloc.stop()
    
    print(f"Regular segmentation memory: {regular_memory:.2f} MB")
    print(f"Tiled segmentation memory: {tiled_memory:.2f} MB")
    print(f"Memory reduction: {(regular_memory - tiled_memory)/regular_memory*100:.1f}%")
    
    return labels_regular, labels_tiled

# Usage
labels_regular, labels_tiled = compare_memory_usage(img)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues

```python
# Problem: Out of memory error
# Solution: Use smaller tiles and preprocess the image beforehand if needed

labels = sp.cellpose_cellseg_tiled(
    img,
    seg_channels=[0],
    diameter=30,
    scaling=1,
    tile_size=(256, 256),  # Smaller tiles
    overlap=32             # Smaller overlap
)
```

#### 2. Poor Segmentation Quality

```python
# Problem: Poor segmentation at tile boundaries
# Solution: Increase overlap and use larger tiles

labels = sp.cellpose_cellseg_tiled(
    img,
    seg_channels=[0],
    diameter=30,
    scaling=1,
    tile_size=(512, 512),  # Larger tiles
    overlap=128            # Larger overlap
)
```

#### 3. Slow Processing

```python
# Problem: Processing is too slow
# Solution: Optimise tile size for throughput

labels = sp.cellpose_cellseg_tiled(
    img,
    seg_channels=[0],
    diameter=30,
    scaling=1,
    tile_size=(1024, 1024), # Larger tiles for speed
    overlap=64              # Moderate overlap
)
```

#### 4. Inconsistent Results

```python
# Problem: Results vary between runs
# Solution: Set random seeds and use consistent parameters

import random
import numpy as np

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

labels = sp.cellpose_cellseg_tiled(
    img,
    seg_channels=[0],
    diameter=30,
    scaling=1,
    tile_size=(512, 512),
    overlap=64
)
```

### Best Practices

1. **Choose appropriate tile size**: Larger tiles for better quality, smaller tiles for memory efficiency
2. **Set adequate overlap**: 10-20% of tile size for good boundary handling
3. **Preprocess noisy images manually**: Apply denoising or background subtraction before tiling
4. **Monitor memory usage**: Use system monitoring tools
5. **Test on small images first**: Validate parameters before processing large images
6. **Use consistent parameters**: For reproducible results

### Performance Tips

- Use `tile_size=(512, 512)` for most applications
- Use `overlap=64` for good boundary handling
- Apply denoising/background subtraction prior to running tiled segmentation on noisy images
- Use smaller tiles for very large images (>4K)
- Monitor memory usage during processing
- Clean up variables after processing large images
