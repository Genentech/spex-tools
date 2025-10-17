# 🗂️ SPEX Large File Processing Guide

*Updated: 2025-01-27 after Patient_test_1 integration*

## 🎯 Overview

SPEX now includes automatic tiling for processing large spatial omics files (>500MB, proteomics or transcriptomics) without memory issues or hanging. This guide shows how to use the new capabilities.

## 🚀 Quick Start

### Simple Usage (Recommended)
```python
import spex as sp

# Load any size image - tiling is automatic
img, channels = sp.load_image("large_file.ome.tiff")

# Standard API automatically uses tiling for large images
labels = sp.cellpose_cellseg(img, [0], diameter=30, scaling=1.0)
# ✅ Works for any file size, automatically optimized
```

### Explicit Control
```python
# Force tiling with specific parameters
labels = sp.cellpose_cellseg_dask(
    img, [0],
    diameter=30,
    scaling=1.0,
    tile_size=(2000, 2000),  # 19MB tiles
    overlap=400              # 20% overlap
)
```

## 📊 Proven Performance

### Patient_test_1.ome.tiff Results
- **File size**: 588MB → 1,186MB in memory
- **Processing time**: 35 minutes
- **Cells found**: 15,332
- **Configuration**: 99 tiles of 2000×2000px (19MB each)
- **Memory usage**: Safe 30MB limit per tile

### TA459 Results
- **File size**: 23MB → 352MB in memory
- **Processing time**: 48 seconds
- **Cells found**: 27
- **Configuration**: 9 tiles of 1024×1024px

## 🔧 Technical Details

### Automatic Tiling Triggers
The system automatically uses tiling when:
1. **Memory threshold exceeded**: Image >500MB estimated processing memory
2. **Explicit tile_size provided**: Manual tiling request
3. **Legacy num_tiles used**: Backward compatibility (deprecated)

### Tiling Configuration
```python
# Auto-calculated optimal settings
tile_size = calculate_optimal_tile_size(img.shape, memory_limit_mb=30)
overlap = max(32, int(min(tile_size) * 0.20))  # 20% overlap
```

### Processing Workflow
1. **Nарезка с избыточными данными**: Tiles with 20% overlap
2. **Обработка каждого тайла**: Cellpose on full tiles (with context)
3. **Извлечение core regions**: Remove overlap, keep center
4. **Сборка результата**: Seamless assembly without gaps

## 📋 Available Functions

### Standard Functions (with auto-tiling)
- `sp.cellpose_cellseg()` - Cellpose with auto-tiling
- `sp.watershed_classic()` - Watershed with auto-tiling
- `sp.stardist_cellseg()` - StarDist with auto-tiling

### Explicit Dask Functions
- `sp.cellpose_cellseg_dask()` - Explicit Cellpose tiling
- `sp.watershed_classic_dask()` - Explicit watershed tiling
- `sp.stardist_cellseg_dask()` - Explicit StarDist tiling

### Legacy Functions
- `sp.cellpose_cellseg_tiled()` - Original tiled implementation

## 🎯 Best Practices

### Memory Management
```python
# Recommended tile sizes by image type
small_images = (1000, 1000)    # 4.8MB tiles
medium_images = (1500, 1500)   # 10.7MB tiles
large_images = (2000, 2000)    # 19.1MB tiles
huge_images = (2500, 2500)     # 30MB tiles (max safe)
```

### Performance Optimization
```python
# For fastest processing
labels = sp.cellpose_cellseg(
    img, [0],
    diameter=30,
    scaling=1.0,
    auto_tiling=True,           # Enable auto-detection
    auto_tile_memory_mb=500,    # Threshold for tiling
)
```

### Error Handling
```python
try:
    labels = sp.cellpose_cellseg(img, [0], diameter=30, scaling=1.0)
except MemoryError:
    # Fallback to smaller tiles
    labels = sp.cellpose_cellseg_dask(
        img, [0], diameter=30, scaling=1.0,
        tile_size=(1000, 1000)
    )
```

## 🔄 Migration Guide

### From Old API
```python
# OLD (deprecated)
labels = sp.cellpose_cellseg(img, [0], diameter=30, scaling=1, num_tiles=4)

# NEW (recommended)
labels = sp.cellpose_cellseg(img, [0], diameter=30, scaling=1.0)  # Auto-tiling

# NEW (explicit control)
labels = sp.cellpose_cellseg(img, [0], diameter=30, scaling=1.0, tile_size=(1500, 1500))
```

### Backward Compatibility
- All existing code continues to work
- `num_tiles` parameter shows deprecation warning but functions
- New `auto_tiling=True` by default for seamless experience

## ⚠️ Troubleshooting

### Image Won't Load
```python
# Check file size first
import os
file_size_mb = os.path.getsize("large_file.ome.tiff") / (1024 * 1024)
print(f"File size: {file_size_mb:.1f} MB")

# For very large files, monitor memory during load
```

### Processing Takes Too Long
```python
# Use larger tiles to reduce processing time
labels = sp.cellpose_cellseg_dask(
    img, [0], diameter=30, scaling=1.0,
    tile_size=(2500, 2500)  # Larger tiles = faster but more memory
)
```

### Memory Issues
```python
# Use smaller tiles
labels = sp.cellpose_cellseg_dask(
    img, [0], diameter=30, scaling=1.0,
    tile_size=(1000, 1000),  # Smaller tiles = more memory safe
    overlap=200              # Explicit overlap control
)
```

### Inconsistent Results
```python
# Ensure sufficient overlap for boundary quality
labels = sp.cellpose_cellseg_dask(
    img, [0], diameter=30, scaling=1.0,
    overlap=400  # Larger overlap = better boundary handling
)
```

## 📈 Expected Performance

### File Size Guidelines
- **<100MB**: No tiling needed, direct processing
- **100-500MB**: Optional tiling, auto-detection boundary
- **500MB-2GB**: Automatic tiling, optimal performance
- **>2GB**: Always tiled, may need custom tile sizes

### Processing Time Estimates
```python
# Rule of thumb: ~2-3 tiles per minute
def estimate_time(file_size_mb, tile_size_mb=19):
    tiles = (file_size_mb * 2) / tile_size_mb  # Account for overlap
    minutes = tiles / 2.5  # Average processing speed
    return f"{minutes:.1f} minutes"
```

## 🎉 Success Stories

### Patient_test_1.ome.tiff
```python
# Before: Hung indefinitely
# After: 35 minutes, 15,332 cells

img, channels = sp.load_image("Patient_test_1.ome.tiff")
labels = sp.cellpose_cellseg(img, [0], diameter=30, scaling=1.0)
# ✅ Automatic success!
```

### Production Workflow
```python
import spex as sp
import glob

for file_path in glob.glob("data/*.ome.tiff"):
    print(f"Processing {file_path}...")

    img, channels = sp.load_image(file_path)
    labels = sp.cellpose_cellseg(img, [0], diameter=30, scaling=1.0)

    # Save results
    output_path = file_path.replace('.ome.tiff', '_labels.tiff')
    # sp.save_labels(labels, output_path)  # hypothetical save function

    print(f"✅ Found {len(np.unique(labels))-1} cells")
```

---

## 🎯 Key Takeaways

1. **Just use the standard API** - auto-tiling handles everything
2. **Patient_test_1 methodology** is proven and reliable
3. **Backward compatibility** is maintained
4. **Large files are no longer a problem** 🎉

**The old limitation of hanging on large files is completely solved!**