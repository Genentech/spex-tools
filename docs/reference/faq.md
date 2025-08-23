# ❓ Frequently Asked Questions (FAQ)

## General Questions

### What is SPEX?

SPEX is a comprehensive Python library for spatial transcriptomics analysis. It provides tools for image processing, cell segmentation, feature extraction, clustering, and spatial analysis of multi-channel fluorescence microscopy data.

### What types of data does SPEX support?

SPEX supports:
- Multi-channel TIFF images
- Fluorescence microscopy data
- Spatial transcriptomics datasets
- Single-cell imaging data
- Any multi-dimensional image data with spatial information

### Is SPEX free to use?

Yes, SPEX is open-source and free to use under the Apache License 2.0.

### What are the system requirements?

**Minimum Requirements:**
- Python 3.8+
- 8GB RAM
- 4 CPU cores

**Recommended Requirements:**
- Python 3.9+
- 16GB+ RAM
- 8+ CPU cores
- GPU support (optional, for acceleration)

## Installation

### How do I install SPEX?

```bash
pip install spex-tools
```

### I'm getting installation errors. What should I do?

**Common Solutions:**

1. **Update pip and setuptools:**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **Install in a virtual environment:**
   ```bash
   python -m venv spex_env
   source spex_env/bin/activate  # On Windows: spex_env\Scripts\activate
   pip install spex-tools
   ```

3. **Install system dependencies (Ubuntu/Debian):**
   ```bash
   sudo apt-get update
   sudo apt-get install libgl1-mesa-glx libglib2.0-0
   ```

4. **For M1/M2 Macs:**
   ```bash
   conda install -c conda-forge spex-tools
   ```

### How do I install optional dependencies?

```bash
# For GPU support
pip install spex-tools[gpu]

# For development
pip install spex-tools[dev]

# For all optional dependencies
pip install spex-tools[all]
```

## Data Loading

### What image formats are supported?

SPEX supports:
- TIFF/TIF (recommended)
- PNG
- JPEG/JPG
- BMP
- Most formats supported by PIL/Pillow

### How do I load my image data?

```python
import spex as sp

# Load single image
Image, channel = sp.load_image('path/to/image.tiff')

# Load multiple images
images = []
for file in image_files:
    img, ch = sp.load_image(file)
    images.append(img)
```

### My image has multiple channels. How do I handle them?

```python
# Load multi-channel image
Image, channel = sp.load_image('multichannel.tiff')

print(f"Image shape: {Image.shape}")  # (height, width, channels)
print(f"Channels: {channel}")

# Access specific channel
channel_0 = Image[:, :, 0]
```

### I'm getting memory errors when loading large images. What can I do?

1. **Use chunked loading:**
   ```python
   # Load in chunks
   Image, channel = sp.load_image('large_image.tiff', chunk_size=1000)
   ```

2. **Reduce image resolution:**
   ```python
   from PIL import Image
   img = Image.open('large_image.tiff')
   img_resized = img.resize((img.width//2, img.height//2))
   img_resized.save('resized_image.tiff')
   ```

3. **Use memory-efficient data types:**
   ```python
   import numpy as np
   Image = Image.astype(np.float32)  # Instead of float64
   ```

## Image Preprocessing

### What preprocessing steps should I apply?

**Recommended pipeline:**
```python
# 1. Background subtraction
Image_bg = sp.background_subtract(Image, list(range(len(channel))))

# 2. Denoising
Image_denoised = sp.nlm_denoise(Image_bg, list(range(len(channel))))

# 3. Optional: Additional filtering
Image_filtered = sp.median_denoise(Image_denoised, list(range(len(channel))))
```

### How do I choose the right preprocessing parameters?

**Background Subtraction:**
- Use default parameters for most cases
- Increase `window_size` for larger background variations
- Decrease for fine detail preservation

**Denoising:**
- `nlm_denoise`: Better for preserving edges, slower
- `median_denoise`: Faster, good for salt-and-pepper noise

### My images are too noisy. What should I do?

1. **Increase denoising strength:**
   ```python
   Image_denoised = sp.nlm_denoise(Image, list(range(len(channel))), h=0.1)
   ```

2. **Apply multiple denoising steps:**
   ```python
   Image_denoised1 = sp.nlm_denoise(Image, list(range(len(channel))))
   Image_denoised2 = sp.median_denoise(Image_denoised1, list(range(len(channel))))
   ```

3. **Check image acquisition settings:**
   - Increase exposure time
   - Use higher quality cameras
   - Optimize illumination

## Cell Segmentation

### Which segmentation method should I use?

**Cellpose (Recommended):**
- Best for most cell types
- Automatic parameter estimation
- Good for irregular shapes

**StarDist:**
- Good for star-shaped objects
- Faster than Cellpose
- Requires more parameter tuning

**Watershed:**
- Good for round, regular cells
- Fastest method
- Requires good preprocessing

### How do I set the cell diameter parameter?

```python
# Automatic estimation (recommended)
labels = sp.cellpose_cellseg(Image, [0])

# Manual setting
labels = sp.cellpose_cellseg(Image, [0], diameter=30)

# Estimate from image
from skimage import measure
props = measure.regionprops(labels)
diameters = [prop.equivalent_diameter for prop in props]
print(f"Average cell diameter: {np.mean(diameters):.1f} pixels")
```

### My segmentation is missing cells. What can I do?

1. **Adjust diameter parameter:**
   ```python
   # Try smaller diameter
   labels = sp.cellpose_cellseg(Image, [0], diameter=20)
   ```

2. **Use cell rescue:**
   ```python
   labels_rescued = sp.rescue_cells(Image, labels, [0])
   ```

3. **Improve preprocessing:**
   ```python
   # Better contrast enhancement
   Image_enhanced = sp.background_subtract(Image, [0], window_size=50)
   ```

### How do I remove false positive cells?

```python
# Remove small objects
labels_clean = sp.remove_small_objects(labels, min_size=50)

# Remove large objects
labels_clean = sp.remove_large_objects(labels_clean, max_size=1000)

# Combine both
labels_clean = sp.remove_small_objects(labels, min_size=50)
labels_clean = sp.remove_large_objects(labels_clean, max_size=1000)
```

## Feature Extraction

### What features are extracted?

SPEX extracts:
- **Morphological features**: Area, perimeter, eccentricity, etc.
- **Intensity features**: Mean, std, min, max, etc.
- **Texture features**: Haralick features, GLCM features
- **Spatial features**: Centroid, bounding box, etc.

### How do I extract features for specific channels?

```python
# Extract features for all channels
features = sp.feature_extraction(Image, labels, list(range(len(channel))))

# Extract features for specific channels
features = sp.feature_extraction(Image, labels, [0, 2])  # Channels 0 and 2

# Extract features for single channel
features = sp.feature_extraction(Image, labels, [0])
```

### How do I handle missing values in features?

```python
# Remove cells with missing values
features_clean = features.dropna()

# Fill missing values
features_filled = features.fillna(features.mean())

# Interpolate missing values
features_interpolated = features.interpolate(method='linear')
```

### Can I extract custom features?

```python
# Extract basic features first
features = sp.feature_extraction(Image, labels, [0])

# Add custom features
features['custom_ratio'] = features['area'] / features['perimeter']
features['intensity_density'] = features['mean_intensity'] / features['area']
```

## Clustering

### Which clustering method should I use?

**Leiden (Recommended):**
- Best for most datasets
- Handles large datasets well
- Good cluster quality

**Louvain:**
- Similar to Leiden
- Slightly faster
- May produce fewer clusters

**Phenograph:**
- Good for complex datasets
- Automatic parameter estimation
- May be slower

### How do I choose the resolution parameter?

```python
# Try multiple resolutions
resolutions = [0.1, 0.3, 0.5, 0.7, 1.0]
results = {}

for res in resolutions:
    clusters = sp.cluster(adata, method='leiden', resolution=res)
    n_clusters = len(np.unique(clusters))
    results[res] = n_clusters
    print(f"Resolution {res}: {n_clusters} clusters")

# Choose based on expected number of cell types
```

### How do I validate clustering results?

```python
from sklearn.metrics import silhouette_score

# Calculate silhouette score
silhouette_avg = silhouette_score(features, clusters)
print(f"Silhouette score: {silhouette_avg:.3f}")

# Visualize clusters
import matplotlib.pyplot as plt
plt.scatter(features['umap_1'], features['umap_2'], c=clusters, cmap='tab10')
plt.colorbar()
plt.show()
```

### My clusters don't make biological sense. What should I do?

1. **Check feature quality:**
   ```python
   # Remove low-quality features
   feature_variance = features.var()
   good_features = feature_variance[feature_variance > 0.01].index
   features_filtered = features[good_features]
   ```

2. **Try different preprocessing:**
   ```python
   # Normalize features
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   features_normalized = scaler.fit_transform(features)
   ```

3. **Use different clustering method:**
   ```python
   clusters = sp.phenograph_cluster(features, k=30)
   ```

## Spatial Analysis

### What spatial analysis methods are available?

SPEX provides:
- **CLQ (Co-Localization Quotient)**: Measures spatial co-occurrence
- **Niche Analysis**: Identifies spatial niches
- **Spatial Autocorrelation**: Moran's I, Geary's C
- **Differential Expression**: Spatial-aware DE analysis

### How do I calculate spatial relationships between cell types?

```python
# Calculate CLQ between cell types
clq_matrix = sp.CLQ_vec_numba(labels, cluster_labels, 
                             cell_types=[0, 1, 2], 
                             radius=50)

# Visualize CLQ matrix
import seaborn as sns
sns.heatmap(clq_matrix, annot=True, cmap='RdBu_r', center=0)
plt.show()
```

### How do I identify spatial niches?

```python
# Perform niche analysis
niche_results = sp.niche(labels, cluster_labels, 
                        radius=100, 
                        min_cells=10)

# Visualize niches
niche_results.plot_niches()
```

### How do I interpret spatial autocorrelation results?

```python
# Calculate Moran's I
moran_i = sp.spatial_autocorrelation(features, labels, method='moran')

# Interpretation:
# Moran's I > 0: Positive spatial autocorrelation (clustering)
# Moran's I < 0: Negative spatial autocorrelation (dispersion)
# Moran's I ≈ 0: Random spatial distribution
```

## Performance and Optimization

### My analysis is running slowly. How can I speed it up?

1. **Use parallel processing:**
   ```python
   # Set number of jobs
   clusters = sp.cluster(adata, method='leiden', n_jobs=4)
   ```

2. **Reduce image size:**
   ```python
   # Resize image before processing
   from skimage.transform import resize
   Image_small = resize(Image, (Image.shape[0]//2, Image.shape[1]//2))
   ```

3. **Use chunked processing:**
   ```python
   # Process large images in chunks
   labels = sp.cellpose_cellseg(Image, [0], chunk_size=512)
   ```

### How do I handle large datasets that don't fit in memory?

1. **Use Dask for out-of-memory processing:**
   ```python
   import dask.array as da
   Image_dask = da.from_array(Image, chunks=(1000, 1000, -1))
   ```

2. **Process in batches:**
   ```python
   # Process multiple files in batches
   for batch in file_batches:
       process_batch(batch)
       gc.collect()  # Clear memory
   ```

3. **Use memory-efficient data types:**
   ```python
   Image = Image.astype(np.float32)  # Instead of float64
   ```

### How do I monitor memory usage?

```python
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory usage: {get_memory_usage():.1f} MB")
```

## Troubleshooting

### I'm getting "CUDA out of memory" errors

**Solutions:**
1. **Reduce batch size:**
   ```python
   labels = sp.cellpose_cellseg(Image, [0], batch_size=1)
   ```

2. **Use CPU instead of GPU:**
   ```python
   labels = sp.cellpose_cellseg(Image, [0], use_gpu=False)
   ```

3. **Process smaller image chunks:**
   ```python
   labels = sp.cellpose_cellseg(Image, [0], chunk_size=256)
   ```

### My segmentation is producing too many/few cells

**Too many cells:**
```python
# Increase minimum cell size
labels = sp.remove_small_objects(labels, min_size=100)

# Increase diameter parameter
labels = sp.cellpose_cellseg(Image, [0], diameter=40)
```

**Too few cells:**
```python
# Decrease diameter parameter
labels = sp.cellpose_cellseg(Image, [0], diameter=20)

# Use cell rescue
labels = sp.rescue_cells(Image, labels, [0])
```

### I'm getting "No module named 'spex'" error

**Solutions:**
1. **Check installation:**
   ```bash
   pip list | grep spex
   ```

2. **Reinstall SPEX:**
   ```bash
   pip uninstall spex-tools
   pip install spex-tools
   ```

3. **Check Python environment:**
   ```bash
   which python
   pip --version
   ```

### My clustering is producing only one cluster

**Solutions:**
1. **Check feature variance:**
   ```python
   feature_variance = features.var()
   print(f"Features with zero variance: {(feature_variance == 0).sum()}")
   ```

2. **Increase resolution:**
   ```python
   clusters = sp.cluster(adata, method='leiden', resolution=1.0)
   ```

3. **Try different clustering method:**
   ```python
   clusters = sp.phenograph_cluster(features, k=30)
   ```

### How do I save and load my analysis results?

**Save results:**
```python
import pickle

# Save segmentation
with open('segmentation.pkl', 'wb') as f:
    pickle.dump(labels, f)

# Save features
features.to_csv('features.csv')

# Save clustering
with open('clustering.pkl', 'wb') as f:
    pickle.dump(clusters, f)
```

**Load results:**
```python
# Load segmentation
with open('segmentation.pkl', 'rb') as f:
    labels = pickle.load(f)

# Load features
features = pd.read_csv('features.csv', index_col=0)

# Load clustering
with open('clustering.pkl', 'rb') as f:
    clusters = pickle.load(f)
```

## Best Practices

### What is the recommended workflow?

1. **Data Loading and Preprocessing:**
   ```python
   Image, channel = sp.load_image('data.tiff')
   Image_bg = sp.background_subtract(Image, list(range(len(channel))))
   Image_denoised = sp.nlm_denoise(Image_bg, list(range(len(channel))))
   ```

2. **Cell Segmentation:**
   ```python
   labels = sp.cellpose_cellseg(Image_denoised, [0])
   labels_clean = sp.remove_small_objects(labels, min_size=50)
   labels_clean = sp.remove_large_objects(labels_clean, max_size=1000)
   ```

3. **Feature Extraction:**
   ```python
   features = sp.feature_extraction(Image_denoised, labels_clean, list(range(len(channel))))
   ```

4. **Clustering:**
   ```python
   adata = sp.feature_extraction_adata(Image_denoised, labels_clean, list(range(len(channel))))
   clusters = sp.cluster(adata, method='leiden', resolution=0.5)
   ```

5. **Spatial Analysis:**
   ```python
   clq_matrix = sp.CLQ_vec_numba(labels_clean, clusters, cell_types=[0, 1, 2])
   ```

### How do I ensure reproducible results?

```python
import numpy as np
import random

# Set random seeds
np.random.seed(42)
random.seed(42)

# Use deterministic algorithms
clusters = sp.cluster(adata, method='leiden', resolution=0.5, random_state=42)
```

### How do I validate my analysis pipeline?

1. **Use known datasets:**
   - Test with published datasets
   - Compare results with known ground truth

2. **Cross-validation:**
   ```python
   # Split data and compare results
   from sklearn.model_selection import train_test_split
   features_train, features_test = train_test_split(features, test_size=0.3)
   ```

3. **Parameter sensitivity analysis:**
   ```python
   # Test different parameters
   resolutions = [0.1, 0.3, 0.5, 0.7, 1.0]
   for res in resolutions:
       clusters = sp.cluster(adata, method='leiden', resolution=res)
       # Evaluate clustering quality
   ```

## Getting Help

### Where can I find more documentation?

- **Official Documentation**: [SPEX Documentation](https://spex-docs.readthedocs.io/)
- **API Reference**: Complete function documentation
- **Tutorials**: Step-by-step guides
- **Examples**: Practical code examples

### How do I report bugs or request features?

1. **GitHub Issues**: Create an issue on the SPEX GitHub repository
2. **Email**: Contact the development team
3. **Documentation**: Check existing issues and documentation

### How do I contribute to SPEX?

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests**
5. **Submit a pull request**

### Where can I find example datasets?

- **SPEX Examples**: Included with the library
- **Public Repositories**: 
  - 10X Genomics datasets
  - Human Cell Atlas
  - Allen Brain Atlas
- **Synthetic Data**: Generate using SPEX simulation functions

---

**Need more help?** Check the troubleshooting section or contact the SPEX development team.
