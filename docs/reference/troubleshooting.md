# 🔧 Troubleshooting Guide

## Common Issues and Solutions

This guide provides solutions to the most common problems encountered when using SPEX for spatial omics analysis (proteomics and transcriptomics).

## Installation Issues

### Problem: "No module named 'spex'" Error

**Symptoms:**
```python
ImportError: No module named 'spex'
```

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

4. **Install in virtual environment:**
   ```bash
   python -m venv spex_env
   source spex_env/bin/activate  # On Windows: spex_env\Scripts\activate
   pip install spex-tools
   ```

### Problem: CUDA/GPU Installation Issues

**Symptoms:**
```python
RuntimeError: CUDA out of memory
ImportError: No module named 'torch'
```

**Solutions:**

1. **Install PyTorch with CUDA support:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Use CPU-only version:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Check GPU availability:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU count: {torch.cuda.device_count()}")
   ```

### Problem: OpenCV Installation Issues

**Symptoms:**
```python
ImportError: libGL.so.1: cannot open shared object file
```

**Solutions:**

1. **Install system dependencies (Ubuntu/Debian):**
   ```bash
   sudo apt-get update
   sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
   ```

2. **Install system dependencies (CentOS/RHEL):**
   ```bash
   sudo yum install mesa-libGL libglib2.0-0 libSM libXext libXrender
   ```

3. **Reinstall OpenCV:**
   ```bash
   pip uninstall opencv-python
   pip install opencv-python-headless
   ```

## Data Loading Issues

### Problem: Memory Errors When Loading Large Images

**Symptoms:**
```python
MemoryError: Unable to allocate array
```

**Solutions:**

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

4. **Monitor memory usage:**
   ```python
   import psutil
   import os
   
   def get_memory_usage():
       process = psutil.Process(os.getpid())
       return process.memory_info().rss / 1024 / 1024  # MB
   
   print(f"Memory usage: {get_memory_usage():.1f} MB")
   ```

### Problem: Unsupported Image Format

**Symptoms:**
```python
ValueError: Unsupported image format
```

**Solutions:**

1. **Convert image format:**
   ```python
   from PIL import Image
   img = Image.open('unsupported_format.bmp')
   img.save('converted_image.tiff')
   ```

2. **Check supported formats:**
   ```python
   from PIL import Image
   print(Image.OPEN)
   ```

3. **Use alternative loading:**
   ```python
   import tifffile
   Image = tifffile.imread('image.tiff')
   ```

### Problem: Incorrect Channel Information

**Symptoms:**
```python
IndexError: Channel index out of bounds
```

**Solutions:**

1. **Check image dimensions:**
   ```python
   print(f"Image shape: {Image.shape}")
   print(f"Number of channels: {Image.shape[2] if len(Image.shape) == 3 else 1}")
   ```

2. **Verify channel indices:**
   ```python
   # Use valid channel indices
   valid_channels = list(range(Image.shape[2]))
   features = sp.feature_extraction(Image, labels, valid_channels)
   ```

3. **Handle single-channel images:**
   ```python
   if len(Image.shape) == 2:
       Image = Image[:, :, np.newaxis]
   ```

## Image Preprocessing Issues

### Problem: Background Subtraction Not Working

**Symptoms:**
- No visible change after background subtraction
- Images become too dark or bright

**Solutions:**

1. **Adjust window size:**
   ```python
   # Increase window size for larger background variations
   Image_bg = sp.background_subtract(Image, [0], window_size=100)
   
   # Decrease for fine detail preservation
   Image_bg = sp.background_subtract(Image, [0], window_size=20)
   ```

2. **Check image statistics:**
   ```python
   print(f"Original range: {Image.min():.2f} - {Image.max():.2f}")
   print(f"Background subtracted range: {Image_bg.min():.2f} - {Image_bg.max():.2f}")
   ```

3. **Visualize results:**
   ```python
   import matplotlib.pyplot as plt
   
   fig, axes = plt.subplots(1, 2, figsize=(10, 5))
   axes[0].imshow(Image[:, :, 0], cmap='gray')
   axes[0].set_title('Original')
   axes[1].imshow(Image_bg[:, :, 0], cmap='gray')
   axes[1].set_title('Background Subtracted')
   plt.show()
   ```

### Problem: Denoising Too Aggressive

**Symptoms:**
- Loss of important features
- Over-smoothed images

**Solutions:**

1. **Reduce denoising strength:**
   ```python
   # Use smaller h parameter for less aggressive denoising
   Image_denoised = sp.nlm_denoise(Image, [0], h=0.05)
   ```

2. **Try different denoising methods:**
   ```python
   # Use median filter for salt-and-pepper noise
   Image_denoised = sp.median_denoise(Image, [0])
   ```

3. **Apply denoising selectively:**
   ```python
   # Denoise only noisy channels
   Image_denoised = Image.copy()
   noisy_channels = [0, 2]  # Only denoise channels 0 and 2
   for ch in noisy_channels:
       Image_denoised[:, :, ch] = sp.nlm_denoise(Image[:, :, ch], [0])
   ```

## Segmentation Issues

### Problem: No Cells Detected

**Symptoms:**
```python
print(f"Cells detected: {labels.max()}")  # Output: 0
```

**Solutions:**

1. **Check image quality:**
   ```python
   # Verify image has sufficient contrast
   print(f"Image range: {Image.min():.2f} - {Image.max():.2f}")
   print(f"Image mean: {Image.mean():.2f}")
   print(f"Image std: {Image.std():.2f}")
   ```

2. **Improve preprocessing:**
   ```python
   # Apply better preprocessing
   Image_bg = sp.background_subtract(Image, [0], window_size=50)
   Image_denoised = sp.nlm_denoise(Image_bg, [0])
   ```

3. **Adjust segmentation parameters:**
   ```python
   # Try smaller diameter
   labels = sp.cellpose_cellseg(Image, [0], diameter=15)
   
   # Try different flow threshold
   labels = sp.cellpose_cellseg(Image, [0], flow_threshold=0.4)
   ```

4. **Use alternative segmentation method:**
   ```python
   # Try watershed segmentation
   labels = sp.watershed_classic(Image, [0])
   ```

### Problem: Too Many False Positive Cells

**Symptoms:**
- Many small objects detected
- Noise identified as cells

**Solutions:**

1. **Remove small objects:**
   ```python
   # Increase minimum size threshold
   labels_clean = sp.remove_small_objects(labels, min_size=100)
   ```

2. **Remove large objects:**
   ```python
   # Remove objects that are too large
   labels_clean = sp.remove_large_objects(labels, max_size=500)
   ```

3. **Adjust segmentation parameters:**
   ```python
   # Increase diameter to avoid detecting small noise
   labels = sp.cellpose_cellseg(Image, [0], diameter=40)
   ```

4. **Improve image quality:**
   ```python
   # Apply stronger denoising
   Image_denoised = sp.nlm_denoise(Image, [0], h=0.1)
   Image_denoised = sp.median_denoise(Image_denoised, [0])
   ```

### Problem: Missing Cells in Segmentation

**Symptoms:**
- Expected cells not detected
- Incomplete segmentation

**Solutions:**

1. **Use cell rescue:**
   ```python
   # Rescue missing cells
   labels_rescued = sp.rescue_cells(Image, labels, [0])
   ```

2. **Adjust diameter parameter:**
   ```python
   # Try smaller diameter for smaller cells
   labels = sp.cellpose_cellseg(Image, [0], diameter=20)
   ```

3. **Improve contrast:**
   ```python
   # Enhance contrast before segmentation
   from skimage import exposure
   Image_enhanced = exposure.equalize_hist(Image[:, :, 0])
   Image_enhanced = np.stack([Image_enhanced] * Image.shape[2], axis=2)
   ```

4. **Try different segmentation method:**
   ```python
   # Try StarDist for different cell shapes
   labels = sp.stardist_cellseg(Image, [0])
   ```

### Problem: Segmentation Boundaries Not Accurate

**Symptoms:**
- Cell boundaries don't follow actual cell edges
- Over-segmentation or under-segmentation

**Solutions:**

1. **Validate boundaries:**
   ```python
   def validate_boundaries(image, labels):
       from scipy import ndimage
       
       # Calculate image gradients
       grad_x = ndimage.sobel(image[:, :, 0], axis=0)
       grad_y = ndimage.sobel(image[:, :, 0], axis=1)
       gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
       
       # Find cell boundaries
       boundaries = labels == 0
       boundary_gradients = gradient_magnitude[boundaries]
       
       quality_score = np.mean(boundary_gradients) / np.max(gradient_magnitude)
       return quality_score
   
   quality = validate_boundaries(Image, labels)
   print(f"Boundary quality: {quality:.3f}")
   ```

2. **Adjust flow threshold:**
   ```python
   # Lower threshold for more precise boundaries
   labels = sp.cellpose_cellseg(Image, [0], flow_threshold=0.2)
   ```

3. **Use post-processing:**
   ```python
   # Apply morphological operations
   from scipy import ndimage
   labels_processed = ndimage.binary_fill_holes(labels > 0)
   labels_processed = ndimage.label(labels_processed)[0]
   ```

## Feature Extraction Issues

### Problem: Feature Extraction Fails

**Symptoms:**
```python
ValueError: No valid cells found for feature extraction
```

**Solutions:**

1. **Check segmentation:**
   ```python
   print(f"Number of cells: {labels.max()}")
   print(f"Unique labels: {np.unique(labels)}")
   ```

2. **Remove background label:**
   ```python
   # Ensure labels start from 1
   if labels.min() == 0:
       labels = labels + 1
   ```

3. **Check channel indices:**
   ```python
   # Ensure channel indices are valid
   valid_channels = list(range(min(len(channel), Image.shape[2])))
   features = sp.feature_extraction(Image, labels, valid_channels)
   ```

### Problem: Missing Values in Features

**Symptoms:**
```python
print(f"Missing values: {features.isnull().sum().sum()}")
```

**Solutions:**

1. **Handle missing values:**
   ```python
   # Remove cells with missing values
   features_clean = features.dropna()
   
   # Fill missing values
   features_filled = features.fillna(features.mean())
   
   # Interpolate missing values
   features_interpolated = features.interpolate(method='linear')
   ```

2. **Check for zero-variance features:**
   ```python
   # Remove features with zero variance
   feature_variance = features.var()
   good_features = feature_variance[feature_variance > 0].index
   features_filtered = features[good_features]
   ```

3. **Investigate specific features:**
   ```python
   # Check which features have missing values
   missing_features = features.columns[features.isnull().any()].tolist()
   print(f"Features with missing values: {missing_features}")
   ```

### Problem: Too Many/Low-Quality Features

**Symptoms:**
- Too many features extracted
- Features with low information content

**Solutions:**

1. **Select informative features:**
   ```python
   # Select features with high variance
   feature_variance = features.var()
   informative_features = feature_variance[feature_variance > 0.01].index
   features_selected = features[informative_features]
   ```

2. **Remove correlated features:**
   ```python
   # Remove highly correlated features
   corr_matrix = features.corr()
   high_corr_pairs = []
   
   for i in range(len(corr_matrix.columns)):
       for j in range(i+1, len(corr_matrix.columns)):
           if abs(corr_matrix.iloc[i, j]) > 0.95:
               high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
   
   # Remove one feature from each highly correlated pair
   features_to_remove = [pair[1] for pair in high_corr_pairs]
   features_uncorr = features.drop(columns=features_to_remove)
   ```

3. **Use feature selection methods:**
   ```python
   from sklearn.feature_selection import SelectKBest, f_classif
   
   # Select top k features
   selector = SelectKBest(score_func=f_classif, k=50)
   features_selected = selector.fit_transform(features, clusters)
   ```

## Clustering Issues

### Problem: Only One Cluster Produced

**Symptoms:**
```python
print(f"Number of clusters: {len(np.unique(clusters))}")  # Output: 1
```

**Solutions:**

1. **Check feature quality:**
   ```python
   # Check feature variance
   feature_variance = features.var()
   print(f"Features with zero variance: {(feature_variance == 0).sum()}")
   
   # Remove low-variance features
   good_features = feature_variance[feature_variance > 0.01].index
   features_filtered = features[good_features]
   ```

2. **Increase resolution:**
   ```python
   # Try higher resolution
   clusters = sp.cluster(adata, method='leiden', resolution=1.0)
   ```

3. **Try different clustering method:**
   ```python
   # Use Phenograph
   clusters = sp.phenograph_cluster(features, k=30)
   ```

4. **Normalize features:**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   features_normalized = scaler.fit_transform(features)
   ```

### Problem: Too Many Clusters

**Symptoms:**
```python
print(f"Number of clusters: {len(np.unique(clusters))}")  # Too many
```

**Solutions:**

1. **Decrease resolution:**
   ```python
   # Try lower resolution
   clusters = sp.cluster(adata, method='leiden', resolution=0.1)
   ```

2. **Reduce feature dimensionality:**
   ```python
   from sklearn.decomposition import PCA
   pca = PCA(n_components=20)
   features_pca = pca.fit_transform(features)
   ```

3. **Use different clustering method:**
   ```python
   # Use Louvain with lower resolution
   clusters = sp.cluster(adata, method='louvain', resolution=0.3)
   ```

### Problem: Clusters Don't Make Biological Sense

**Symptoms:**
- Clusters don't correspond to known cell types
- Poor cluster separation

**Solutions:**

1. **Validate clustering quality:**
   ```python
   from sklearn.metrics import silhouette_score
   
   silhouette_avg = silhouette_score(features, clusters)
   print(f"Silhouette score: {silhouette_avg:.3f}")
   
   # Good: > 0.3, Acceptable: > 0.1, Poor: < 0.1
   ```

2. **Check feature selection:**
   ```python
   # Use domain-specific features
   biological_features = ['area', 'perimeter', 'mean_intensity', 'eccentricity']
   features_bio = features[biological_features]
   ```

3. **Try different preprocessing:**
   ```python
   # Log transform features
   features_log = np.log1p(features)
   
   # Z-score normalization
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   features_scaled = scaler.fit_transform(features)
   ```

## Spatial Analysis Issues

### Problem: CLQ Calculation Fails

**Symptoms:**
```python
ValueError: Invalid cell types specified
```

**Solutions:**

1. **Check cell type labels:**
   ```python
   # Ensure cell types exist in cluster labels
   unique_clusters = np.unique(clusters)
   print(f"Available cell types: {unique_clusters}")
   
   # Use valid cell types
   valid_cell_types = [0, 1, 2]  # Must exist in clusters
   clq_matrix = sp.CLQ_vec_numba(labels, clusters, cell_types=valid_cell_types)
   ```

2. **Check radius parameter:**
   ```python
   # Use appropriate radius
   clq_matrix = sp.CLQ_vec_numba(labels, clusters, cell_types=[0, 1, 2], radius=50)
   ```

3. **Ensure sufficient cells per type:**
   ```python
   # Check cell counts per type
   for cell_type in [0, 1, 2]:
       count = np.sum(clusters == cell_type)
       print(f"Cell type {cell_type}: {count} cells")
   ```

### Problem: Spatial Analysis Results Don't Make Sense

**Symptoms:**
- Unexpected spatial patterns
- Counterintuitive results

**Solutions:**

1. **Validate spatial coordinates:**
   ```python
   from skimage import measure
   
   # Get cell centroids
   props = measure.regionprops_table(labels, properties=['centroid'])
   centroids = np.column_stack([props['centroid-0'], props['centroid-1']])
   
   # Check coordinate ranges
   print(f"X range: {centroids[:, 0].min():.1f} - {centroids[:, 0].max():.1f}")
   print(f"Y range: {centroids[:, 1].min():.1f} - {centroids[:, 1].max():.1f}")
   ```

2. **Visualize spatial distribution:**
   ```python
   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(10, 8))
   for cell_type in np.unique(clusters):
       mask = clusters == cell_type
       plt.scatter(centroids[mask, 0], centroids[mask, 1], 
                  label=f'Type {cell_type}', alpha=0.7)
   plt.legend()
   plt.title('Spatial Distribution of Cell Types')
   plt.show()
   ```

3. **Check for edge effects:**
   ```python
   # Remove cells near image edges
   edge_distance = 50
   valid_cells = (centroids[:, 0] > edge_distance) & (centroids[:, 0] < Image.shape[1] - edge_distance) & \
                 (centroids[:, 1] > edge_distance) & (centroids[:, 1] < Image.shape[0] - edge_distance)
   
   labels_center = labels.copy()
   labels_center[~valid_cells] = 0
   ```

## Performance Issues

### Problem: Analysis Takes Too Long

**Symptoms:**
- Processing time is excessive
- Memory usage is high

**Solutions:**

1. **Use parallel processing:**
   ```python
   # Set number of jobs
   clusters = sp.cluster(adata, method='leiden', n_jobs=4)
   ```

2. **Reduce image size:**
   ```python
   from skimage.transform import resize
   Image_small = resize(Image, (Image.shape[0]//2, Image.shape[1]//2))
   ```

3. **Use chunked processing:**
   ```python
   # Process large images in chunks
   labels = sp.cellpose_cellseg(Image, [0], chunk_size=512)
   ```

4. **Optimize memory usage:**
   ```python
   # Use memory-efficient data types
   Image = Image.astype(np.float32)
   
   # Clear memory after processing
   import gc
   gc.collect()
   ```

### Problem: CUDA Out of Memory

**Symptoms:**
```python
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
   ```python
   labels = sp.cellpose_cellseg(Image, [0], batch_size=1)
   ```

2. **Use CPU instead of GPU:**
   ```python
   labels = sp.cellpose_cellseg(Image, [0], use_gpu=False)
   ```

3. **Process smaller chunks:**
   ```python
   labels = sp.cellpose_cellseg(Image, [0], chunk_size=256)
   ```

4. **Clear GPU memory:**
   ```python
   import torch
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
   ```

## Data Export Issues

### Problem: Results Can't Be Saved

**Symptoms:**
```python
PermissionError: [Errno 13] Permission denied
```

**Solutions:**

1. **Check file permissions:**
   ```python
   import os
   print(f"Current directory: {os.getcwd()}")
   print(f"Write permission: {os.access('.', os.W_OK)}")
   ```

2. **Use absolute paths:**
   ```python
   import os
   output_dir = '/path/to/output/directory'
   os.makedirs(output_dir, exist_ok=True)
   
   # Save results
   features.to_csv(os.path.join(output_dir, 'features.csv'))
   ```

3. **Handle file formats:**
   ```python
   # Save in different formats
   features.to_csv('features.csv')
   features.to_excel('features.xlsx')
   features.to_hdf('features.h5', key='features')
   ```

### Problem: Large Files Can't Be Saved

**Symptoms:**
```python
MemoryError: Unable to save large file
```

**Solutions:**

1. **Save in chunks:**
   ```python
   # Save large DataFrames in chunks
   chunk_size = 1000
   for i in range(0, len(features), chunk_size):
       chunk = features.iloc[i:i+chunk_size]
       chunk.to_csv(f'features_chunk_{i//chunk_size}.csv')
   ```

2. **Use efficient formats:**
   ```python
   # Use HDF5 for large files
   features.to_hdf('features.h5', key='features', mode='w', complevel=9)
   
   # Use Parquet for large files
   features.to_parquet('features.parquet')
   ```

3. **Compress files:**
   ```python
   # Save with compression
   features.to_csv('features.csv.gz', compression='gzip')
   ```

## Getting Help

### When to Contact Support

Contact the SPEX development team if you encounter:

1. **Bugs**: Unexpected behavior or crashes
2. **Performance issues**: Unreasonably slow processing
3. **Missing features**: Functionality you need but isn't available
4. **Documentation issues**: Unclear or incorrect documentation

### How to Report Issues

When reporting issues, include:

1. **Clear description** of the problem
2. **Minimal reproducible example** with code
3. **System information**:
   - Operating system
   - Python version
   - SPEX version
   - Hardware specifications
4. **Error messages** and stack traces
5. **Expected vs actual behavior**

### Example Issue Report

```
Title: Segmentation fails with memory error on large images

Description:
When processing images larger than 4000x4000 pixels, segmentation fails with a memory error.

Steps to reproduce:
1. Load large image: Image, channel = sp.load_image('large_image.tiff')
2. Run segmentation: labels = sp.cellpose_cellseg(Image, [0])
3. Error occurs during processing

System information:
- OS: Ubuntu 20.04
- Python: 3.9.7
- SPEX: 0.3.1055
- RAM: 16GB
- GPU: NVIDIA RTX 3080

Error message:
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB

Expected behavior:
Segmentation should complete successfully or provide a clear error message about memory requirements.
```

---

**Still having issues?** Check the FAQ section or contact the SPEX development team for additional support.
