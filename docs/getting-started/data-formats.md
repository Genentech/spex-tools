# 📊 Data Formats Guide

## Input Data Formats

### Image Files

SPEX supports various image formats for input:

#### Multi-channel TIFF (Recommended)
```python
# Load multi-channel TIFF
Image, channel = sp.load_image('multichannel_image.tiff')
# Image shape: (height, width, channels)
# channel: list of channel names
```

#### Single-channel Images
```python
# PNG, JPEG, or single-channel TIFF
Image, channel = sp.load_image('single_channel.png')
# Image shape: (height, width, 1)
```

#### Supported Formats
- **TIFF**: Multi-channel, lossless compression
- **PNG**: Single-channel, lossless compression
- **JPEG**: Single-channel, lossy compression
- **BMP**: Single-channel, uncompressed

### Image Data Structure

```python
# Typical image structure
Image.shape  # (height, width, channels)
Image.dtype  # uint8, uint16, or float32

# Example
print(f"Image dimensions: {Image.shape}")
print(f"Data type: {Image.dtype}")
print(f"Value range: {Image.min()} - {Image.max()}")
```

## Output Data Formats

### Segmentation Labels

```python
# Watershed segmentation output
labels = sp.watershed_classic(Image, [0])
# labels.shape: (height, width)
# labels.dtype: int32
# Each unique value represents a cell/object
```

### Feature Extraction Output

#### AnnData Object (Recommended)
```python
# Extract features into AnnData format
adata = sp.feature_extraction_adata(Image, labels, channel)

# AnnData structure
print(f"Cells: {adata.n_obs}")
print(f"Features: {adata.n_vars}")
print(f"Channels: {list(adata.var_names)}")

# Access data
adata.X  # Expression matrix (cells x features)
adata.obs  # Cell metadata
adata.var  # Feature metadata
adata.obsm['spatial']  # Spatial coordinates
```

#### Dictionary Format
```python
# Alternative dictionary output
features_dict = sp.feature_extraction(Image, labels, channel)

# Structure
features_dict = {
    'expression': np.array,  # (cells, features)
    'coordinates': np.array,  # (cells, 2) - x, y coordinates
    'areas': np.array,       # (cells,) - cell areas
    'channels': list         # channel names
}
```

## Data Processing Pipeline

### Preprocessing

```python
# Background subtraction
Image_processed = sp.background_subtract(Image, [0])

# Denoising
Image_denoised = sp.median_denoise(Image_processed, [0])
Image_nlm = sp.nlm_denoise(Image_processed, [0])
```

### Post-processing

```python
# Remove small objects
labels_clean = sp.remove_small_objects(labels, min_size=50)

# Remove large objects
labels_clean = sp.remove_large_objects(labels_clean, max_size=1000)

# Rescue missing cells
labels_rescued = sp.rescue_cells(Image, labels_clean, [0])
```

## Spatial Data Integration

### Adding Spatial Information

```python
# Extract spatial coordinates
coordinates = sp.get_cell_coordinates(labels)

# Add to AnnData
adata.obsm['spatial'] = coordinates

# Add cell areas
areas = sp.get_cell_areas(labels)
adata.obs['area'] = areas
```

### Multi-sample Analysis

```python
# Combine multiple samples
import scanpy as sc

# List of AnnData objects
adata_list = [adata1, adata2, adata3]

# Concatenate
adata_combined = sc.concat(adata_list, 
                          keys=['sample1', 'sample2', 'sample3'],
                          index_unique='-')

# Add sample information
adata_combined.obs['sample'] = adata_combined.obs.index.str.split('-').str[0]
```

## Data Quality Checks

### Image Quality

```python
# Check image properties
print(f"Image shape: {Image.shape}")
print(f"Data type: {Image.dtype}")
print(f"Value range: {Image.min()} - {Image.max()}")
print(f"Channels: {channel}")

# Check for issues
if Image.min() == Image.max():
    print("Warning: Image has no contrast")
if np.isnan(Image).any():
    print("Warning: Image contains NaN values")
```

### Segmentation Quality

```python
# Check segmentation results
print(f"Number of cells: {labels.max()}")
print(f"Cell size range: {np.bincount(labels.ravel())[1:].min()} - {np.bincount(labels.ravel())[1:].max()}")

# Visualize distribution
import matplotlib.pyplot as plt
plt.hist(np.bincount(labels.ravel())[1:], bins=50)
plt.xlabel('Cell Area (pixels)')
plt.ylabel('Frequency')
plt.title('Cell Size Distribution')
plt.show()
```

### Feature Quality

```python
# Check feature extraction
print(f"Expression matrix shape: {adata.X.shape}")
print(f"Missing values: {np.isnan(adata.X).sum()}")
print(f"Zero values: {(adata.X == 0).sum()}")

# Check spatial coordinates
print(f"Spatial coordinates shape: {adata.obsm['spatial'].shape}")
print(f"Coordinate range: X({adata.obsm['spatial'][:, 0].min():.1f}, {adata.obsm['spatial'][:, 0].max():.1f})")
print(f"Coordinate range: Y({adata.obsm['spatial'][:, 1].min():.1f}, {adata.obsm['spatial'][:, 1].max():.1f})")
```

## File I/O Operations

### Saving Results

```python
# Save AnnData
adata.write('results.h5ad')

# Save segmentation labels
import tifffile
tifffile.imwrite('segmentation_labels.tiff', labels)

# Save coordinates
import pandas as pd
coords_df = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'])
coords_df.to_csv('cell_coordinates.csv', index=False)
```

### Loading Results

```python
# Load AnnData
import scanpy as sc
adata = sc.read('results.h5ad')

# Load segmentation labels
labels = tifffile.imread('segmentation_labels.tiff')

# Load coordinates
coords_df = pd.read_csv('cell_coordinates.csv')
coordinates = coords_df[['x', 'y']].values
```

## Best Practices

1. **Use TIFF format** for multi-channel images
2. **Check data types** and value ranges
3. **Validate segmentation** results visually
4. **Save intermediate results** for reproducibility
5. **Document data sources** and processing steps
6. **Use AnnData format** for downstream analysis
