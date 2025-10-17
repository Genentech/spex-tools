# Preprocessing API Reference

This module provides comprehensive data preprocessing functions for spatial omics data analysis (proteomics and transcriptomics) in SPEX.

## Functions

### preprocess

```python
preprocess(adata, scale_max=10, size_factor=None, do_QC=False)
```

Comprehensive preprocessing pipeline for AnnData objects.

Performs quality control, normalization, feature selection, and scaling in a single function call.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | AnnData | - | AnnData object to preprocess |
| `scale_max` | float | 10 | Maximum value for scaling (clips larger values) |
| `size_factor` | float | None | Size factor for normalization. If None, uses median library size |
| `do_QC` | bool | False | Whether to perform quality control filtering |

**Returns:**

| Type | Description |
|------|-------------|
| AnnData | Preprocessed AnnData object |

**Notes:**

The preprocessing pipeline includes the following steps:

1. **Quality Control** (if `do_QC=True`):
   - Filter cells with low gene counts using MAD threshold
   - Filter genes with low cell counts
   - Remove cells with low total counts

2. **Normalization**:
   - Calculate size factors (median library size if not provided)
   - Normalize total counts per cell
   - Apply log1p transformation

3. **Feature Selection**:
   - Identify highly variable genes
   - Account for batch effects if present

4. **Scaling**:
   - Z-transform data
   - Clip values to `scale_max`

**Examples:**

```python
import spex as sp
import scanpy as sc

# Load data
adata = sc.read_h5ad("data.h5ad")

# Basic preprocessing
adata = sp.preprocess(adata)

# Preprocessing with quality control
adata = sp.preprocess(
    adata,
    scale_max=10,
    do_QC=True
)

# Custom size factor
adata = sp.preprocess(
    adata,
    size_factor=10000,
    do_QC=True
)
```

**Storage:**

Results are stored in the AnnData object:
- `adata.raw`: Original data backup
- `adata.uns['prepro']`: Preprocessing parameters
- `adata.var.highly_variable`: Highly variable gene flags
- `adata.layers['counts']`: Normalized count data

---

### MAD_threshold

```python
MAD_threshold(variable, ndevs=2.5)
```

Calculate threshold using Median Absolute Deviation (MAD).

This function calculates a robust threshold value using the MAD method, which is less sensitive to outliers than standard deviation-based methods.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variable` | array-like | - | Input data array |
| `ndevs` | float | 2.5 | Number of MAD deviations from the median |

**Returns:**

| Type | Description |
|------|-------------|
| float | Threshold value calculated as median - ndevs * MAD |

**Notes:**

- MAD is calculated as `median(|x - median(x)|)`
- The threshold is computed as `median - ndevs * MAD`
- This method is robust to outliers and non-normal distributions
- Commonly used for quality control filtering in single-cell data

**Examples:**

```python
import spex as sp
import numpy as np

# Calculate threshold for total counts
total_counts = adata.obs['total_counts']
threshold = sp.MAD_threshold(total_counts, ndevs=2.5)

# Filter cells below threshold
filtered_adata = adata[adata.obs['total_counts'] > threshold, :]

# Use different threshold for genes
n_genes = adata.obs['n_genes_by_counts']
gene_threshold = sp.MAD_threshold(n_genes, ndevs=1.5)
```

---

### should_batch_correct

```python
should_batch_correct(adata)
```

Check if batch correction should be performed.

This function checks if batch correction is needed by looking for batch information in the AnnData object.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `adata` | AnnData | AnnData object to check for batch information |

**Returns:**

| Type | Description |
|------|-------------|
| bool | True if batch correction should be performed, False otherwise |

**Notes:**

- Checks for 'batch_key' in `adata.uns`
- Returns True if batch_key exists and is not None
- Used internally by preprocessing functions to determine batch correction strategy

**Examples:**

```python
import spex as sp

# Check if batch correction is needed
needs_batch_correction = sp.should_batch_correct(adata)

if needs_batch_correction:
    print(f"Batch correction will be performed using: {adata.uns['batch_key']}")
else:
    print("No batch correction needed")
```

---

### reduce_dimensionality

```python
reduce_dimensionality(adata, prefilter=False, method='pca', mdist=0.5, n_neighbors=None, latent_dim=None)
```

Reduce dimensionality of AnnData object.

This function performs dimensionality reduction using various methods including PCA, UMAP, t-SNE, and scVI. It also constructs neighborhood graphs for downstream analysis.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | AnnData | - | AnnData object to reduce dimensionality |
| `prefilter` | bool | False | Whether to prefilter highly variable genes |
| `method` | str | 'pca' | Dimensionality reduction method. Options: 'pca', 'umap', 'tsne', 'scvi', 'diff_map' |
| `mdist` | float | 0.5 | Minimum distance for UMAP |
| `n_neighbors` | int | None | Number of neighbors for UMAP/t-SNE. If None, uses sqrt(n_cells) |
| `latent_dim` | int | None | Number of latent dimensions. If None, uses automatic estimation |

**Returns:**

| Type | Description |
|------|-------------|
| AnnData | Updated AnnData object with dimensionality reduction results |

**Notes:**

**Supported Methods:**

- **PCA**: Principal Component Analysis (fastest, most common)
- **UMAP**: Uniform Manifold Approximation and Projection (good for visualization)
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding (good for visualization)
- **scVI**: Single-Cell Variational Inference (good for batch correction)
- **diff_map**: Diffusion maps (good for trajectory analysis)

**Automatic Parameter Estimation:**

- `n_neighbors`: If None, uses `sqrt(n_cells)`
- `latent_dim`: If None, uses elbow method on singular values

**Batch Correction:**

- If batch information is present, applies Harmony integration
- Results stored in `adata.obsm['X_pca_harmony']`

**Examples:**

```python
import spex as sp

# Basic PCA reduction
adata = sp.reduce_dimensionality(adata, method='pca')

# UMAP with custom parameters
adata = sp.reduce_dimensionality(
    adata,
    method='umap',
    n_neighbors=30,
    latent_dim=50,
    mdist=0.3
)

# scVI for batch correction
adata = sp.reduce_dimensionality(
    adata,
    method='scvi',
    latent_dim=20
)
```

**Storage:**

Results are stored in the AnnData object:
- `adata.obsm['X_pca']`: PCA coordinates
- `adata.obsm['X_umap']`: UMAP coordinates
- `adata.obsm['X_tsne']`: t-SNE coordinates (if computed)
- `adata.obsm['X_scvi']`: scVI latent space (if computed)
- `adata.obsp['connectivities']`: Neighborhood graph
- `adata.obsm['distances']`: Distance matrix

---

### load_anndata

```python
load_anndata(path=None, files=None)
```

Load AnnData objects from file(s).

This function loads AnnData objects from single or multiple files. When loading multiple files, they are combined into a single AnnData object.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | None | Path to a single AnnData file (.h5ad) |
| `files` | list | None | List of file paths to load and combine |

**Returns:**

| Type | Description |
|------|-------------|
| dict | Dictionary containing the loaded AnnData object(s): 'adata': AnnData object |

**Notes:**

- If both `path` and `files` are provided, `files` takes precedence
- When combining multiple files, a 'filename' column is added to `adata.obs`
- Files are combined using `scanpy.concat` with outer join
- Supports .h5ad format files

**Examples:**

```python
import spex as sp

# Load single file
result = sp.load_anndata(path="data.h5ad")
adata = result['adata']

# Load multiple files
files = ["sample1.h5ad", "sample2.h5ad", "sample3.h5ad"]
result = sp.load_anndata(files=files)
combined_adata = result['adata']

# Check combined data
print(f"Combined shape: {combined_adata.shape}")
print(f"Files included: {combined_adata.obs['filename'].unique()}")
```

## Best Practices

### Quality Control

1. **Always perform QC** before analysis:
   ```python
   adata = sp.preprocess(adata, do_QC=True)
   ```

2. **Check data quality metrics**:
   ```python
   import scanpy as sc
   sc.pp.calculate_qc_metrics(adata, inplace=True)
   print(f"Cells: {adata.n_obs}, Genes: {adata.n_vars}")
   ```

3. **Use appropriate thresholds**:
   ```python
   # Conservative thresholds for high-quality data
   threshold_counts = sp.MAD_threshold(adata.obs['total_counts'], ndevs=2.0)
   threshold_genes = sp.MAD_threshold(adata.obs['n_genes_by_counts'], ndevs=1.5)
   ```

### Dimensionality Reduction

1. **Choose appropriate method**:
   - **PCA**: For fast preprocessing and clustering
   - **UMAP**: For visualization and exploration
   - **scVI**: For batch correction and integration

2. **Estimate parameters automatically**:
   ```python
   # Let the function estimate optimal parameters
   adata = sp.reduce_dimensionality(adata, method='pca')
   ```

3. **Check results**:
   ```python
   # Verify dimensionality reduction worked
   print(f"PCA components: {adata.obsm['X_pca'].shape}")
   print(f"UMAP coordinates: {adata.obsm['X_umap'].shape}")
   ```

### Batch Correction

1. **Set up batch information**:
   ```python
   adata.uns['batch_key'] = 'batch'
   ```

2. **Use batch-aware preprocessing**:
   ```python
   # Preprocessing will automatically detect and handle batches
   adata = sp.preprocess(adata, do_QC=True)
   ```

3. **Verify batch correction**:
   ```python
   # Check if batch correction was applied
   if 'X_pca_harmony' in adata.obsm:
       print("Batch correction applied successfully")
   ```

## Troubleshooting

### Common Issues

1. **Memory errors during preprocessing**:
   - Reduce `scale_max` parameter
   - Use data subsampling
   - Increase available RAM

2. **Slow dimensionality reduction**:
   - Use `prefilter=True` to reduce gene number
   - Choose faster methods (PCA over UMAP)
   - Reduce `latent_dim` parameter

3. **Batch correction not working**:
   - Ensure `adata.uns['batch_key']` is set
   - Check that batch column exists in `adata.obs`
   - Verify batch column contains valid values

4. **File loading errors**:
   - Check file paths are correct
   - Ensure files are in .h5ad format
   - Verify file permissions

### Performance Optimization

1. **Use appropriate data types**:
   ```python
   # Convert to float32 for memory efficiency
   adata.X = adata.X.astype('float32')
   ```

2. **Subsample large datasets**:
   ```python
   # Random subsampling
   import scanpy as sc
   sc.pp.subsample(adata, n_obs=10000, random_state=42)
   ```

3. **Save intermediate results**:
   ```python
   # Save after preprocessing
   adata.write("preprocessed_data.h5ad")
   ```

## Integration with Other Modules

### With Clustering

```python
# Complete preprocessing for clustering
adata = sp.preprocess(adata, do_QC=True)
adata = sp.reduce_dimensionality(adata, method='pca')
adata = sp.cluster(adata, method='leiden')
```

### With Spatial Analysis

```python
# Preprocessing for spatial analysis
adata = sp.preprocess(adata, do_QC=True)
adata = sp.reduce_dimensionality(adata, method='pca')
# Spatial analysis functions can now use preprocessed data
```

### With Segmentation Results

```python
# Load and preprocess feature extraction results
adata = sp.load_anndata(path="segmentation_features.h5ad")['adata']
adata = sp.preprocess(adata, do_QC=True)
```
