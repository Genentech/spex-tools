# Data Preprocessing Guide

This tutorial provides a comprehensive guide to preprocessing spatial omics data (proteomics and transcriptomics) using SPEX. Preprocessing is a crucial step that prepares your data for downstream analysis including clustering, spatial analysis, and differential expression.

## Overview

The preprocessing pipeline in SPEX consists of several key steps:

1. **Data Loading**: Loading AnnData objects from files
2. **Quality Control**: Filtering low-quality cells and genes
3. **Normalization**: Correcting for technical biases
4. **Feature Selection**: Identifying highly variable genes
5. **Dimensionality Reduction**: Reducing data complexity
6. **Batch Correction**: Handling multiple samples/datasets

## Prerequisites

Before starting this tutorial, make sure you have:

```python
import spex as sp
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## Step 1: Loading Data

### Loading Single Files

```python
# Load a single AnnData file
result = sp.load_anndata(path="your_data.h5ad")
adata = result['adata']

print(f"Data shape: {adata.shape}")
print(f"Number of cells: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")
```

### Loading Multiple Files

```python
# Load multiple files and combine them
files = [
    "sample1.h5ad",
    "sample2.h5ad", 
    "sample3.h5ad"
]

result = sp.load_anndata(files=files)
combined_adata = result['adata']

print(f"Combined data shape: {combined_adata.shape}")
print(f"Files included: {combined_adata.obs['filename'].unique()}")
```

### Data Inspection

```python
# Check data structure
print("Data structure:")
print(f"- Expression matrix: {adata.X.shape}")
print(f"- Cell annotations: {list(adata.obs.columns)}")
print(f"- Gene annotations: {list(adata.var.columns)}")
print(f"- Cell embeddings: {list(adata.obsm.keys())}")

# Check for spatial coordinates
if 'spatial' in adata.obsm:
    print(f"Spatial coordinates found: {adata.obsm['spatial'].shape}")
else:
    print("No spatial coordinates found")
```

## Step 2: Quality Control

### Calculating QC Metrics

```python
# Calculate quality control metrics
sc.pp.calculate_qc_metrics(adata, inplace=True)

# Display QC summary
print("Quality Control Summary:")
print(f"Total counts per cell - Mean: {adata.obs['total_counts'].mean():.0f}")
print(f"Total counts per cell - Median: {adata.obs['total_counts'].median():.0f}")
print(f"Genes per cell - Mean: {adata.obs['n_genes_by_counts'].mean():.0f}")
print(f"Genes per cell - Median: {adata.obs['n_genes_by_counts'].median():.0f}")
```

### Visualizing QC Metrics

```python
# Create QC plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Total counts distribution
axes[0, 0].hist(adata.obs['total_counts'], bins=50, alpha=0.7)
axes[0, 0].set_xlabel('Total Counts')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Total Counts')

# Genes per cell distribution
axes[0, 1].hist(adata.obs['n_genes_by_counts'], bins=50, alpha=0.7)
axes[0, 1].set_xlabel('Genes per Cell')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Genes per Cell')

# Scatter plot: counts vs genes
axes[1, 0].scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'], alpha=0.5)
axes[1, 0].set_xlabel('Total Counts')
axes[1, 0].set_ylabel('Genes per Cell')
axes[1, 0].set_title('Counts vs Genes')

# Mitochondrial genes (if available)
if 'mt' in adata.var.columns:
    axes[1, 1].hist(adata.obs['pct_counts_mt'], bins=50, alpha=0.7)
    axes[1, 1].set_xlabel('Mitochondrial %')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Mitochondrial Gene Percentage')
else:
    axes[1, 1].text(0.5, 0.5, 'No mitochondrial genes\nannotated', 
                    ha='center', va='center', transform=axes[1, 1].transAxes)

plt.tight_layout()
plt.show()
```

### Setting QC Thresholds

```python
# Calculate MAD-based thresholds
counts_threshold = sp.MAD_threshold(adata.obs['total_counts'], ndevs=2.5)
genes_threshold = sp.MAD_threshold(adata.obs['n_genes_by_counts'], ndevs=1.5)

print(f"Counts threshold (MAD): {counts_threshold:.0f}")
print(f"Genes threshold (MAD): {genes_threshold:.0f}")

# Filter cells based on thresholds
cells_before = adata.n_obs
adata = adata[adata.obs['total_counts'] > counts_threshold, :]
adata = adata[adata.obs['n_genes_by_counts'] > genes_threshold, :]
cells_after = adata.n_obs

print(f"Cells filtered: {cells_before - cells_after} ({100*(cells_before-cells_after)/cells_before:.1f}%)")
```

## Step 3: Preprocessing Pipeline

### Basic Preprocessing

```python
# Run the complete preprocessing pipeline
adata = sp.preprocess(
    adata,
    scale_max=10,
    do_QC=True
)

print("Preprocessing completed!")
print(f"Final data shape: {adata.shape}")
print(f"Highly variable genes: {adata.var.highly_variable.sum()}")
```

### Custom Preprocessing

```python
# Preprocessing with custom parameters
adata = sp.preprocess(
    adata,
    scale_max=8,           # More conservative scaling
    size_factor=10000,     # Custom size factor
    do_QC=True            # Enable quality control
)

# Check preprocessing results
print("Preprocessing parameters:")
print(f"- Scale max: {adata.uns.get('prepro', {}).get('scale_max', 'Not set')}")
print(f"- Size factor: {adata.uns.get('prepro', {}).get('size_factor', 'Not set')}")
print(f"- QC thresholds: {adata.uns.get('prepro', {}).get('total_count_thresh', 'Not set')}")
```

### Batch-Aware Preprocessing

```python
# Set up batch information (if you have multiple samples)
adata.uns['batch_key'] = 'batch'  # Column name in adata.obs

# Check if batch correction will be applied
needs_batch_correction = sp.should_batch_correct(adata)
print(f"Batch correction needed: {needs_batch_correction}")

# Run preprocessing with batch correction
adata = sp.preprocess(adata, do_QC=True)
```

## Step 4: Dimensionality Reduction

### PCA Reduction

```python
# Basic PCA reduction
adata = sp.reduce_dimensionality(
    adata,
    method='pca',
    latent_dim=50
)

print(f"PCA components: {adata.obsm['X_pca'].shape}")
print(f"Explained variance: {np.sum(adata.varm['PCs'].var(axis=0)[:50]):.2f}")
```

### UMAP Visualization

```python
# UMAP for visualization
adata = sp.reduce_dimensionality(
    adata,
    method='umap',
    n_neighbors=30,
    mdist=0.3
)

# Plot UMAP
plt.figure(figsize=(8, 6))
plt.scatter(
    adata.obsm['X_umap'][:, 0],
    adata.obsm['X_umap'][:, 1],
    s=20,
    alpha=0.7
)
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('UMAP Visualization')
plt.show()
```

### Advanced Dimensionality Reduction

```python
# scVI for batch correction
if sp.should_batch_correct(adata):
    adata = sp.reduce_dimensionality(
        adata,
        method='scvi',
        latent_dim=20
    )
    print("scVI reduction completed with batch correction")
```

## Step 5: Data Validation

### Check Preprocessing Results

```python
# Verify preprocessing worked correctly
print("Preprocessing Validation:")
print(f"✓ Data shape: {adata.shape}")
print(f"✓ Highly variable genes: {adata.var.highly_variable.sum()}")
print(f"✓ PCA components: {'X_pca' in adata.obsm}")
print(f"✓ UMAP coordinates: {'X_umap' in adata.obsm}")
print(f"✓ Neighborhood graph: {'connectivities' in adata.obsp}")
print(f"✓ Raw data backup: {adata.raw is not None}")

# Check data ranges
print(f"\nData ranges:")
print(f"- Expression matrix: {adata.X.min():.2f} to {adata.X.max():.2f}")
print(f"- PCA components: {adata.obsm['X_pca'].min():.2f} to {adata.obsm['X_pca'].max():.2f}")
```

### Quality Metrics After Preprocessing

```python
# Calculate and display quality metrics
sc.pp.calculate_qc_metrics(adata, inplace=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Total counts after preprocessing
axes[0].hist(adata.obs['total_counts'], bins=30, alpha=0.7)
axes[0].set_xlabel('Total Counts')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Total Counts (Post-Preprocessing)')

# Genes per cell after preprocessing
axes[1].hist(adata.obs['n_genes_by_counts'], bins=30, alpha=0.7)
axes[1].set_xlabel('Genes per Cell')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Genes per Cell (Post-Preprocessing)')

# Expression distribution
axes[2].hist(adata.X.flatten(), bins=50, alpha=0.7)
axes[2].set_xlabel('Expression Values')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Expression Distribution')

plt.tight_layout()
plt.show()
```

## Step 6: Saving Preprocessed Data

```python
# Save preprocessed data
adata.write("preprocessed_data.h5ad")

# Save preprocessing parameters
preprocessing_params = {
    'scale_max': 10,
    'size_factor': adata.uns.get('prepro', {}).get('size_factor'),
    'qc_thresholds': {
        'total_counts': adata.uns.get('prepro', {}).get('total_count_thresh'),
        'n_genes': adata.uns.get('prepro', {}).get('n_genes_by_count_thresh')
    },
    'n_hvgs': adata.var.highly_variable.sum(),
    'n_pcs': adata.obsm['X_pca'].shape[1] if 'X_pca' in adata.obsm else 0
}

import json
with open("preprocessing_params.json", "w") as f:
    json.dump(preprocessing_params, f, indent=2)

print("Preprocessed data and parameters saved!")
```

## Advanced Preprocessing Workflows

### Multi-Sample Integration

```python
# Load multiple samples
files = ["sample1.h5ad", "sample2.h5ad", "sample3.h5ad"]
result = sp.load_anndata(files=files)
adata = result['adata']

# Add batch information
adata.obs['batch'] = adata.obs['filename']
adata.uns['batch_key'] = 'batch'

# Preprocess with batch correction
adata = sp.preprocess(adata, do_QC=True)
adata = sp.reduce_dimensionality(adata, method='pca')

print(f"Integrated data shape: {adata.shape}")
print(f"Batches: {adata.obs['batch'].unique()}")
```

### Large Dataset Processing

```python
# For very large datasets, use subsampling
if adata.n_obs > 50000:
    print("Large dataset detected, subsampling...")
    sc.pp.subsample(adata, n_obs=50000, random_state=42)

# Use memory-efficient preprocessing
adata.X = adata.X.astype('float32')  # Reduce memory usage
adata = sp.preprocess(adata, do_QC=True)
```

### Spatial Data Preprocessing

```python
# For spatial omics data (proteomics or transcriptomics)
if 'spatial' in adata.obsm:
    print("Spatial data detected!")
    
    # Check spatial coordinates
    spatial_coords = adata.obsm['spatial']
    print(f"Spatial coordinates shape: {spatial_coords.shape}")
    
    # Visualize spatial distribution
    plt.figure(figsize=(8, 6))
    plt.scatter(spatial_coords[:, 0], spatial_coords[:, 1], s=1, alpha=0.5)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Spatial Distribution of Cells')
    plt.axis('equal')
    plt.show()
```

## Troubleshooting

### Common Issues and Solutions

#### Memory Errors

```python
# Solution: Reduce data size
if adata.n_obs > 100000:
    sc.pp.subsample(adata, n_obs=50000, random_state=42)

# Use float32 instead of float64
adata.X = adata.X.astype('float32')
```

#### Slow Processing

```python
# Solution: Use faster methods
adata = sp.reduce_dimensionality(
    adata,
    method='pca',  # Faster than UMAP
    prefilter=True  # Reduce gene number
)
```

#### Batch Correction Issues

```python
# Solution: Check batch information
if 'batch_key' not in adata.uns:
    print("No batch key found. Setting up batch information...")
    adata.uns['batch_key'] = 'batch'  # Adjust to your column name

# Verify batch column exists
if adata.uns['batch_key'] not in adata.obs.columns:
    print(f"Batch column '{adata.uns['batch_key']}' not found in adata.obs")
```

#### Quality Control Problems

```python
# Solution: Adjust thresholds
# Use more conservative thresholds
counts_threshold = sp.MAD_threshold(adata.obs['total_counts'], ndevs=3.0)
genes_threshold = sp.MAD_threshold(adata.obs['n_genes_by_counts'], ndevs=2.0)

print(f"Conservative thresholds:")
print(f"- Counts: {counts_threshold:.0f}")
print(f"- Genes: {genes_threshold:.0f}")
```

## Best Practices Summary

1. **Always perform quality control** before analysis
2. **Use MAD-based thresholds** for robust filtering
3. **Check data quality metrics** before and after preprocessing
4. **Save intermediate results** for reproducibility
5. **Use appropriate data types** for memory efficiency
6. **Validate preprocessing results** before downstream analysis
7. **Document preprocessing parameters** for reproducibility
8. **Consider batch effects** in multi-sample studies

## Next Steps

After preprocessing, your data is ready for:

- [Clustering Analysis](clustering-guide.md)
- [Spatial Analysis](spatial-analysis.md)
- [Differential Expression Analysis](../api-reference/clustering.md#differential-expression)
- [Pathway Analysis](../api-reference/spatial-analysis.md#analyze_pathways)

The preprocessed AnnData object contains all necessary information for these downstream analyses.
