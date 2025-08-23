# 🔄 Complete Pipeline Example

## Overview

This example demonstrates a complete spatial transcriptomics analysis pipeline from image loading to spatial analysis using SPEX. We'll process a multi-channel fluorescence image, segment cells, extract features, perform clustering, and analyze spatial relationships.

## Prerequisites

```python
import spex as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy import stats

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')
```

## Step 1: Data Loading and Preprocessing

### Load Image

```python
# Load multi-channel image
Image, channel = sp.load_image('sample_data.tiff')

print(f"Image shape: {Image.shape}")
print(f"Channels: {channel}")

# Display all channels
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

### Preprocess Image

```python
# Background subtraction for all channels
Image_bg = sp.background_subtract(Image, list(range(len(channel))))

# Denoise using non-local means
Image_denoised = sp.nlm_denoise(Image_bg, list(range(len(channel))))

# Compare preprocessing results
fig, axes = plt.subplots(2, len(channel), figsize=(4*len(channel), 8))

for i, ch in enumerate(channel):
    # Original
    axes[0, i].imshow(Image[:, :, i], cmap='gray')
    axes[0, i].set_title(f'Original - {ch}')
    axes[0, i].axis('off')
    
    # Processed
    axes[1, i].imshow(Image_denoised[:, :, i], cmap='gray')
    axes[1, i].set_title(f'Processed - {ch}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()
```

## Step 2: Cell Segmentation

### Perform Segmentation

```python
# Use Cellpose for robust cell segmentation
labels = sp.cellpose_cellseg(Image_denoised, [0], diameter=30)

print(f"Initial segmentation: {labels.max()} cells detected")

# Clean segmentation results
labels_clean = sp.remove_small_objects(labels, min_size=50)
labels_clean = sp.remove_large_objects(labels_clean, max_size=1000)

print(f"After cleaning: {labels_clean.max()} cells")

# Rescue missing cells
labels_final = sp.rescue_cells(Image_denoised, labels_clean, [0])

print(f"Final segmentation: {labels_final.max()} cells")
```

### Visualize Segmentation

```python
# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Original image
axes[0, 0].imshow(Image[:, :, 0], cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# Initial segmentation
axes[0, 1].imshow(Image[:, :, 0], cmap='gray')
axes[0, 1].imshow(labels, cmap='tab20', alpha=0.5)
axes[0, 1].set_title(f'Initial Segmentation ({labels.max()} cells)')
axes[0, 1].axis('off')

# Cleaned segmentation
axes[0, 2].imshow(Image[:, :, 0], cmap='gray')
axes[0, 2].imshow(labels_clean, cmap='tab20', alpha=0.5)
axes[0, 2].set_title(f'Cleaned ({labels_clean.max()} cells)')
axes[0, 2].axis('off')

# Final segmentation
axes[1, 0].imshow(Image[:, :, 0], cmap='gray')
axes[1, 0].imshow(labels_final, cmap='tab20', alpha=0.5)
axes[1, 0].set_title(f'Final ({labels_final.max()} cells)')
axes[1, 0].axis('off')

# Cell size distribution
areas = np.bincount(labels_final.ravel())[1:]
axes[1, 1].hist(areas, bins=30, alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Cell Area (pixels)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Cell Size Distribution')

# Cell coordinates
coordinates = sp.get_cell_coordinates(labels_final)
axes[1, 2].scatter(coordinates[:, 0], coordinates[:, 1], alpha=0.6, s=20)
axes[1, 2].set_xlabel('X Position')
axes[1, 2].set_ylabel('Y Position')
axes[1, 2].set_title('Cell Positions')
axes[1, 2].set_aspect('equal')

plt.tight_layout()
plt.show()
```

## Step 3: Feature Extraction

### Extract Features

```python
# Extract features into AnnData format
adata = sp.feature_extraction_adata(Image_denoised, labels_final, channel)

print(f"AnnData shape: {adata.X.shape}")
print(f"Features: {list(adata.var_names)}")
print(f"Cells: {adata.n_obs}")

# Add spatial information
adata.obsm['spatial'] = coordinates

# Add cell areas
areas = sp.get_cell_areas(labels_final)
adata.obs['area'] = areas

# Add cell properties
properties = sp.get_cell_properties(Image_denoised, labels_final, list(range(len(channel))))
adata.obs['shape_factor'] = properties['shape_factors']
adata.obs['eccentricity'] = properties['eccentricities']
```

### Quality Control

```python
# Basic QC metrics
print("Quality Control Metrics:")
print("-" * 40)
print(f"Total cells: {adata.n_obs}")
print(f"Total features: {adata.n_vars}")
print(f"Mean cell area: {adata.obs['area'].mean():.1f} pixels")
print(f"Area range: {adata.obs['area'].min():.1f} - {adata.obs['area'].max():.1f} pixels")

# Check for missing values
missing_values = np.isnan(adata.X).sum()
print(f"Missing values: {missing_values}")

# Check expression levels
mean_expression = adata.X.mean(axis=0)
print(f"Mean expression per feature: {mean_expression.min():.3f} - {mean_expression.max():.3f}")

# Plot QC metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Cell area distribution
axes[0, 0].hist(adata.obs['area'], bins=30, alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Cell Area (pixels)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Cell Size Distribution')

# Expression distribution
axes[0, 1].hist(adata.X.flatten(), bins=50, alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Expression Level')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Expression Distribution')

# Shape factor distribution
axes[1, 0].hist(adata.obs['shape_factor'], bins=30, alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Shape Factor')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Cell Shape Distribution')

# Spatial distribution
axes[1, 1].scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1], 
                   c=adata.obs['area'], s=20, alpha=0.6, cmap='viridis')
axes[1, 1].set_xlabel('X Position')
axes[1, 1].set_ylabel('Y Position')
axes[1, 1].set_title('Cell Positions (colored by area)')
axes[1, 1].set_aspect('equal')

plt.tight_layout()
plt.show()
```

## Step 4: Data Preprocessing

### Normalize and Scale

```python
# Normalize data
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Scale data
sc.pp.scale(adata, max_value=10)

# Find highly variable features
sc.pp.highly_variable_features(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
print(f"Highly variable features: {sum(adata.var.highly_variable)}")

# Plot highly variable features
sc.pl.highly_variable_features(adata)
plt.show()
```

### Dimensionality Reduction

```python
# PCA
sc.tl.pca(adata, use_highly_variable=True)
sc.pl.pca(adata, color=channel[0] if len(channel) > 0 else 'area')
plt.show()

# Compute neighborhood graph
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)

# UMAP
sc.tl.umap(adata)
sc.pl.umap(adata, color=channel[0] if len(channel) > 0 else 'area')
plt.show()
```

## Step 5: Clustering Analysis

### Perform Clustering

```python
# Leiden clustering
sc.tl.leiden(adata, resolution=0.5)
print(f"Number of clusters: {len(adata.obs['leiden'].unique())}")

# Louvain clustering (alternative)
sc.tl.louvain(adata, resolution=0.5)
print(f"Number of clusters (Louvain): {len(adata.obs['louvain'].unique())}")

# Visualize clusters
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# UMAP with Leiden clusters
sc.pl.umap(adata, color='leiden', ax=axes[0, 0], show=False)
axes[0, 0].set_title('Leiden Clusters (UMAP)')

# UMAP with Louvain clusters
sc.pl.umap(adata, color='louvain', ax=axes[0, 1], show=False)
axes[0, 1].set_title('Louvain Clusters (UMAP)')

# Spatial plot with Leiden clusters
sc.pl.spatial(adata, color='leiden', ax=axes[1, 0], show=False)
axes[1, 0].set_title('Leiden Clusters (Spatial)')

# Spatial plot with Louvain clusters
sc.pl.spatial(adata, color='louvain', ax=axes[1, 1], show=False)
axes[1, 1].set_title('Louvain Clusters (Spatial)')

plt.tight_layout()
plt.show()
```

### Cluster Analysis

```python
# Find marker features for each cluster
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

# Plot top markers
sc.pl.rank_genes_groups(adata, n_genes=5, sharey=False)
plt.show()

# Get marker features
markers = sc.get.rank_genes_groups_df(adata, group='0')
print("Top markers for cluster 0:")
print(markers.head(10))

# Plot expression of top markers
top_markers = markers.head(5)['names'].tolist()
sc.pl.umap(adata, color=top_markers, ncols=3)
plt.show()
```

## Step 6: Spatial Analysis

### Spatial Distribution Analysis

```python
# Analyze spatial distribution of clusters
cluster_positions = {}
for cluster in adata.obs['leiden'].unique():
    mask = adata.obs['leiden'] == cluster
    cluster_positions[cluster] = adata.obsm['spatial'][mask]

# Plot spatial distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_positions)))

for i, (cluster, positions) in enumerate(cluster_positions.items()):
    row = i // 3
    col = i % 3
    
    axes[row, col].scatter(positions[:, 0], positions[:, 1], 
                          c=[colors[i]], s=20, alpha=0.7)
    axes[row, col].set_title(f'Cluster {cluster} ({len(positions)} cells)')
    axes[row, col].set_aspect('equal')
    axes[row, col].set_xlabel('X Position')
    axes[row, col].set_ylabel('Y Position')

plt.tight_layout()
plt.show()
```

### Spatial Autocorrelation

```python
# Calculate spatial autocorrelation for each feature
from scipy.spatial.distance import pdist, squareform

def spatial_autocorrelation(positions, values):
    """Calculate Moran's I for spatial autocorrelation"""
    if len(positions) < 2:
        return 0
    
    # Calculate distances
    dist_matrix = squareform(pdist(positions))
    
    # Calculate weights (inverse distance)
    weights = 1 / (dist_matrix + 1e-10)  # Add small value to avoid division by zero
    np.fill_diagonal(weights, 0)
    
    # Calculate Moran's I
    n = len(values)
    mean_val = np.mean(values)
    variance = np.var(values)
    
    if variance == 0:
        return 0
    
    numerator = 0
    denominator = 0
    
    for i in range(n):
        for j in range(n):
            if i != j:
                numerator += weights[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
                denominator += weights[i, j]
    
    if denominator == 0:
        return 0
    
    moran_i = (n / (2 * denominator)) * (numerator / variance)
    return moran_i

# Calculate spatial autocorrelation for each feature
spatial_autocorr = {}
for feature in adata.var_names[:10]:  # Use first 10 features for speed
    values = adata[:, feature].X.flatten()
    moran_i = spatial_autocorrelation(adata.obsm['spatial'], values)
    spatial_autocorr[feature] = moran_i

# Plot spatial autocorrelation
fig, ax = plt.subplots(figsize=(10, 6))
features = list(spatial_autocorr.keys())
values = list(spatial_autocorr.values())

bars = ax.bar(features, values, color='skyblue', alpha=0.7)
ax.set_xlabel('Features')
ax.set_ylabel("Moran's I")
ax.set_title('Spatial Autocorrelation by Feature')
ax.tick_params(axis='x', rotation=45)

# Add horizontal line at 0
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

print("Spatial autocorrelation results:")
for feature, value in spatial_autocorr.items():
    print(f"{feature}: {value:.3f}")
```

## Step 7: Differential Expression Analysis

### Compare Clusters

```python
# Compare all clusters
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

# Get results for all comparisons
all_markers = sc.get.rank_genes_groups_df(adata, group=None)

# Filter significant markers
significant_markers = all_markers[all_markers['pvals_adj'] < 0.05]
print(f"Significant markers: {len(significant_markers)}")

# Plot heatmap of top markers
top_markers_per_cluster = []
for cluster in adata.obs['leiden'].unique():
    cluster_markers = all_markers[all_markers['group'] == cluster]
    top_markers_per_cluster.extend(cluster_markers.head(3)['names'].tolist())

# Remove duplicates
top_markers_per_cluster = list(set(top_markers_per_cluster))

# Plot heatmap
sc.pl.heatmap(adata, top_markers_per_cluster, groupby='leiden', 
              use_raw=True, show_gene_labels=True)
plt.show()
```

## Step 8: Save Results

### Export Results

```python
# Save AnnData object
adata.write('complete_pipeline_results.h5ad')

# Save segmentation labels
import tifffile
tifffile.imwrite('segmentation_labels.tiff', labels_final)

# Save cluster assignments
cluster_df = adata.obs[['leiden', 'louvain', 'area', 'shape_factor']].copy()
cluster_df['x'] = adata.obsm['spatial'][:, 0]
cluster_df['y'] = adata.obsm['spatial'][:, 1]
cluster_df.to_csv('cell_clusters.csv')

# Save marker genes
all_markers.to_csv('marker_genes.csv')

print("Results saved:")
print("- complete_pipeline_results.h5ad: AnnData object")
print("- segmentation_labels.tiff: Segmentation labels")
print("- cell_clusters.csv: Cell metadata and cluster assignments")
print("- marker_genes.csv: Differential expression results")
```

## Summary

This complete pipeline demonstrates:

1. **Image preprocessing** with background subtraction and denoising
2. **Cell segmentation** using AI-based methods with post-processing
3. **Feature extraction** into structured AnnData format
4. **Quality control** and data preprocessing
5. **Clustering analysis** using multiple algorithms
6. **Spatial analysis** including distribution and autocorrelation
7. **Differential expression** analysis between clusters
8. **Result export** in multiple formats

The pipeline is modular and can be adapted for different experimental designs and analysis goals.

## Next Steps

- 🎯 [Clustering Guide](../tutorials/clustering-guide.md) - Advanced clustering techniques
- 🧬 [Spatial Analysis](../tutorials/spatial-analysis.md) - Advanced spatial methods
- 📦 [Batch Processing](batch-processing.md) - Process multiple samples
- 🎨 [Visualization](visualization.md) - Advanced plotting techniques
