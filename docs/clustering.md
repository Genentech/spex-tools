# Clustering and Spatial Transcriptomics

This page provides comprehensive guidance on clustering and spatial transcriptomics analysis using SPEX.

## Overview

SPEX provides powerful tools for clustering single-cell data and performing spatial transcriptomics analysis. The library integrates seamlessly with AnnData objects and provides specialized functions for spatial-aware clustering and analysis.

## Key Features

- **PhenoGraph Clustering**: Advanced clustering algorithm for complex cell populations
- **Spatial-Aware Clustering**: Incorporates spatial information in clustering decisions
- **Differential Expression Analysis**: Find marker genes for each cluster
- **Pathway Analysis**: Enrichment analysis for biological pathways
- **Niche Analysis**: Identify spatial patterns and cell neighborhoods
- **AnnData Integration**: Full compatibility with AnnData ecosystem

## Quick Start

```python
import spex as sp
import anndata as ad

# Load your AnnData object
adata = ad.read_h5ad("your_data.h5ad")

# Perform PhenoGraph clustering
adata = sp.phenograph_cluster(
    adata, 
    channel_names=["CD3", "CD8", "CD20"], 
    knn=30
)

# Basic clustering with spatial information
adata = sp.cluster(
    adata,
    spatial_weight=0.5,
    resolution=1.0,
    method='leiden'
)

print(f"Number of clusters: {adata.obs['leiden'].nunique()}")
```

## Clustering Methods

### PhenoGraph Clustering

PhenoGraph is a clustering algorithm designed for high-dimensional data that automatically determines the optimal number of clusters.

```python
# PhenoGraph clustering with custom parameters
adata = sp.phenograph_cluster(
    adata,
    channel_names=["CD3", "CD8", "CD20", "CD68"],
    knn=30,                    # Number of neighbors
    transformation='arcsin',   # Data transformation
    scaling='z-score',         # Scaling method
    cofactor=5.0,             # Arcsinh cofactor
    umap_min_dist=0.5         # UMAP parameter
)
```

**Parameters:**
- `channel_names`: List of marker channels to use for clustering
- `knn`: Number of neighbors for graph construction
- `transformation`: Data transformation ('arcsin', 'log', 'none')
- `scaling`: Feature scaling ('z-score', 'winsorize', 'none')
- `cofactor`: Cofactor for arcsinh transformation
- `umap_min_dist`: Minimum distance for UMAP

### Spatial-Aware Clustering

Incorporate spatial information into clustering decisions:

```python
# Preprocess data first
adata = sp.preprocess(adata, scale_max=10, do_QC=True)

# Reduce dimensionality
adata = sp.reduce_dimensionality(adata, n_components=50)

# Compute neighborhood graph
import scanpy as sc
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

# Clustering with spatial weights
adata = sp.cluster(
    adata,
    spatial_weight=0.5,    # Weight for spatial neighbors
    resolution=1.0,        # Clustering resolution
    method='leiden'        # Clustering algorithm
)
```

**Parameters:**
- `spatial_weight`: Weight given to spatial neighbors (0-1)
- `resolution`: Clustering resolution (higher = more clusters)
- `method`: Clustering algorithm ('leiden', 'louvain')

## Data Preprocessing

### Quality Control and Normalization

```python
# Comprehensive preprocessing
adata = sp.preprocess(
    adata,
    scale_max=10,          # Maximum scaling value
    size_factor=None,      # Size factor for normalization
    do_QC=True            # Perform quality control
)
```

### Dimensionality Reduction

```python
# Reduce dimensionality using PCA
adata = sp.reduce_dimensionality(
    adata,
    n_components=50,       # Number of components
    method='pca'          # Reduction method
)
```

## Differential Expression Analysis

Find marker genes for each cluster:

```python
# Perform differential expression analysis
adata = sp.differential_expression(
    adata,
    groupby='leiden',      # Group by cluster
    method='wilcoxon'      # Statistical test
)

# Access results
markers = adata.uns['rank_genes_groups']

# Print top markers for each cluster
for cluster in adata.obs['leiden'].unique():
    top_genes = markers['names'][cluster][:5]
    print(f"Cluster {cluster}: {list(top_genes)}")
```

## Pathway Analysis

### Cluster Annotation

Annotate clusters based on known marker genes:

```python
# Define reference markers
reference_markers = {
    'T_cells': ['CD3', 'CD8', 'CD4'],
    'B_cells': ['CD20', 'CD19'],
    'Macrophages': ['CD68', 'CD11b'],
    'Endothelial': ['CD31', 'Vimentin']
}

# Annotate clusters
adata = sp.annotate_clusters(
    adata,
    groupby='leiden',
    reference_markers=reference_markers
)
```

### Pathway Enrichment

Analyze pathway enrichment for each cluster:

```python
# Perform pathway analysis
pathway_results = sp.analyze_pathways(
    adata,
    groupby='leiden',
    database='reactome'    # Pathway database
)

# Display results
for cluster in pathway_results.keys():
    print(f"\nCluster {cluster}:")
    top_pathways = pathway_results[cluster].head(3)
    for _, row in top_pathways.iterrows():
        print(f"  {row['pathway']}: p-value={row['p_value']:.3e}")
```

## Spatial Analysis

### Niche Analysis

Identify spatial patterns and cell neighborhoods:

```python
# Perform niche analysis
niche_results = sp.niche(
    adata,
    groupby='leiden',
    spatial_key='spatial',
    radius=50              # Analysis radius
)

# Display results
for niche_type, cells in niche_results.items():
    print(f"{niche_type}: {len(cells)} cells")
```

### Spatial Visualization

Create spatial plots of your data:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Extract spatial coordinates
coordinates = pd.DataFrame({
    'x': adata.obsm['spatial'][:, 0],
    'y': adata.obsm['spatial'][:, 1],
    'cluster': adata.obs['leiden']
})

# Create spatial plot
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=coordinates,
    x='x',
    y='y',
    hue='cluster',
    palette='tab20',
    s=20,
    alpha=0.8
)
plt.title('Spatial Distribution of Clusters')
plt.axis('equal')
plt.show()
```

## Working with AnnData

### Loading Data

```python
# Load single file
adata = sp.load_anndata(path="data.h5ad")['adata']

# Load multiple files
files = ["file1.h5ad", "file2.h5ad", "file3.h5ad"]
combined_data = sp.load_anndata(files=files)['adata']
```

### Data Structure

SPEX works with AnnData objects that contain:

- **`.X`**: Expression matrix (cells × genes)
- **`.obs`**: Cell annotations (clusters, coordinates, etc.)
- **`.var`**: Gene annotations
- **`.obsm`**: Cell embeddings (UMAP, spatial coordinates)
- **`.uns`**: Unstructured data (analysis results)

### Saving Results

```python
# Save processed data
adata.write("processed_data.h5ad")

# Save specific results
import pandas as pd

# Save cluster assignments
clusters_df = pd.DataFrame({
    'cell_id': adata.obs.index,
    'cluster': adata.obs['leiden'],
    'annotation': adata.obs['annotation']
})
clusters_df.to_csv("clusters.csv", index=False)
```

## Best Practices

### Parameter Selection

- **`knn`**: Start with 30-50 for PhenoGraph, adjust based on data size
- **`spatial_weight`**: Use 0.3-0.7 for spatial-aware clustering
- **`resolution`**: Start with 1.0, increase for more clusters
- **`radius`**: Choose based on your tissue type and cell density

### Quality Control

- Always perform QC before clustering
- Check for batch effects in multi-sample data
- Validate clustering with known marker genes
- Use multiple clustering methods for comparison

### Visualization

- Create UMAP plots for cluster visualization
- Use spatial plots to understand tissue organization
- Generate heatmaps for marker gene expression
- Create pathway enrichment plots

### Performance

- Use appropriate data types (float32 vs float64)
- Consider downsampling for very large datasets
- Save intermediate results for long workflows
- Use parallel processing when available

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `n_components` or use data subsampling
2. **Poor clustering**: Check data quality and try different parameters
3. **Spatial artifacts**: Verify spatial coordinates and scaling
4. **Missing markers**: Ensure channel names match your data

### Debugging Tips

```python
# Check data structure
print(f"Data shape: {adata.shape}")
print(f"Available annotations: {list(adata.obs.columns)}")
print(f"Available embeddings: {list(adata.obsm.keys())}")

# Verify spatial coordinates
if 'spatial' in adata.obsm:
    coords = adata.obsm['spatial']
    print(f"Spatial range: X({coords[:, 0].min():.1f}, {coords[:, 0].max():.1f})")
    print(f"Spatial range: Y({coords[:, 1].min():.1f}, {coords[:, 1].max():.1f})")
```

## Advanced Topics

### Custom Clustering

```python
# Custom clustering workflow
import scanpy as sc

# Custom preprocessing
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.pp.scale(adata)

# Custom clustering
sc.pp.neighbors(adata, n_neighbors=15)
sc.tl.leiden(adata, resolution=1.0)

# Add to SPEX workflow
adata = sp.differential_expression(adata, groupby='leiden')
```

### Batch Correction

```python
# Handle batch effects
adata.uns['batch_key'] = 'batch'  # Set batch key
adata = sp.preprocess(adata, scale_max=10, do_QC=True)

# Batch-aware clustering
adata = sp.cluster(adata, spatial_weight=0.3, resolution=1.0)
```

This comprehensive guide covers the main aspects of clustering and spatial transcriptomics analysis with SPEX. For detailed API documentation, see the [API Reference](api.md).
