# 🔬 Clustering API Reference

This document provides comprehensive API documentation for all clustering functions in the SPEX library.

## Table of Contents

1. [Overview](#overview)
2. [Basic Clustering](#basic-clustering)
3. [Phenograph Clustering](#phenograph-clustering)
4. [Spatial Clustering](#spatial-clustering)
5. [Differential Expression](#differential-expression)
6. [Pathway Analysis](#pathway-analysis)
7. [Cluster Validation](#cluster-validation)
8. [Complete Workflow](#complete-workflow)

## Overview

The SPEX clustering module provides comprehensive tools for analyzing spatial omics data (proteomics and transcriptomics), including:

- **Traditional Clustering**: Leiden, Louvain algorithms
- **Spatial-Aware Clustering**: Phenograph with spatial weights
- **Marker Gene Analysis**: Identification of cluster-specific genes
- **Pathway Analysis**: Biological pathway enrichment
- **Cluster Validation**: Quality assessment metrics
- **Cell Type Annotation**: Automated and manual annotation

## Basic Clustering

### `cluster()`

Perform clustering with optional spatial weights.

```python
spex.cluster(adata, spatial_weight=0.0, resolution=1.0, method='leiden')
```

**Parameters:**
- `adata` (AnnData): AnnData object after preprocessing and dimensionality reduction
- `spatial_weight` (float, optional): Weight for spatial neighbors in adjacency calculation. If > 0, spatial neighbors are forced to be close
- `resolution` (float, optional): Resolution of the modularity cost function. Lower values result in fewer clusters, higher values in more clusters
- `method` (str, optional): Clustering algorithm to use. Options: 'leiden', 'louvain'

**Returns:**
- `AnnData`: Updated AnnData object with clustering results in adata.obs

**Requirements:**
- Requires 'connectivities' in adata.obsp
- For spatial clustering, requires 'spatial_connectivities' in adata.obsp
- Results are stored in adata.obs with the method name as column

**Example:**
```python
import spex as sp
import scanpy as sc

# Ensure neighbors are computed
if 'neighbors' not in adata.uns:
    sc.pp.neighbors(adata, use_rep='X_pca')

# Basic Leiden clustering
adata = sp.cluster(adata, method='leiden', resolution=0.5)

# Spatial-aware clustering
adata = sp.cluster(adata, method='leiden', resolution=0.5, spatial_weight=0.3)

print(f"Found {len(adata.obs['leiden'].unique())} clusters")
```

---

## Phenograph Clustering

### `phenograph_cluster()`

Perform PhenoGraph clustering with advanced preprocessing options.

```python
spex.phenograph_cluster(adata, channel_names, knn=30, transformation='arcsin', 
                       scaling='z-score', cofactor=5.0, umap_min_dist=0.5)
```

**Parameters:**
- `adata` (AnnData): AnnData object containing single-cell features
- `channel_names` (List[str]): List of channel names (e.g., ['CD3', 'CD8']) to be used for clustering
- `knn` (int): Number of neighbors for graph construction (used in PhenoGraph and UMAP)
- `transformation` (str): Data transformation to apply before clustering. Options: 'arcsin', 'log', 'none'
- `scaling` (str): Feature scaling strategy. Options: 'z-score', 'winsorize', 'none'
- `cofactor` (float): Cofactor for arcsinh transformation (if used)
- `umap_min_dist` (float): Minimum distance parameter for UMAP projection

**Returns:**
- `AnnData`: Updated AnnData object with:
  - `adata.obs['cluster_phenograph']`: cluster labels (as strings)
  - `adata.obsm['X_umap']`: 2D UMAP coordinates

**Example:**
```python
import spex as sp

# Perform PhenoGraph clustering
adata = sp.phenograph_cluster(
    adata,
    channel_names=['CD3', 'CD8', 'CD4', 'CD19'],
    knn=30,
    transformation='arcsin',
    scaling='z-score',
    cofactor=5.0
)

print(f"Found {len(adata.obs['cluster_phenograph'].unique())} clusters")

# Visualize results
import matplotlib.pyplot as plt
plt.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1], 
           c=adata.obs['cluster_phenograph'].astype('category').cat.codes)
plt.title('PhenoGraph Clustering Results')
plt.show()
```

**Transformation Options:**
- `'arcsin'`: Arcsinh transformation with cofactor (recommended for cytometry data)
- `'log'`: Log10 transformation
- `'none'`: No transformation

**Scaling Options:**
- `'z-score'`: Standardization to zero mean and unit variance
- `'winsorize'`: Winsorization to handle outliers
- `'none'`: No scaling

---

## Spatial Clustering

### Spatial-Aware Clustering with Weights

SPEX supports spatial-aware clustering by incorporating spatial neighbor information:

```python
# Compute spatial neighbors
import squidpy as sq
sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=10)

# Perform spatial clustering
adata = sp.cluster(adata, method='leiden', resolution=0.5, spatial_weight=0.3)
```

**Spatial Weight Guidelines:**
- `spatial_weight = 0.0`: Traditional clustering (no spatial constraints)
- `spatial_weight = 0.1-0.3`: Mild spatial influence
- `spatial_weight = 0.5-1.0`: Strong spatial influence
- `spatial_weight > 1.0`: Very strong spatial constraints

---

## Differential Expression

### `differential_expression()`

Identify differentially expressed genes between clusters.

```python
spex.differential_expression(adata, cluster_key='leiden', method='wilcoxon', 
                           n_genes=10, logfc_threshold=0.25)
```

**Parameters:**
- `adata` (AnnData): AnnData object with clustering results
- `cluster_key` (str): Key in adata.obs containing cluster labels
- `method` (str): Statistical test method ('wilcoxon', 't-test', 'logreg')
- `n_genes` (int): Number of top genes to return per cluster
- `logfc_threshold` (float): Minimum log fold change threshold

**Returns:**
- `pandas.DataFrame`: Differential expression results

**Example:**
```python
# Find marker genes
marker_genes = sp.differential_expression(
    adata, 
    cluster_key='leiden',
    method='wilcoxon',
    n_genes=20,
    logfc_threshold=0.5
)

# Display top markers for each cluster
for cluster in adata.obs['leiden'].unique():
    cluster_markers = marker_genes[marker_genes['cluster'] == cluster]
    print(f"\nCluster {cluster} markers:")
    print(cluster_markers.head(5)[['gene', 'logfoldchanges', 'pvals_adj']])
```

---

## Pathway Analysis

### `analyze_pathways()`

Perform pathway enrichment analysis on cluster marker genes.

```python
spex.analyze_pathways(adata, cluster_key='leiden', marker_genes=None, 
                     database='GO_Biological_Process_2021', p_threshold=0.05)
```

**Parameters:**
- `adata` (AnnData): AnnData object with clustering results
- `cluster_key` (str): Key in adata.obs containing cluster labels
- `marker_genes` (dict, optional): Dictionary of marker genes per cluster
- `database` (str): Pathway database to use
- `p_threshold` (float): P-value threshold for significance

**Returns:**
- `pandas.DataFrame`: Pathway enrichment results

**Example:**
```python
# Analyze pathways
pathway_results = sp.analyze_pathways(
    adata,
    cluster_key='leiden',
    database='GO_Biological_Process_2021',
    p_threshold=0.01
)

# Display top pathways for each cluster
for cluster in adata.obs['leiden'].unique():
    cluster_pathways = pathway_results[pathway_results['cluster'] == cluster]
    print(f"\nCluster {cluster} pathways:")
    print(cluster_pathways.head(3)[['pathway', 'p_value', 'enrichment_score']])
```

---

### `annotate_clusters()`

Automatically annotate clusters based on marker genes and pathway analysis.

```python
spex.annotate_clusters(adata, cluster_key='leiden', marker_genes=None, 
                      pathway_results=None, method='enrichment')
```

**Parameters:**
- `adata` (AnnData): AnnData object with clustering results
- `cluster_key` (str): Key in adata.obs containing cluster labels
- `marker_genes` (dict, optional): Dictionary of marker genes per cluster
- `pathway_results` (pandas.DataFrame, optional): Pathway analysis results
- `method` (str): Annotation method ('enrichment', 'marker_based', 'hybrid')

**Returns:**
- `AnnData`: Updated AnnData object with annotations in adata.obs

**Example:**
```python
# Annotate clusters
adata = sp.annotate_clusters(
    adata,
    cluster_key='leiden',
    method='hybrid'
)

# View annotations
print("Cluster annotations:")
for cluster in adata.obs['leiden'].unique():
    annotation = adata.obs[adata.obs['leiden'] == cluster]['cell_type'].iloc[0]
    print(f"Cluster {cluster}: {annotation}")
```

---

## Cluster Validation

### Validation Metrics

SPEX provides several metrics to validate clustering quality:

```python
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def validate_clustering(adata, cluster_key='leiden', embedding='X_pca'):
    """
    Calculate clustering validation metrics.
    """
    clusters = adata.obs[cluster_key].values
    X = adata.obsm[embedding]
    
    # Silhouette score (higher is better)
    silhouette = silhouette_score(X, clusters)
    
    # Calinski-Harabasz score (higher is better)
    calinski = calinski_harabasz_score(X, clusters)
    
    # Davies-Bouldin score (lower is better)
    davies = davies_bouldin_score(X, clusters)
    
    return {
        'silhouette_score': silhouette,
        'calinski_harabasz_score': calinski,
        'davies_bouldin_score': davies,
        'n_clusters': len(np.unique(clusters))
    }

# Validate clustering
validation_results = validate_clustering(adata, cluster_key='leiden')
print("Clustering validation results:")
for metric, value in validation_results.items():
    print(f"{metric}: {value:.3f}")
```

**Interpretation:**
- **Silhouette Score**: Range [-1, 1], higher is better
- **Calinski-Harabasz Score**: Higher is better
- **Davies-Bouldin Score**: Lower is better

---

## Complete Workflow

Here's a complete clustering workflow example:

```python
import spex as sp
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and preprocess data
adata = sc.read_h5ad("path/to/your/data.h5ad")

# Basic preprocessing
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)

# Feature selection
sc.pp.highly_variable_features(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata_hvg = adata[:, adata.var.highly_variable].copy()

# Dimensionality reduction
sc.tl.pca(adata_hvg, use_highly_variable=True)
sc.pp.neighbors(adata_hvg, use_rep='X_pca')

# 2. Perform clustering
# Option A: Traditional clustering
adata_hvg = sp.cluster(adata_hvg, method='leiden', resolution=0.5)

# Option B: Spatial-aware clustering
# sq.gr.spatial_neighbors(adata_hvg, coord_type="generic", n_neighs=10)
# adata_hvg = sp.cluster(adata_hvg, method='leiden', resolution=0.5, spatial_weight=0.3)

# Option C: PhenoGraph clustering
# adata_hvg = sp.phenograph_cluster(adata_hvg, channel_names=['CD3', 'CD8', 'CD4'])

# 3. Find marker genes
marker_genes = sp.differential_expression(
    adata_hvg, 
    cluster_key='leiden',
    method='wilcoxon',
    n_genes=20
)

# 4. Pathway analysis
pathway_results = sp.analyze_pathways(
    adata_hvg,
    cluster_key='leiden',
    database='GO_Biological_Process_2021'
)

# 5. Annotate clusters
adata_hvg = sp.annotate_clusters(adata_hvg, cluster_key='leiden')

# 6. Visualize results
sc.tl.umap(adata_hvg, use_rep='X_pca')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# UMAP colored by clusters
sc.pl.umap(adata_hvg, color='leiden', ax=axes[0], show=False)
axes[0].set_title('Clusters')

# UMAP colored by cell type
sc.pl.umap(adata_hvg, color='cell_type', ax=axes[1], show=False)
axes[1].set_title('Cell Types')

# Top marker genes heatmap
top_genes = []
for cluster in adata_hvg.obs['leiden'].unique():
    cluster_markers = marker_genes[marker_genes['cluster'] == cluster]
    top_genes.extend(cluster_markers.head(3)['gene'].tolist())

sc.pl.heatmap(adata_hvg, top_genes, groupby='leiden', ax=axes[2], show=False)
axes[2].set_title('Marker Genes')

plt.tight_layout()
plt.show()

# 7. Validation
validation_results = validate_clustering(adata_hvg, cluster_key='leiden')
print(f"\nClustering validation:")
print(f"Silhouette score: {validation_results['silhouette_score']:.3f}")
print(f"Number of clusters: {validation_results['n_clusters']}")

print(f"\nAnalysis complete!")
print(f"Found {len(adata_hvg.obs['leiden'].unique())} clusters")
print(f"Identified {len(marker_genes)} marker genes")
print(f"Analyzed {len(pathway_results)} pathways")
```

---

## Troubleshooting

### Common Issues and Solutions

**1. Clustering produces too many/few clusters**
- Adjust `resolution` parameter (higher = more clusters)
- Try different clustering methods
- Check data preprocessing quality

**2. Poor cluster separation**
- Improve data preprocessing
- Use different dimensionality reduction
- Consider spatial weights for spatial data

**3. No marker genes found**
- Lower `logfc_threshold`
- Use different statistical test
- Check data quality and normalization

**4. Memory issues with large datasets**
- Use subset of highly variable genes
- Reduce dimensionality before clustering
- Process data in batches

**5. PhenoGraph clustering fails**
- Check channel names match data
- Verify data transformation parameters
- Ensure sufficient data quality

---

**Next Steps:**
- [Spatial Analysis](../tutorials/spatial-analysis.md)
- [Complete Pipeline Example](../examples/complete-pipeline.md)
- [Troubleshooting Guide](../reference/troubleshooting.md)
