# 🌐 Spatial Analysis API Reference

This document provides comprehensive API documentation for all spatial analysis functions in the SPEX library.

## Table of Contents

1. [Overview](#overview)
2. [CLQ Analysis](#clq-analysis)
3. [Niche Analysis](#niche-analysis)
4. [Differential Expression](#differential-expression)
5. [Pathway Analysis](#pathway-analysis)
6. [Spatial Autocorrelation](#spatial-autocorrelation)
7. [Complete Workflow](#complete-workflow)

## Overview

The SPEX spatial analysis module provides comprehensive tools for analyzing spatial relationships in transcriptomics data, including:

- **CLQ Analysis**: Co-Localization Quotient for cell type interactions
- **Niche Analysis**: Spatial microenvironment characterization
- **Differential Expression**: Spatial-aware gene expression analysis
- **Pathway Analysis**: Biological pathway enrichment in spatial context
- **Spatial Autocorrelation**: Spatial dependency analysis

## CLQ Analysis

### `CLQ_vec_numba()`

Calculate Co-Localization Quotient (CLQ) using Numba optimization.

```python
spex.CLQ_vec_numba(adata, clust_col='leiden', clust_uniq=None, radius=50, n_perms=1000)
```

**Parameters:**
- `adata` (AnnData): AnnData object containing spatial data and cluster labels
- `clust_col` (str, optional): Column name in adata.obs containing cluster/cell type labels. Default: 'leiden'
- `clust_uniq` (array-like, optional): Unique cluster labels. If None, will be inferred from adata.obs[clust_col]
- `radius` (float, optional): Radius for spatial neighbor calculation in microns. Default: 50
- `n_perms` (int, optional): Number of permutations for significance testing. Default: 1000

**Returns:**
- `adata_out` (AnnData): Updated AnnData object with CLQ results in adata.obsm and adata.uns
- `results` (dict): Dictionary containing CLQ results:
  - 'global_clq': Global CLQ matrix
  - 'permute_test': Permutation test p-values
  - 'local_clq': Local CLQ values for each cell
  - 'NCV': Neighborhood count vectors

**Notes:**
- CLQ > 1 indicates attraction between cell types
- CLQ < 1 indicates avoidance between cell types
- CLQ = 1 indicates random spatial distribution
- Results are stored in adata.obsm['local_clq'] and adata.obsm['NCV']
- Global results are stored in adata.uns['CLQ']

**Example:**
```python
import spex as sp
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

# Load your AnnData object with spatial coordinates
adata = sc.read_h5ad("your_data.h5ad")

# Ensure spatial coordinates are available
if 'spatial' not in adata.obsm:
    adata.obsm['spatial'] = adata.obs[['x_coordinate', 'y_coordinate']].to_numpy()

# Perform CLQ analysis
adata_out, results = sp.CLQ_vec_numba(
    adata,
    clust_col='leiden',      # Column with cluster labels
    radius=50,              # Analysis radius in microns
    n_perms=1000            # Number of permutations for significance testing
)

# Access results
print("Global CLQ matrix:")
print(results['global_clq'])

print("\nPermutation test results:")
print(results['permute_test'])

# Visualize global CLQ matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    results['global_clq'],
    annot=True,
    cmap='RdBu_r',
    center=1.0,
    square=True,
    fmt='.3f'
)
plt.title('Global Co-Localization Quotient')
plt.xlabel('Cell Type')
plt.ylabel('Cell Type')
plt.tight_layout()
plt.show()

# Local CLQ values for each cell
local_clq = adata_out.obsm['local_clq']
neighborhood_vectors = adata_out.obsm['NCV']

print(f"Local CLQ shape: {local_clq.shape}")
print(f"Neighborhood vectors shape: {neighborhood_vectors.shape}")
```

---

## Niche Analysis

### `niche()`

Analyze spatial niches and microenvironment characteristics.

```python
spex.niche(adata, cluster_key='leiden', spatial_weight=0.5, resolution=1.0, 
           min_cells=10, max_distance=100)
```

**Parameters:**
- `adata` (AnnData): AnnData object with spatial coordinates and cluster labels
- `cluster_key` (str): Key in adata.obs containing cluster labels
- `spatial_weight` (float): Weight for spatial proximity in niche definition
- `resolution` (float): Resolution parameter for niche detection
- `min_cells` (int): Minimum number of cells required for a niche
- `max_distance` (float): Maximum distance for spatial neighbor calculation

**Returns:**
- `AnnData`: Updated AnnData object with niche analysis results

**Example:**
```python
import spex as sp
import scanpy as sc

# Perform niche analysis
adata = sp.niche(
    adata,
    cluster_key='leiden',
    spatial_weight=0.5,
    resolution=1.0,
    min_cells=10,
    max_distance=100
)

# Access niche results
niche_labels = adata.obs['niche_labels']
niche_composition = adata.uns['niche_composition']

print(f"Found {len(niche_labels.unique())} niches")
print("Niche composition:")
for niche_id in niche_labels.unique():
    composition = niche_composition[niche_id]
    print(f"Niche {niche_id}: {composition}")

# Visualize niches
sc.pl.spatial(adata, color='niche_labels', size=50)
```

---

## Differential Expression

### `differential_expression()`

Perform differential expression analysis with spatial awareness.

```python
spex.differential_expression(adata, cluster_key='leiden', method='wilcoxon', 
                           n_genes=10, logfc_threshold=0.25, spatial_weight=0.0)
```

**Parameters:**
- `adata` (AnnData): AnnData object with clustering results
- `cluster_key` (str): Key in adata.obs containing cluster labels
- `method` (str): Statistical test method ('wilcoxon', 't-test', 'logreg')
- `n_genes` (int): Number of top genes to return per cluster
- `logfc_threshold` (float): Minimum log fold change threshold
- `spatial_weight` (float): Weight for spatial proximity in analysis

**Returns:**
- `pandas.DataFrame`: Differential expression results

**Example:**
```python
import spex as sp

# Find marker genes with spatial awareness
marker_genes = sp.differential_expression(
    adata, 
    cluster_key='leiden',
    method='wilcoxon',
    n_genes=20,
    logfc_threshold=0.5,
    spatial_weight=0.3
)

# Display top markers for each cluster
for cluster in adata.obs['leiden'].unique():
    cluster_markers = marker_genes[marker_genes['cluster'] == cluster]
    print(f"\nCluster {cluster} markers:")
    print(cluster_markers.head(5)[['gene', 'logfoldchanges', 'pvals_adj']])

# Visualize spatial expression of top markers
top_markers = marker_genes.groupby('cluster').head(3)['gene'].tolist()
sc.pl.spatial(adata, color=top_markers[:6], ncols=3, size=50)
```

---

## Pathway Analysis

### `analyze_pathways()`

Perform pathway enrichment analysis on spatial data.

```python
spex.analyze_pathways(adata, cluster_key='leiden', marker_genes=None, 
                     database='GO_Biological_Process_2021', p_threshold=0.05,
                     spatial_context=True)
```

**Parameters:**
- `adata` (AnnData): AnnData object with clustering results
- `cluster_key` (str): Key in adata.obs containing cluster labels
- `marker_genes` (dict, optional): Dictionary of marker genes per cluster
- `database` (str): Pathway database to use
- `p_threshold` (float): P-value threshold for significance
- `spatial_context` (bool): Whether to consider spatial context in analysis

**Returns:**
- `pandas.DataFrame`: Pathway enrichment results

**Example:**
```python
import spex as sp

# Analyze pathways with spatial context
pathway_results = sp.analyze_pathways(
    adata,
    cluster_key='leiden',
    database='GO_Biological_Process_2021',
    p_threshold=0.01,
    spatial_context=True
)

# Display top pathways for each cluster
for cluster in adata.obs['leiden'].unique():
    cluster_pathways = pathway_results[pathway_results['cluster'] == cluster]
    print(f"\nCluster {cluster} pathways:")
    print(cluster_pathways.head(3)[['pathway', 'p_value', 'enrichment_score']])

# Visualize pathway enrichment
import matplotlib.pyplot as plt
import seaborn as sns

# Create pathway heatmap
pathway_matrix = pathway_results.pivot_table(
    index='pathway', 
    columns='cluster', 
    values='enrichment_score'
).fillna(0)

plt.figure(figsize=(12, 8))
sns.heatmap(pathway_matrix, cmap='RdBu_r', center=0, annot=True, fmt='.2f')
plt.title('Pathway Enrichment by Cluster')
plt.tight_layout()
plt.show()
```

---

### `annotate_clusters()`

Automatically annotate clusters based on spatial context and pathway analysis.

```python
spex.annotate_clusters(adata, cluster_key='leiden', marker_genes=None, 
                      pathway_results=None, method='spatial_aware')
```

**Parameters:**
- `adata` (AnnData): AnnData object with clustering results
- `cluster_key` (str): Key in adata.obs containing cluster labels
- `marker_genes` (dict, optional): Dictionary of marker genes per cluster
- `pathway_results` (pandas.DataFrame, optional): Pathway analysis results
- `method` (str): Annotation method ('spatial_aware', 'marker_based', 'hybrid')

**Returns:**
- `AnnData`: Updated AnnData object with annotations in adata.obs

**Example:**
```python
import spex as sp

# Annotate clusters with spatial awareness
adata = sp.annotate_clusters(
    adata,
    cluster_key='leiden',
    method='spatial_aware'
)

# View annotations
print("Cluster annotations:")
for cluster in adata.obs['leiden'].unique():
    annotation = adata.obs[adata.obs['leiden'] == cluster]['cell_type'].iloc[0]
    print(f"Cluster {cluster}: {annotation}")

# Visualize annotated clusters
sc.pl.spatial(adata, color='cell_type', size=50)
```

---

## Spatial Autocorrelation

### Spatial Autocorrelation Analysis

SPEX provides tools for analyzing spatial autocorrelation in gene expression:

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

def moran_i(adata, gene, spatial_key='spatial', weight_type='inverse_distance'):
    """
    Calculate Moran's I statistic for spatial autocorrelation.
    
    Parameters:
    - adata: AnnData object
    - gene: Gene name to analyze
    - spatial_key: Key for spatial coordinates
    - weight_type: Type of spatial weights ('inverse_distance', 'binary')
    
    Returns:
    - moran_i: Moran's I statistic
    - p_value: P-value for significance test
    """
    # Get spatial coordinates and gene expression
    coords = adata.obsm[spatial_key]
    expression = adata[:, gene].X.flatten()
    
    # Calculate spatial weights
    distances = squareform(pdist(coords))
    
    if weight_type == 'inverse_distance':
        weights = 1 / (distances + 1e-10)  # Add small constant to avoid division by zero
    elif weight_type == 'binary':
        threshold = np.median(distances[distances > 0])
        weights = (distances <= threshold).astype(float)
    
    # Set diagonal to zero
    np.fill_diagonal(weights, 0)
    
    # Calculate Moran's I
    n = len(expression)
    mean_expr = np.mean(expression)
    variance = np.var(expression)
    
    numerator = 0
    denominator = 0
    
    for i in range(n):
        for j in range(n):
            if i != j:
                numerator += weights[i, j] * (expression[i] - mean_expr) * (expression[j] - mean_expr)
                denominator += weights[i, j]
    
    moran_i = (n / (2 * denominator)) * (numerator / variance)
    
    # Calculate p-value (simplified)
    p_value = 0.05  # Placeholder - would need proper permutation test
    
    return moran_i, p_value

# Example usage
genes_to_test = ['Gene1', 'Gene2', 'Gene3']
spatial_autocorr_results = {}

for gene in genes_to_test:
    if gene in adata.var_names:
        moran_i_val, p_val = moran_i(adata, gene)
        spatial_autocorr_results[gene] = {'moran_i': moran_i_val, 'p_value': p_val}

# Display results
print("Spatial autocorrelation results:")
for gene, results in spatial_autocorr_results.items():
    print(f"{gene}: Moran's I = {results['moran_i']:.3f}, p = {results['p_value']:.3f}")
```

---

## Complete Workflow

Here's a complete spatial analysis workflow example:

```python
import spex as sp
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and preprocess data
adata = sc.read_h5ad("path/to/your/spatial_data.h5ad")

# Ensure spatial coordinates are available
if 'spatial' not in adata.obsm:
    adata.obsm['spatial'] = adata.obs[['x_coordinate', 'y_coordinate']].to_numpy()

# Basic preprocessing
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)

# 2. Clustering (if not already done)
sc.pp.highly_variable_features(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata_hvg = adata[:, adata.var.highly_variable].copy()
sc.tl.pca(adata_hvg, use_highly_variable=True)
sc.pp.neighbors(adata_hvg, use_rep='X_pca')
adata_hvg = sp.cluster(adata_hvg, method='leiden', resolution=0.5)

# 3. CLQ Analysis
print("Performing CLQ analysis...")
adata_clq, clq_results = sp.CLQ_vec_numba(
    adata_hvg,
    clust_col='leiden',
    radius=50,
    n_perms=1000
)

# 4. Niche Analysis
print("Performing niche analysis...")
adata_niche = sp.niche(
    adata_clq,
    cluster_key='leiden',
    spatial_weight=0.5,
    resolution=1.0
)

# 5. Differential Expression
print("Performing differential expression analysis...")
marker_genes = sp.differential_expression(
    adata_niche,
    cluster_key='leiden',
    method='wilcoxon',
    n_genes=20,
    spatial_weight=0.3
)

# 6. Pathway Analysis
print("Performing pathway analysis...")
pathway_results = sp.analyze_pathways(
    adata_niche,
    cluster_key='leiden',
    database='GO_Biological_Process_2021',
    spatial_context=True
)

# 7. Cluster Annotation
print("Annotating clusters...")
adata_final = sp.annotate_clusters(
    adata_niche,
    cluster_key='leiden',
    method='spatial_aware'
)

# 8. Visualize Results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Spatial plot colored by clusters
sc.pl.spatial(adata_final, color='leiden', ax=axes[0, 0], show=False)
axes[0, 0].set_title('Clusters')

# Spatial plot colored by cell types
sc.pl.spatial(adata_final, color='cell_type', ax=axes[0, 1], show=False)
axes[0, 1].set_title('Cell Types')

# Spatial plot colored by niches
sc.pl.spatial(adata_final, color='niche_labels', ax=axes[0, 2], show=False)
axes[0, 2].set_title('Niches')

# CLQ heatmap
sns.heatmap(clq_results['global_clq'], annot=True, cmap='RdBu_r', center=1.0, 
            square=True, fmt='.3f', ax=axes[1, 0])
axes[1, 0].set_title('Global CLQ Matrix')

# Top marker genes heatmap
top_genes = []
for cluster in adata_final.obs['leiden'].unique():
    cluster_markers = marker_genes[marker_genes['cluster'] == cluster]
    top_genes.extend(cluster_markers.head(3)['gene'].tolist())

sc.pl.heatmap(adata_final, top_genes, groupby='leiden', ax=axes[1, 1], show=False)
axes[1, 1].set_title('Marker Genes')

# Pathway enrichment heatmap
pathway_matrix = pathway_results.pivot_table(
    index='pathway', columns='cluster', values='enrichment_score'
).fillna(0)
sns.heatmap(pathway_matrix.head(10), cmap='RdBu_r', center=0, ax=axes[1, 2])
axes[1, 2].set_title('Pathway Enrichment')

plt.tight_layout()
plt.show()

# 9. Summary Statistics
print("\n=== SPATIAL ANALYSIS SUMMARY ===")
print(f"Dataset: {adata_final.n_obs} cells, {adata_final.n_vars} genes")
print(f"Clusters: {len(adata_final.obs['leiden'].unique())}")
print(f"Niches: {len(adata_final.obs['niche_labels'].unique())}")
print(f"Cell types: {len(adata_final.obs['cell_type'].unique())}")
print(f"Marker genes identified: {len(marker_genes)}")
print(f"Pathways analyzed: {len(pathway_results)}")

# CLQ interpretation
print("\n=== CLQ INTERPRETATION ===")
global_clq = clq_results['global_clq']
for i in range(len(global_clq)):
    for j in range(i+1, len(global_clq)):
        clq_val = global_clq[i, j]
        if clq_val > 1.5:
            print(f"Strong attraction between cell types {i} and {j} (CLQ = {clq_val:.3f})")
        elif clq_val < 0.5:
            print(f"Strong avoidance between cell types {i} and {j} (CLQ = {clq_val:.3f})")

print("\n✅ Spatial analysis complete!")
```

---

## Troubleshooting

### Common Issues and Solutions

**1. CLQ analysis fails or produces errors**
- Check spatial coordinates are properly formatted
- Ensure cluster labels are available in adata.obs
- Verify sufficient cells per cluster (minimum 10 recommended)
- Adjust radius parameter based on your data scale

**2. No significant spatial relationships found**
- Increase radius for neighbor calculation
- Check data quality and preprocessing
- Verify spatial coordinates are in correct units
- Consider different clustering resolution

**3. Memory issues with large datasets**
- Reduce number of permutations in CLQ analysis
- Process data in spatial chunks
- Use subset of genes for analysis
- Optimize spatial neighbor calculation

**4. Niche analysis produces too many/few niches**
- Adjust spatial_weight parameter
- Modify resolution parameter
- Change min_cells threshold
- Check spatial distribution of cells

**5. Pathway analysis returns no significant results**
- Lower p_threshold
- Use different pathway database
- Check marker gene quality
- Verify gene annotation

---

**Next Steps:**
- [Complete Pipeline Example](../examples/complete-pipeline.md)
- [Troubleshooting Guide](../reference/troubleshooting.md)
- [Installation Guide](../getting-started/installation.md)
