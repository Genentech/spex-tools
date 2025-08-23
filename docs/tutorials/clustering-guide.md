# 🎯 Clustering Guide Tutorial

## Overview

This tutorial covers clustering analysis for cell type identification in spatial transcriptomics data. You'll learn how to perform clustering, identify cell types, and analyze spatial relationships between different cell populations.

## Prerequisites

```python
import spex as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')
```

## Data Preparation

### Load and Preprocess Data

```python
# Load processed data (from segmentation tutorial)
# Assuming we have AnnData object from previous steps
adata = sc.read('segmentation_results.h5ad')

print(f"Dataset shape: {adata.X.shape}")
print(f"Features: {list(adata.var_names)}")
print(f"Cells: {adata.n_obs}")

# Check data quality
print(f"Missing values: {np.isnan(adata.X).sum()}")
print(f"Zero values: {(adata.X == 0).sum()}")
```

### Data Normalization

```python
# Normalize data
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Scale data
sc.pp.scale(adata, max_value=10)

print("Data normalized and scaled")
print(f"Expression range: {adata.X.min():.3f} - {adata.X.max():.3f}")
```

### Feature Selection

```python
# Find highly variable features
sc.pp.highly_variable_features(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
print(f"Highly variable features: {sum(adata.var.highly_variable)}")

# Plot highly variable features
sc.pl.highly_variable_features(adata)
plt.show()

# Subset to highly variable features
adata_hvg = adata[:, adata.var.highly_variable].copy()
print(f"Subset shape: {adata_hvg.X.shape}")
```

## Dimensionality Reduction

### Principal Component Analysis (PCA)

```python
# Perform PCA
sc.tl.pca(adata_hvg, use_highly_variable=True)

# Plot explained variance
sc.pl.pca_variance_ratio(adata_hvg, log=True)
plt.show()

# Plot first few PCs
sc.pl.pca(adata_hvg, color='area', components=[1, 2])
plt.show()

# Determine number of PCs to use
# Look for elbow in explained variance
cumsum = np.cumsum(adata_hvg.uns['pca']['variance_ratio'])
n_pcs = np.argmax(cumsum > 0.8) + 1
print(f"Using {n_pcs} PCs (80% variance explained)")
```

### UMAP and t-SNE

```python
# Compute neighborhood graph
sc.pp.neighbors(adata_hvg, n_neighbors=10, n_pcs=n_pcs)

# UMAP
sc.tl.umap(adata_hvg)
sc.pl.umap(adata_hvg, color='area')
plt.show()

# t-SNE (alternative)
sc.tl.tsne(adata_hvg, n_pcs=n_pcs)
sc.pl.tsne(adata_hvg, color='area')
plt.show()
```

## Clustering Methods

### 1. Leiden Clustering

```python
# Leiden clustering
sc.tl.leiden(adata_hvg, resolution=0.5)
print(f"Number of clusters: {len(adata_hvg.obs['leiden'].unique())}")

# Visualize clusters
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sc.pl.umap(adata_hvg, color='leiden', ax=axes[0], show=False)
axes[0].set_title('Leiden Clusters (UMAP)')

sc.pl.tsne(adata_hvg, color='leiden', ax=axes[1], show=False)
axes[1].set_title('Leiden Clusters (t-SNE)')

sc.pl.pca(adata_hvg, color='leiden', ax=axes[2], show=False)
axes[2].set_title('Leiden Clusters (PCA)')

plt.tight_layout()
plt.show()
```

### 2. Louvain Clustering

```python
# Louvain clustering
sc.tl.louvain(adata_hvg, resolution=0.5)
print(f"Number of clusters: {len(adata_hvg.obs['louvain'].unique())}")

# Compare clustering methods
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sc.pl.umap(adata_hvg, color='leiden', ax=axes[0, 0], show=False)
axes[0, 0].set_title('Leiden Clusters')

sc.pl.umap(adata_hvg, color='louvain', ax=axes[0, 1], show=False)
axes[0, 1].set_title('Louvain Clusters')

sc.pl.tsne(adata_hvg, color='leiden', ax=axes[1, 0], show=False)
axes[1, 0].set_title('Leiden Clusters (t-SNE)')

sc.pl.tsne(adata_hvg, color='louvain', ax=axes[1, 1], show=False)
axes[1, 1].set_title('Louvain Clusters (t-SNE)')

plt.tight_layout()
plt.show()
```

### 3. K-Means Clustering

```python
# K-means clustering
from sklearn.cluster import KMeans

# Determine optimal number of clusters using elbow method
inertias = []
K_range = range(2, 15)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(adata_hvg.obsm['X_pca'][:, :n_pcs])
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Choose optimal k (example: k=6)
optimal_k = 6
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
adata_hvg.obs['kmeans'] = kmeans.fit_predict(adata_hvg.obsm['X_pca'][:, :n_pcs])

# Visualize K-means clusters
sc.pl.umap(adata_hvg, color='kmeans')
plt.show()
```

### 4. DBSCAN Clustering

```python
# DBSCAN clustering
from sklearn.cluster import DBSCAN

# Try different parameters
eps_values = [0.5, 1.0, 1.5, 2.0]
min_samples_values = [5, 10, 15]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, (eps, min_samples) in enumerate([(0.5, 5), (1.0, 10), (1.5, 15), (2.0, 10)]):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(adata_hvg.obsm['X_pca'][:, :n_pcs])
    
    axes[i].scatter(adata_hvg.obsm['X_umap'][:, 0], 
                   adata_hvg.obsm['X_umap'][:, 1], 
                   c=clusters, cmap='tab10', alpha=0.7)
    axes[i].set_title(f'DBSCAN (eps={eps}, min_samples={min_samples})')
    axes[i].set_xlabel('UMAP1')
    axes[i].set_ylabel('UMAP2')

plt.tight_layout()
plt.show()

# Choose best parameters and apply
best_eps, best_min_samples = 1.0, 10
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
adata_hvg.obs['dbscan'] = dbscan.fit_predict(adata_hvg.obsm['X_pca'][:, :n_pcs])

print(f"DBSCAN clusters: {len(set(adata_hvg.obs['dbscan'])) - (1 if -1 in adata_hvg.obs['dbscan'] else 0)}")
print(f"Noise points: {sum(adata_hvg.obs['dbscan'] == -1)}")
```

## Cluster Analysis

### Cluster Statistics

```python
# Analyze cluster characteristics
def analyze_clusters(adata, cluster_key='leiden'):
    """Analyze cluster characteristics"""
    
    cluster_stats = {}
    
    for cluster in adata.obs[cluster_key].unique():
        mask = adata.obs[cluster_key] == cluster
        cluster_cells = adata[mask]
        
        stats = {
            'n_cells': len(cluster_cells),
            'mean_area': cluster_cells.obs['area'].mean(),
            'std_area': cluster_cells.obs['area'].std(),
            'mean_shape_factor': cluster_cells.obs['shape_factor'].mean(),
            'mean_expression': cluster_cells.X.mean(),
            'std_expression': cluster_cells.X.std()
        }
        
        cluster_stats[cluster] = stats
    
    return pd.DataFrame(cluster_stats).T

# Analyze different clustering methods
clustering_methods = ['leiden', 'louvain', 'kmeans', 'dbscan']
cluster_analyses = {}

for method in clustering_methods:
    if method in adata_hvg.obs.columns:
        cluster_analyses[method] = analyze_clusters(adata_hvg, method)

# Display results
for method, analysis in cluster_analyses.items():
    print(f"\n{method.upper()} Clustering Analysis:")
    print(analysis.round(3))
```

### Marker Feature Identification

```python
# Find marker features for each cluster
sc.tl.rank_genes_groups(adata_hvg, 'leiden', method='wilcoxon')

# Plot top markers
sc.pl.rank_genes_groups(adata_hvg, n_genes=5, sharey=False)
plt.show()

# Get marker features
markers = sc.get.rank_genes_groups_df(adata_hvg, group=None)
print("Top markers for each cluster:")
for cluster in adata_hvg.obs['leiden'].unique():
    cluster_markers = markers[markers['group'] == cluster].head(5)
    print(f"\nCluster {cluster}:")
    print(cluster_markers[['names', 'scores', 'pvals_adj']].to_string(index=False))
```

### Cluster Visualization

```python
# Plot expression of top markers
top_markers_per_cluster = []
for cluster in adata_hvg.obs['leiden'].unique():
    cluster_markers = markers[markers['group'] == cluster]
    top_markers_per_cluster.extend(cluster_markers.head(3)['names'].tolist())

# Remove duplicates
top_markers_per_cluster = list(set(top_markers_per_cluster))

# Plot heatmap
sc.pl.heatmap(adata_hvg, top_markers_per_cluster, groupby='leiden', 
              use_raw=True, show_gene_labels=True)
plt.show()

# Plot violin plots
sc.pl.violin(adata_hvg, top_markers_per_cluster[:6], groupby='leiden')
plt.show()
```

## Spatial Analysis of Clusters

### Spatial Distribution

```python
# Analyze spatial distribution of clusters
if 'spatial' in adata_hvg.obsm:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Leiden clusters spatial
    sc.pl.spatial(adata_hvg, color='leiden', ax=axes[0, 0], show=False)
    axes[0, 0].set_title('Leiden Clusters (Spatial)')
    
    # Louvain clusters spatial
    sc.pl.spatial(adata_hvg, color='louvain', ax=axes[0, 1], show=False)
    axes[0, 1].set_title('Louvain Clusters (Spatial)')
    
    # K-means clusters spatial
    sc.pl.spatial(adata_hvg, color='kmeans', ax=axes[1, 0], show=False)
    axes[1, 0].set_title('K-means Clusters (Spatial)')
    
    # DBSCAN clusters spatial
    sc.pl.spatial(adata_hvg, color='dbscan', ax=axes[1, 1], show=False)
    axes[1, 1].set_title('DBSCAN Clusters (Spatial)')
    
    plt.tight_layout()
    plt.show()
```

### Spatial Autocorrelation

```python
# Calculate spatial autocorrelation for clusters
def calculate_spatial_autocorrelation(positions, cluster_labels):
    """Calculate Moran's I for spatial autocorrelation of clusters"""
    
    from scipy.spatial.distance import pdist, squareform
    
    if len(positions) < 2:
        return 0
    
    # Calculate distances
    dist_matrix = squareform(pdist(positions))
    
    # Calculate weights (inverse distance)
    weights = 1 / (dist_matrix + 1e-10)
    np.fill_diagonal(weights, 0)
    
    # Convert cluster labels to numeric
    unique_clusters = np.unique(cluster_labels)
    cluster_numeric = np.array([np.where(unique_clusters == label)[0][0] 
                               for label in cluster_labels])
    
    # Calculate Moran's I
    n = len(cluster_numeric)
    mean_val = np.mean(cluster_numeric)
    variance = np.var(cluster_numeric)
    
    if variance == 0:
        return 0
    
    numerator = 0
    denominator = 0
    
    for i in range(n):
        for j in range(n):
            if i != j:
                numerator += weights[i, j] * (cluster_numeric[i] - mean_val) * (cluster_numeric[j] - mean_val)
                denominator += weights[i, j]
    
    if denominator == 0:
        return 0
    
    moran_i = (n / (2 * denominator)) * (numerator / variance)
    return moran_i

# Calculate for each clustering method
if 'spatial' in adata_hvg.obsm:
    spatial_autocorr = {}
    positions = adata_hvg.obsm['spatial']
    
    for method in clustering_methods:
        if method in adata_hvg.obs.columns:
            moran_i = calculate_spatial_autocorrelation(positions, adata_hvg.obs[method])
            spatial_autocorr[method] = moran_i
    
    print("Spatial Autocorrelation (Moran's I):")
    for method, value in spatial_autocorr.items():
        print(f"  {method}: {value:.3f}")
```

### Cluster Neighbor Analysis

```python
# Analyze spatial relationships between clusters
def analyze_cluster_neighbors(positions, cluster_labels, radius=50):
    """Analyze which clusters are neighbors"""
    
    from scipy.spatial.distance import cdist
    
    neighbor_counts = {}
    unique_clusters = np.unique(cluster_labels)
    
    for cluster in unique_clusters:
        if cluster == -1:  # Skip noise points for DBSCAN
            continue
            
        cluster_mask = cluster_labels == cluster
        cluster_positions = positions[cluster_mask]
        
        # Find neighbors within radius
        distances = cdist(cluster_positions, positions)
        neighbor_mask = distances <= radius
        
        # Count neighbors by cluster
        neighbor_clusters = cluster_labels[np.any(neighbor_mask, axis=0)]
        neighbor_counts[cluster] = pd.Series(neighbor_clusters).value_counts()
    
    return neighbor_counts

# Analyze neighbors for each clustering method
if 'spatial' in adata_hvg.obsm:
    neighbor_analyses = {}
    
    for method in clustering_methods:
        if method in adata_hvg.obs.columns:
            neighbor_counts = analyze_cluster_neighbors(
                adata_hvg.obsm['spatial'], 
                adata_hvg.obs[method]
            )
            neighbor_analyses[method] = neighbor_counts
    
    # Display results
    for method, neighbor_counts in neighbor_analyses.items():
        print(f"\n{method.upper()} Cluster Neighbors:")
        for cluster, counts in neighbor_counts.items():
            print(f"  Cluster {cluster}:")
            for neighbor, count in counts.head(5).items():
                print(f"    -> Cluster {neighbor}: {count} neighbors")
```

## Cluster Validation

### Silhouette Analysis

```python
from sklearn.metrics import silhouette_score, silhouette_samples

def analyze_silhouette(adata, cluster_key='leiden'):
    """Analyze silhouette scores for clustering"""
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(adata.obsm['X_pca'][:, :n_pcs], 
                                     adata.obs[cluster_key])
    
    # Calculate silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(adata.obsm['X_pca'][:, :n_pcs], 
                                                 adata.obs[cluster_key])
    
    return silhouette_avg, sample_silhouette_values

# Analyze silhouette for each clustering method
silhouette_scores = {}

for method in clustering_methods:
    if method in adata_hvg.obs.columns:
        avg_score, sample_scores = analyze_silhouette(adata_hvg, method)
        silhouette_scores[method] = avg_score
        adata_hvg.obs[f'{method}_silhouette'] = sample_scores

print("Silhouette Scores:")
for method, score in silhouette_scores.items():
    print(f"  {method}: {score:.3f}")

# Plot silhouette scores
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, method in enumerate(clustering_methods):
    if method in adata_hvg.obs.columns:
        sample_scores = adata_hvg.obs[f'{method}_silhouette']
        axes[i].hist(sample_scores, bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel('Silhouette Score')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{method.upper()} Silhouette Distribution')
        axes[i].axvline(silhouette_scores[method], color='red', linestyle='--', 
                       label=f'Mean: {silhouette_scores[method]:.3f}')
        axes[i].legend()

plt.tight_layout()
plt.show()
```

### Adjusted Rand Index

```python
from sklearn.metrics import adjusted_rand_score

# Compare clustering methods
if len(clustering_methods) > 1:
    comparison_matrix = np.zeros((len(clustering_methods), len(clustering_methods)))
    
    for i, method1 in enumerate(clustering_methods):
        for j, method2 in enumerate(clustering_methods):
            if method1 in adata_hvg.obs.columns and method2 in adata_hvg.obs.columns:
                ari = adjusted_rand_score(adata_hvg.obs[method1], adata_hvg.obs[method2])
                comparison_matrix[i, j] = ari
    
    # Plot comparison matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(comparison_matrix, annot=True, cmap='viridis', 
                xticklabels=clustering_methods, yticklabels=clustering_methods)
    plt.title('Adjusted Rand Index Comparison')
    plt.show()
```

## Cell Type Annotation

### Manual Annotation

```python
# Create cell type annotations based on marker features
def annotate_cell_types(adata, cluster_key='leiden'):
    """Manually annotate cell types based on markers"""
    
    # Define cell type markers (example)
    cell_type_markers = {
        'T_cells': ['CD3', 'CD8', 'CD4'],
        'B_cells': ['CD19', 'CD20'],
        'Macrophages': ['CD68', 'CD163'],
        'Endothelial': ['CD31', 'CD34'],
        'Fibroblasts': ['Vimentin', 'Collagen']
    }
    
    # Calculate mean expression of markers per cluster
    cluster_annotations = {}
    
    for cluster in adata.obs[cluster_key].unique():
        mask = adata.obs[cluster_key] == cluster
        cluster_cells = adata[mask]
        
        # Calculate expression scores for each cell type
        cell_type_scores = {}
        
        for cell_type, markers in cell_type_markers.items():
            # Check which markers are available
            available_markers = [m for m in markers if m in adata.var_names]
            if available_markers:
                mean_expression = cluster_cells[:, available_markers].X.mean()
                cell_type_scores[cell_type] = mean_expression
        
        # Assign cell type with highest score
        if cell_type_scores:
            best_cell_type = max(cell_type_scores, key=cell_type_scores.get)
            cluster_annotations[cluster] = best_cell_type
        else:
            cluster_annotations[cluster] = 'Unknown'
    
    return cluster_annotations

# Annotate cell types
cell_type_annotations = annotate_cell_types(adata_hvg, 'leiden')

# Add annotations to AnnData
adata_hvg.obs['cell_type'] = adata_hvg.obs['leiden'].map(cell_type_annotations)

print("Cell Type Annotations:")
for cluster, cell_type in cell_type_annotations.items():
    print(f"  Cluster {cluster}: {cell_type}")

# Visualize annotated clusters
sc.pl.umap(adata_hvg, color='cell_type')
plt.show()

if 'spatial' in adata_hvg.obsm:
    sc.pl.spatial(adata_hvg, color='cell_type')
    plt.show()
```

## Save Results

```python
# Save clustering results
adata_hvg.write('clustering_results.h5ad')

# Save cluster statistics
cluster_stats = analyze_clusters(adata_hvg, 'leiden')
cluster_stats.to_csv('cluster_statistics.csv')

# Save marker features
markers.to_csv('marker_features.csv')

# Save cell type annotations
cell_type_df = pd.DataFrame({
    'cluster': list(cell_type_annotations.keys()),
    'cell_type': list(cell_type_annotations.values())
})
cell_type_df.to_csv('cell_type_annotations.csv', index=False)

print("Results saved:")
print("- clustering_results.h5ad: AnnData with clustering results")
print("- cluster_statistics.csv: Cluster characteristics")
print("- marker_features.csv: Marker features for each cluster")
print("- cell_type_annotations.csv: Cell type annotations")
```

## Summary

This clustering tutorial covered:

1. **Data preprocessing** and normalization
2. **Dimensionality reduction** (PCA, UMAP, t-SNE)
3. **Multiple clustering methods** (Leiden, Louvain, K-means, DBSCAN)
4. **Cluster analysis** and marker identification
5. **Spatial analysis** of clusters
6. **Cluster validation** using silhouette scores
7. **Cell type annotation** based on markers

The tutorial provides a comprehensive framework for identifying and characterizing cell types in spatial transcriptomics data.

## Next Steps

- 🧬 [Spatial Analysis Tutorial](spatial-analysis.md) - Advanced spatial methods
- 📊 [Complete Pipeline Example](../examples/complete-pipeline.md) - End-to-end workflow
- 🎨 [Visualization Examples](../examples/visualization.md) - Advanced plotting
- 🔧 [API Reference](../api-reference/) - Detailed function documentation
