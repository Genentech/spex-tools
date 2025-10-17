# Spatial Analysis Tutorial

This tutorial covers spatial analysis techniques for spatial omics data (proteomics and transcriptomics) using SPEX.

## Overview

Spatial analysis in SPEX provides tools to understand spatial relationships between cells and genes in tissue samples. This includes:

- **CLQ Analysis**: Co-occurrence and Local Quotient analysis
- **Niche Analysis**: Identification of spatial microenvironments
- **Spatial Autocorrelation**: Analysis of spatial patterns
- **Neighborhood Analysis**: Cell-to-cell relationships

## Prerequisites

Before starting this tutorial, make sure you have:

- Completed the [Segmentation Basics](segmentation-basics.md) tutorial
- Completed the [Clustering Guide](clustering-guide.md) tutorial
- An AnnData object with spatial coordinates and cell type annotations

## Getting Started

### Import Required Libraries

```python
import spex
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting
sc.settings.set_figure_params(dpi=80, facecolor='white')
```

### Load Your Data

```python
# Load your processed AnnData object
adata = sc.read_h5ad("path/to/your/processed_data.h5ad")

# Ensure spatial coordinates are available
print(f"Spatial coordinates: {adata.obsm.keys()}")
print(f"Cell types: {adata.obs['cell_type'].unique()}")
```

## CLQ Analysis

### Understanding CLQ

CLQ (Co-occurrence and Local Quotient) analysis measures the spatial co-occurrence of different cell types. It helps identify:

- Which cell types tend to be found together
- Spatial patterns in tissue organization
- Potential functional relationships

**Key Concepts:**
- **CLQ > 1**: Cell types are attracted to each other (co-localized)
- **CLQ < 1**: Cell types avoid each other (spatially separated)
- **CLQ = 1**: Cell types are randomly distributed relative to each other

### Basic CLQ Analysis

```python
from spex import CLQ_vec_numba

# Perform CLQ analysis
adata_out, results = CLQ_vec_numba(
    adata,
    clust_col='leiden',      # Column with cluster labels
    radius=50,              # Spatial radius in microns
    n_perms=1000            # Number of permutations for significance testing
)

# View results
print("Global CLQ matrix:")
print(results['global_clq'])

print("\nPermutation test p-values:")
print(results['permute_test'])

# Access local CLQ values for each cell
local_clq = adata_out.obsm['local_clq']
neighborhood_vectors = adata_out.obsm['NCV']

print(f"Local CLQ shape: {local_clq.shape}")
print(f"Neighborhood vectors shape: {neighborhood_vectors.shape}")
```

### Visualizing CLQ Results

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create a comprehensive CLQ visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Global CLQ matrix
sns.heatmap(
    results['global_clq'],
    annot=True,
    cmap='RdBu_r',
    center=1.0,
    square=True,
    fmt='.3f',
    ax=axes[0, 0]
)
axes[0, 0].set_title('Global Co-Localization Quotient')
axes[0, 0].set_xlabel('Cell Type')
axes[0, 0].set_ylabel('Cell Type')

# 2. Permutation test p-values
sns.heatmap(
    results['permute_test'],
    annot=True,
    cmap='RdBu_r',
    center=0.5,
    square=True,
    fmt='.3f',
    ax=axes[0, 1]
)
axes[0, 1].set_title('CLQ Permutation Test P-values')
axes[0, 1].set_xlabel('Cell Type')
axes[0, 1].set_ylabel('Cell Type')

# 3. Significant interactions only
significant_mask = results['permute_test'] < 0.05
significant_clq = results['global_clq'] * significant_mask

sns.heatmap(
    significant_clq,
    annot=True,
    cmap='RdBu_r',
    center=1.0,
    square=True,
    fmt='.3f',
    ax=axes[1, 0]
)
axes[1, 0].set_title('Significant CLQ Interactions (p < 0.05)')
axes[1, 0].set_xlabel('Cell Type')
axes[1, 0].set_ylabel('Cell Type')

# 4. Local CLQ distribution
cell_types = adata.obs['leiden'].unique()
local_clq_df = pd.DataFrame(
    local_clq,
    columns=cell_types
)

local_clq_df.boxplot(ax=axes[1, 1])
axes[1, 1].set_title('Distribution of Local CLQ Values')
axes[1, 1].set_xlabel('Cell Type')
axes[1, 1].set_ylabel('Local CLQ Value')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

### Advanced CLQ Analysis

```python
# Analyze significant interactions
significant_mask = results['permute_test'] < 0.05
significant_clq = results['global_clq'] * significant_mask

print("Significant CLQ interactions (p < 0.05):")
cell_types = adata.obs['leiden'].unique()

for i in range(significant_clq.shape[0]):
    for j in range(significant_clq.shape[1]):
        if significant_mask[i, j] and i != j:
            cell_type_i = cell_types[i]
            cell_type_j = cell_types[j]
            clq_value = significant_clq[i, j]
            p_value = results['permute_test'][i, j]
            
            if clq_value > 1:
                relationship = "attraction"
            else:
                relationship = "avoidance"
                
            print(f"{cell_type_i} - {cell_type_j}: CLQ={clq_value:.3f}, p={p_value:.3f} ({relationship})")

# Create spatial visualization of local CLQ
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot spatial distribution of cell types
scatter = axes[0].scatter(
    adata.obsm['spatial'][:, 0],
    adata.obsm['spatial'][:, 1],
    c=adata.obs['leiden'].astype('category').cat.codes,
    cmap='tab20',
    s=20,
    alpha=0.8
)
axes[0].set_title('Spatial Distribution of Cell Types')
axes[0].set_xlabel('X Coordinate')
axes[0].set_ylabel('Y Coordinate')

# Plot local CLQ for a specific cell type
cell_type_idx = 0  # First cell type
local_clq_values = local_clq[:, cell_type_idx]
scatter = axes[1].scatter(
    adata.obsm['spatial'][:, 0],
    adata.obsm['spatial'][:, 1],
    c=local_clq_values,
    cmap='RdBu_r',
    s=20,
    alpha=0.8
)
axes[1].set_title(f'Local CLQ for {cell_types[cell_type_idx]}')
axes[1].set_xlabel('X Coordinate')
axes[1].set_ylabel('Y Coordinate')
plt.colorbar(scatter, ax=axes[1])

# Plot neighborhood size
neighborhood_sizes = np.sum(neighborhood_vectors, axis=1)
scatter = axes[2].scatter(
    adata.obsm['spatial'][:, 0],
    adata.obsm['spatial'][:, 1],
    c=neighborhood_sizes,
    cmap='viridis',
    s=20,
    alpha=0.8
)
axes[2].set_title('Neighborhood Size')
axes[2].set_xlabel('X Coordinate')
axes[2].set_ylabel('Y Coordinate')
plt.colorbar(scatter, ax=axes[2])

plt.tight_layout()
plt.show()
```

### Parameter Optimization

```python
# Test different radius values
radii = [25, 50, 100, 150]
clq_results = {}

for radius in radii:
    print(f"Testing radius: {radius} microns")
    adata_out, results = CLQ_vec_numba(
        adata,
        clust_col='leiden',
        radius=radius,
        n_perms=100  # Reduced for faster testing
    )
    clq_results[radius] = results['global_clq']

# Compare results across different radii
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, radius in enumerate(radii):
    row, col = idx // 2, idx % 2
    
    sns.heatmap(
        clq_results[radius],
        annot=True,
        cmap='RdBu_r',
        center=1.0,
        square=True,
        fmt='.3f',
        ax=axes[row, col]
    )
    axes[row, col].set_title(f'CLQ Matrix (radius={radius}μm)')

plt.tight_layout()
plt.show()
```

## Niche Analysis

### What are Spatial Niches?

Spatial niches are microenvironments where specific cell types are enriched. They represent functional units in tissue organization and can indicate:

- **Functional microenvironments** with specific cell type compositions
- **Tissue organization patterns** and spatial domains  
- **Potential signaling centers** or specialized regions
- **Disease-related spatial alterations**

### Finding Niches

```python
from spex import niche

# Perform niche analysis
adata = niche(
    adata,
    spatial_weight=0.3,    # Weight for spatial neighbors
    resolution=1.0,        # Clustering resolution
    method='leiden'        # Clustering method
)

# Access results
print("Niche analysis completed!")
print(f"Identified niches: {adata.obs['leiden'].unique()}")
print(f"Number of cells per niche:")
print(adata.obs['leiden'].value_counts())
```

### Visualizing Niche Results

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create comprehensive niche visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Spatial distribution of niches
scatter = axes[0, 0].scatter(
    adata.obsm['spatial'][:, 0],
    adata.obsm['spatial'][:, 1],
    c=adata.obs['leiden'].astype('category').cat.codes,
    cmap='tab20',
    s=20,
    alpha=0.8
)
axes[0, 0].set_title('Spatial Distribution of Niches')
axes[0, 0].set_xlabel('X Coordinate')
axes[0, 0].set_ylabel('Y Coordinate')
axes[0, 0].set_aspect('equal')

# 2. Niche size distribution
niche_counts = adata.obs['leiden'].value_counts()
axes[0, 1].bar(range(len(niche_counts)), niche_counts.values)
axes[0, 1].set_title('Niche Size Distribution')
axes[0, 1].set_xlabel('Niche ID')
axes[0, 1].set_ylabel('Number of Cells')
axes[0, 1].set_xticks(range(len(niche_counts)))
axes[0, 1].set_xticklabels(niche_counts.index, rotation=45)

# 3. Niche composition analysis (if cell types are available)
if 'cell_type' in adata.obs.columns:
    niche_composition = pd.crosstab(adata.obs['leiden'], adata.obs['cell_type'])
    
    sns.heatmap(
        niche_composition,
        annot=True,
        cmap='Blues',
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Niche Composition by Cell Type')
    axes[1, 0].set_xlabel('Cell Type')
    axes[1, 0].set_ylabel('Niche ID')

# 4. Niche spatial characteristics
niche_centroids = []
niche_radii = []

for niche_id in adata.obs['leiden'].unique():
    niche_mask = adata.obs['leiden'] == niche_id
    niche_coords = adata.obsm['spatial'][niche_mask]
    
    # Calculate centroid
    centroid = np.mean(niche_coords, axis=0)
    niche_centroids.append(centroid)
    
    # Calculate radius (distance from centroid to farthest cell)
    distances = np.linalg.norm(niche_coords - centroid, axis=1)
    radius = np.max(distances)
    niche_radii.append(radius)

# Plot niche characteristics
niche_radii = np.array(niche_radii)
axes[1, 1].hist(niche_radii, bins=20, alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Distribution of Niche Radii')
axes[1, 1].set_xlabel('Niche Radius (microns)')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

### Advanced Niche Analysis

```python
# Analyze niche characteristics
niche_stats = {}

for niche_id in adata.obs['leiden'].unique():
    niche_mask = adata.obs['leiden'] == niche_id
    niche_cells = adata[niche_mask]
    
    # Basic statistics
    niche_stats[niche_id] = {
        'n_cells': len(niche_cells),
        'centroid': np.mean(niche_cells.obsm['spatial'], axis=0),
        'radius': np.max(np.linalg.norm(
            niche_cells.obsm['spatial'] - np.mean(niche_cells.obsm['spatial'], axis=0), 
            axis=1
        )),
        'density': len(niche_cells) / (np.pi * np.max(np.linalg.norm(
            niche_cells.obsm['spatial'] - np.mean(niche_cells.obsm['spatial'], axis=0), 
            axis=1
        ))**2)
    }

# Create summary table
niche_summary = pd.DataFrame(niche_stats).T
print("Niche Analysis Summary:")
print(niche_summary)

# Visualize niche relationships
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Niche size vs radius
axes[0].scatter(niche_summary['radius'], niche_summary['n_cells'], alpha=0.7)
axes[0].set_xlabel('Niche Radius (microns)')
axes[0].set_ylabel('Number of Cells')
axes[0].set_title('Niche Size vs Radius')

# Niche density distribution
axes[1].hist(niche_summary['density'], bins=15, alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Cell Density (cells/μm²)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Niche Densities')

plt.tight_layout()
plt.show()
```

### Parameter Optimization for Niche Analysis

```python
# Test different spatial weights
spatial_weights = [0.0, 0.1, 0.3, 0.5]
niche_results = {}

for weight in spatial_weights:
    print(f"Testing spatial weight: {weight}")
    
    # Create a copy for testing
    adata_test = adata.copy()
    
    # Perform niche analysis
    adata_test = niche(
        adata_test,
        spatial_weight=weight,
        resolution=1.0,
        method='leiden'
    )
    
    # Store results
    niche_results[weight] = {
        'n_niches': len(adata_test.obs['leiden'].unique()),
        'niche_sizes': adata_test.obs['leiden'].value_counts().values,
        'niche_labels': adata_test.obs['leiden']
    }

# Compare results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for idx, weight in enumerate(spatial_weights):
    row, col = idx // 2, idx % 2
    
    # Plot spatial distribution
    scatter = axes[row, col].scatter(
        adata.obsm['spatial'][:, 0],
        adata.obsm['spatial'][:, 1],
        c=niche_results[weight]['niche_labels'].astype('category').cat.codes,
        cmap='tab20',
        s=15,
        alpha=0.8
    )
    axes[row, col].set_title(f'Spatial Weight = {weight}\n({niche_results[weight]["n_niches"]} niches)')
    axes[row, col].set_xlabel('X Coordinate')
    axes[row, col].set_ylabel('Y Coordinate')
    axes[row, col].set_aspect('equal')

plt.tight_layout()
plt.show()

# Compare niche statistics
print("Niche Analysis Comparison:")
print("=" * 50)
for weight in spatial_weights:
    n_niches = niche_results[weight]['n_niches']
    avg_size = np.mean(niche_results[weight]['niche_sizes'])
    print(f"Spatial weight {weight}: {n_niches} niches, avg size {avg_size:.1f} cells")
```

### Biological Interpretation of Niche Analysis

**Key Insights:**
1. **Niche Size**: Larger niches may indicate broader functional domains or tissue regions
2. **Niche Composition**: Cell type mixtures suggest functional relationships and signaling networks
3. **Spatial Distribution**: Clustered vs. dispersed niches indicate different tissue organization patterns
4. **Niche Density**: High-density niches may indicate active regions or specialized microenvironments

**Parameter Selection Guidelines:**
- **spatial_weight**: 
  - 0.0-0.1: Focus on gene expression similarity
  - 0.3-0.5: Balance expression and spatial proximity
  - >0.5: Strong emphasis on spatial relationships
- **resolution**:
  - 0.5-1.0: Identify larger, general niches
  - 1.0-2.0: Identify smaller, specific niches
  - >2.0: Very fine-grained niche identification

## Spatial Autocorrelation

### Understanding Spatial Autocorrelation

Spatial autocorrelation measures the degree to which similar values cluster together in space. It helps identify:

- **Spatial gradients** and tissue domains
- **Functional regions** with similar gene expression
- **Spatial patterns** in cell type distribution
- **Tissue organization** principles

**Key Concepts:**
- **Positive autocorrelation**: Similar values cluster together
- **Negative autocorrelation**: Dissimilar values cluster together
- **Random distribution**: No spatial organization

### Moran's I Analysis

Moran's I measures spatial autocorrelation using correlation between values and their spatial neighbors.

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
import pandas as pd

def moran_i(adata, variable, spatial_key='spatial', weight_type='distance', radius=50, k_neighbors=10):
    """Calculate Moran's I spatial autocorrelation coefficient."""
    
    # Get spatial coordinates and variable values
    coords = adata.obsm[spatial_key]
    
    if isinstance(variable, str):
        if variable in adata.obs.columns:
            values = adata.obs[variable].values
        elif variable in adata.var_names:
            values = adata[:, variable].X.toarray().flatten()
        else:
            raise ValueError(f"Variable {variable} not found in adata")
    else:
        values = np.array(variable)
    
    # Remove NaN values
    mask = ~np.isnan(values)
    coords = coords[mask]
    values = values[mask]
    n = len(values)
    
    if n < 2:
        return np.nan, np.nan, {}
    
    # Calculate spatial weights matrix
    if weight_type == 'distance':
        dist_matrix = squareform(pdist(coords))
        weights = (dist_matrix <= radius).astype(float)
        np.fill_diagonal(weights, 0)
    elif weight_type == 'knn':
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        weights = np.zeros((n, n))
        for i, idx in enumerate(indices):
            weights[i, idx[1:]] = 1
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")
    
    # Normalize weights
    row_sums = weights.sum(axis=1)
    row_sums[row_sums == 0] = 1
    weights = weights / row_sums[:, np.newaxis]
    
    # Calculate Moran's I
    mean_val = np.mean(values)
    dev = values - mean_val
    dev_sq = dev ** 2
    
    numerator = np.sum(weights * np.outer(dev, dev))
    denominator = np.sum(dev_sq)
    
    moran_i = (n / (2 * np.sum(weights))) * (numerator / denominator)
    
    # Calculate significance
    expected_i = -1 / (n - 1)
    s1 = np.sum(weights ** 2)
    s2 = np.sum((weights.sum(axis=0)) ** 2)
    
    var_i = (n * ((n**2 - 3*n + 3)*s1 - n*s2 + 3*np.sum(weights)**2) - 
             (n**2 * ((n**2 - n)*s1 - 2*n*s2 + 6*np.sum(weights)**2))) / \
            ((n-1)*(n-2)*(n-3)*np.sum(weights)**2)
    
    z_score = (moran_i - expected_i) / np.sqrt(var_i)
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    
    stats = {
        'expected_i': expected_i,
        'variance': var_i,
        'z_score': z_score,
        'weights_sum': np.sum(weights)
    }
    
    return moran_i, p_value, stats

# Example usage
gene_name = 'CD3'
moran_i_val, p_val, stats = moran_i(
    adata,
    variable=gene_name,
    spatial_key='spatial',
    weight_type='distance',
    radius=100
)

print(f"Moran's I for {gene_name}: {moran_i_val:.4f}")
print(f"P-value: {p_val:.4e}")
print(f"Z-score: {stats['z_score']:.4f}")
```

### Geary's C Analysis

Geary's C is an alternative measure of spatial autocorrelation that focuses on differences between neighboring values.

```python
def geary_c(adata, variable, spatial_key='spatial', weight_type='distance', radius=50, k_neighbors=10):
    """Calculate Geary's C spatial autocorrelation coefficient."""
    
    # Get spatial coordinates and variable values
    coords = adata.obsm[spatial_key]
    
    if isinstance(variable, str):
        if variable in adata.obs.columns:
            values = adata.obs[variable].values
        elif variable in adata.var_names:
            values = adata[:, variable].X.toarray().flatten()
        else:
            raise ValueError(f"Variable {variable} not found in adata")
    else:
        values = np.array(variable)
    
    # Remove NaN values
    mask = ~np.isnan(values)
    coords = coords[mask]
    values = values[mask]
    n = len(values)
    
    if n < 2:
        return np.nan, np.nan, {}
    
    # Calculate spatial weights matrix (same as Moran's I)
    if weight_type == 'distance':
        dist_matrix = squareform(pdist(coords))
        weights = (dist_matrix <= radius).astype(float)
        np.fill_diagonal(weights, 0)
    elif weight_type == 'knn':
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        weights = np.zeros((n, n))
        for i, idx in enumerate(indices):
            weights[i, idx[1:]] = 1
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")
    
    # Calculate Geary's C
    mean_val = np.mean(values)
    dev_sq = (values - mean_val) ** 2
    
    numerator = np.sum(weights * np.outer(dev_sq, np.ones(n)))
    denominator = 2 * np.sum(weights) * np.sum(dev_sq)
    
    geary_c = ((n - 1) / denominator) * numerator
    
    # Calculate significance
    expected_c = 1.0
    s1 = np.sum(weights ** 2)
    s2 = np.sum((weights.sum(axis=0)) ** 2)
    
    var_c = ((2 * s1 + s2) * (n - 1) - 4 * np.sum(weights)**2) / \
            (2 * (n + 1) * np.sum(weights)**2)
    
    z_score = (geary_c - expected_c) / np.sqrt(var_c)
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    
    stats = {
        'expected_c': expected_c,
        'variance': var_c,
        'z_score': z_score,
        'weights_sum': np.sum(weights)
    }
    
    return geary_c, p_value, stats

# Example usage
gene_name = 'CD8'
geary_c_val, p_val, stats = geary_c(
    adata,
    variable=gene_name,
    spatial_key='spatial',
    weight_type='distance',
    radius=100
)

print(f"Geary's C for {gene_name}: {geary_c_val:.4f}")
print(f"P-value: {p_val:.4e}")
print(f"Z-score: {stats['z_score']:.4f}")
```

### Batch Analysis of Multiple Genes

```python
def analyze_spatial_autocorrelation(adata, genes=None, spatial_key='spatial', 
                                  weight_type='distance', radius=100):
    """Perform spatial autocorrelation analysis on multiple genes."""
    
    if genes is None:
        genes = adata.var_names.tolist()
    
    results = []
    
    for gene in genes:
        try:
            # Calculate Moran's I
            moran_i_val, moran_p, moran_stats = moran_i(
                adata, gene, spatial_key, weight_type, radius
            )
            
            # Calculate Geary's C
            geary_c_val, geary_p, geary_stats = geary_c(
                adata, gene, spatial_key, weight_type, radius
            )
            
            results.append({
                'gene': gene,
                'moran_i': moran_i_val,
                'moran_p': moran_p,
                'moran_z': moran_stats.get('z_score', np.nan),
                'geary_c': geary_c_val,
                'geary_p': geary_p,
                'geary_z': geary_stats.get('z_score', np.nan)
            })
            
        except Exception as e:
            print(f"Error analyzing {gene}: {e}")
            results.append({
                'gene': gene,
                'moran_i': np.nan,
                'moran_p': np.nan,
                'moran_z': np.nan,
                'geary_c': np.nan,
                'geary_p': np.nan,
                'geary_z': np.nan
            })
    
    return pd.DataFrame(results)

# Analyze spatial autocorrelation for top variable genes
top_genes = adata.var_names[:50]  # First 50 genes
autocorr_results = analyze_spatial_autocorrelation(
    adata,
    genes=top_genes,
    spatial_key='spatial',
    weight_type='distance',
    radius=100
)

# Display results
print("Spatial Autocorrelation Results:")
print(autocorr_results.head(10))

# Find genes with significant spatial autocorrelation
significant_genes = autocorr_results[
    (autocorr_results['moran_p'] < 0.05) | 
    (autocorr_results['geary_p'] < 0.05)
]

print(f"\nGenes with significant spatial autocorrelation: {len(significant_genes)}")
print(significant_genes[['gene', 'moran_i', 'moran_p', 'geary_c', 'geary_p']].head())
```

### Visualizing Spatial Autocorrelation Results

```python
# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Moran's I distribution
axes[0, 0].hist(autocorr_results['moran_i'].dropna(), bins=30, alpha=0.7, edgecolor='black')
axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
axes[0, 0].set_title('Distribution of Moran\'s I')
axes[0, 0].set_xlabel('Moran\'s I')
axes[0, 0].set_ylabel('Frequency')

# 2. Geary's C distribution
axes[0, 1].hist(autocorr_results['geary_c'].dropna(), bins=30, alpha=0.7, edgecolor='black')
axes[0, 1].axvline(1, color='red', linestyle='--', alpha=0.7)
axes[0, 1].set_title('Distribution of Geary\'s C')
axes[0, 1].set_xlabel('Geary\'s C')
axes[0, 1].set_ylabel('Frequency')

# 3. Moran's I vs Geary's C
axes[0, 2].scatter(autocorr_results['moran_i'], autocorr_results['geary_c'], alpha=0.6)
axes[0, 2].set_xlabel('Moran\'s I')
axes[0, 2].set_ylabel('Geary\'s C')
axes[0, 2].set_title('Moran\'s I vs Geary\'s C')

# 4. P-value distributions
axes[1, 0].hist(autocorr_results['moran_p'].dropna(), bins=30, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(0.05, color='red', linestyle='--', alpha=0.7)
axes[1, 0].set_title('Distribution of Moran\'s I P-values')
axes[1, 0].set_xlabel('P-value')
axes[1, 0].set_ylabel('Frequency')

# 5. Z-score distributions
axes[1, 1].hist(autocorr_results['moran_z'].dropna(), bins=30, alpha=0.7, edgecolor='black')
axes[1, 1].axvline(1.96, color='red', linestyle='--', alpha=0.7)
axes[1, 1].axvline(-1.96, color='red', linestyle='--', alpha=0.7)
axes[1, 1].set_title('Distribution of Moran\'s I Z-scores')
axes[1, 1].set_xlabel('Z-score')
axes[1, 1].set_ylabel('Frequency')

# 6. Top genes with significant autocorrelation
if len(significant_genes) > 0:
    top_10 = significant_genes.head(10)
    axes[1, 2].barh(range(len(top_10)), top_10['moran_i'])
    axes[1, 2].set_yticks(range(len(top_10)))
    axes[1, 2].set_yticklabels(top_10['gene'])
    axes[1, 2].set_xlabel('Moran\'s I')
    axes[1, 2].set_title('Top 10 Genes with Significant Autocorrelation')
    axes[1, 2].axvline(0, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

### Parameter Optimization for Spatial Autocorrelation

```python
# Test different radius values
radii = [25, 50, 100, 150, 200]
radius_results = {}

for radius in radii:
    print(f"Testing radius: {radius} microns")
    
    # Analyze a subset of genes for speed
    test_genes = adata.var_names[:20]
    results = analyze_spatial_autocorrelation(
        adata,
        genes=test_genes,
        spatial_key='spatial',
        weight_type='distance',
        radius=radius
    )
    
    radius_results[radius] = {
        'mean_moran_i': results['moran_i'].mean(),
        'mean_geary_c': results['geary_c'].mean(),
        'significant_count': len(results[
            (results['moran_p'] < 0.05) | (results['geary_p'] < 0.05)
        ])
    }

# Compare results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

radii_list = list(radius_results.keys())
moran_means = [radius_results[r]['mean_moran_i'] for r in radii_list]
geary_means = [radius_results[r]['mean_geary_c'] for r in radii_list]
sig_counts = [radius_results[r]['significant_count'] for r in radii_list]

axes[0].plot(radii_list, moran_means, 'o-')
axes[0].set_xlabel('Radius (microns)')
axes[0].set_ylabel('Mean Moran\'s I')
axes[0].set_title('Mean Moran\'s I vs Radius')

axes[1].plot(radii_list, geary_means, 'o-')
axes[1].set_xlabel('Radius (microns)')
axes[1].set_ylabel('Mean Geary\'s C')
axes[1].set_title('Mean Geary\'s C vs Radius')

axes[2].plot(radii_list, sig_counts, 'o-')
axes[2].set_xlabel('Radius (microns)')
axes[2].set_ylabel('Number of Significant Genes')
axes[2].set_title('Significant Genes vs Radius')

plt.tight_layout()
plt.show()

print("Parameter optimization results:")
for radius in radii:
    print(f"Radius {radius}μm: Moran's I={radius_results[radius]['mean_moran_i']:.4f}, "
          f"Geary's C={radius_results[radius]['mean_geary_c']:.4f}, "
          f"Significant={radius_results[radius]['significant_count']}")
```

### Biological Interpretation of Spatial Autocorrelation

**Key Insights:**
1. **Positive autocorrelation (Moran's I > 0, Geary's C < 1)**: Indicates spatial gradients, tissue domains, or functional regions
2. **Negative autocorrelation (Moran's I < 0, Geary's C > 1)**: Indicates spatial competition, exclusion zones, or alternating patterns
3. **Random distribution (Moran's I ≈ 0, Geary's C ≈ 1)**: Suggests no spatial organization for the analyzed feature

**Parameter Selection Guidelines:**
- **Radius**: Should match the scale of biological features of interest
  - Small radius (25-50μm): Local cell-cell interactions
  - Medium radius (100-150μm): Tissue microenvironments
  - Large radius (200+μm): Tissue domains or gradients
- **Weight type**: 
  - Distance-based: More sensitive to spatial scale
  - KNN-based: More robust to varying cell densities

**Statistical Considerations:**
- P-values < 0.05 indicate statistically significant spatial patterns
- Z-scores > 1.96 or < -1.96 indicate significant autocorrelation
- Both Moran's I and Geary's C should be used together for robust analysis

## Neighborhood Analysis

### Cell Neighborhood Relationships

Neighborhood analysis examines the spatial relationships between different cell types.

```python
from spex import neighborhood_analysis

# Analyze neighborhoods
neighbors = neighborhood_analysis(
    adata,
    radius=100,
    cell_types=['T_cells', 'B_cells', 'Macrophages']
)

# View neighborhood statistics
print(neighbors.head())
```

### Neighborhood Visualization

```python
# Create neighborhood network
import networkx as nx

# Build network from neighborhood data
G = nx.Graph()

for _, row in neighbors.iterrows():
    G.add_edge(row['cell_type_1'], row['cell_type_2'], 
               weight=row['interaction_strength'])

# Plot network
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=2000, font_size=10, font_weight='bold')
plt.title("Cell Type Neighborhood Network")
plt.show()
```

## Complete Spatial Analysis Pipeline

Here's a complete workflow combining all spatial analysis techniques:

```python
def complete_spatial_analysis(adata, cell_types, radius=100):
    """
    Complete spatial analysis pipeline.
    """
    results = {}
    
    # 1. CLQ Analysis
    print("Performing CLQ analysis...")
    results['clq'] = clq_analysis(adata, cell_types, radius=radius)
    
    # 2. Niche Analysis
    print("Finding spatial niches...")
    results['niches'] = niche_analysis(adata, radius=radius//2)
    
    # 3. Spatial Autocorrelation
    print("Calculating spatial autocorrelation...")
    results['moran'] = {}
    for gene in ['CD3D', 'CD19', 'CD68']:
        if gene in adata.var_names:
            results['moran'][gene] = moran_i(adata, gene)
    
    # 4. Neighborhood Analysis
    print("Analyzing neighborhoods...")
    results['neighbors'] = neighborhood_analysis(adata, radius, cell_types)
    
    return results

# Run complete analysis
cell_types = ['T_cells', 'B_cells', 'Macrophages', 'Endothelial']
results = complete_spatial_analysis(adata, cell_types)

print("Spatial analysis completed!")
```

## Best Practices

### Parameter Selection

1. **Radius**: Choose based on tissue scale and cell size
   - Small cells (lymphocytes): 50-100 μm
   - Large cells (macrophages): 100-200 μm
   - Tissue-level analysis: 200-500 μm

2. **Permutations**: Use at least 1000 for statistical significance

3. **Cell Types**: Ensure accurate cell type annotations

### Quality Control

```python
# Check spatial coordinates
print(f"Coordinate range: X={adata.obsm['spatial'][:, 0].min():.1f}-{adata.obsm['spatial'][:, 0].max():.1f}")
print(f"Coordinate range: Y={adata.obsm['spatial'][:, 1].min():.1f}-{adata.obsm['spatial'][:, 1].max():.1f}")

# Check cell type distribution
cell_counts = adata.obs['cell_type'].value_counts()
print("Cell type distribution:")
print(cell_counts)
```

### Interpretation

1. **CLQ > 1**: Cell types co-occur more than expected by chance
2. **CLQ < 1**: Cell types avoid each other
3. **Moran's I > 0**: Positive spatial autocorrelation
4. **Moran's I < 0**: Negative spatial autocorrelation

## Troubleshooting

### Common Issues

1. **No spatial coordinates**: Ensure `adata.obsm['spatial']` exists
2. **Missing cell types**: Check cell type annotations
3. **Memory issues**: Reduce radius or sample size
4. **No significant results**: Increase permutations or check data quality

### Performance Tips

1. Use appropriate radius for your tissue scale
2. Sample data if working with large datasets
3. Use parallel processing for large analyses
4. Cache results for repeated analyses

## Next Steps

After completing this tutorial, you can:

1. Explore advanced spatial analysis techniques
2. Integrate spatial analysis with gene expression analysis
3. Create custom spatial analysis workflows
4. Apply spatial analysis to different tissue types

## Additional Resources

- [API Reference: Spatial Analysis](../api-reference/spatial-analysis.md)
- [Examples: Complete Pipeline](../examples/complete-pipeline.md)
- [Reference: FAQ](../reference/faq.md)

---

*This tutorial is under development. More examples and advanced techniques will be added.*
