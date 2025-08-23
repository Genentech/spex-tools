# Visualization Examples

This section provides comprehensive examples of visualization techniques for spatial transcriptomics data using SPEX.

## Overview

Visualization is crucial for understanding spatial transcriptomics data. SPEX provides various plotting functions for:

- **Segmentation Results**: Cell boundaries and labels
- **Gene Expression**: Spatial gene expression patterns
- **Clustering Results**: Cell type distributions
- **Spatial Analysis**: CLQ, niches, and neighborhoods
- **Quality Control**: Data quality metrics

## Prerequisites

```python
import spex
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set up plotting
sc.settings.set_figure_params(dpi=80, facecolor='white')
plt.style.use('default')
```

## Segmentation Visualization

### Basic Segmentation Plot

```python
from spex import load_image, cellpose_cellseg

# Load and segment image
array, channels = load_image("path/to/image.ome.tiff")
labels = cellpose_cellseg(array, seg_channels=[0], diameter=30)

# Plot segmentation results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(array[0], cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# Segmentation labels
im = axes[1].imshow(labels, cmap='tab20')
axes[1].set_title('Segmentation Labels')
axes[1].axis('off')
plt.colorbar(im, ax=axes[1])

# Overlay
axes[2].imshow(array[0], cmap='gray')
axes[2].imshow(labels, cmap='tab20', alpha=0.7)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

### Multi-channel Segmentation

```python
# Load multi-channel image
array, channels = load_image("path/to/multichannel.ome.tiff")

# Create RGB composite
rgb = np.stack([array[0], array[1], array[2]], axis=-1)
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

# Plot channels
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i in range(3):
    axes[0, i].imshow(array[i], cmap='gray')
    axes[0, i].set_title(f'Channel {i}: {channels[i]}')
    axes[0, i].axis('off')

# RGB composite
axes[1, 0].imshow(rgb)
axes[1, 0].set_title('RGB Composite')
axes[1, 0].axis('off')

# Segmentation overlay
labels = cellpose_cellseg(array, seg_channels=[0], diameter=30)
axes[1, 1].imshow(rgb)
axes[1, 1].imshow(labels, cmap='tab20', alpha=0.7)
axes[1, 1].set_title('Segmentation Overlay')
axes[1, 1].axis('off')

# Cell boundaries
from skimage import measure
contours = measure.find_contours(labels, 0.5)
axes[1, 2].imshow(rgb)
for contour in contours:
    axes[1, 2].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1)
axes[1, 2].set_title('Cell Boundaries')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
```

## Gene Expression Visualization

### Spatial Gene Expression

```python
# Load processed AnnData
adata = sc.read_h5ad("path/to/processed_data.h5ad")

# Plot gene expression
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

genes = ['CD3D', 'CD19', 'CD68', 'PECAM1']

for i, gene in enumerate(genes):
    if gene in adata.var_names:
        ax = axes[i//2, i%2]
        sc.pl.spatial(adata, color=gene, ax=ax, show=False, 
                     title=f'{gene} Expression')
        ax.set_aspect('equal')

plt.tight_layout()
plt.show()
```

### Expression Heatmap

```python
# Create expression heatmap
genes_of_interest = ['CD3D', 'CD19', 'CD68', 'PECAM1', 'CD4', 'CD8A']

# Get expression matrix
expr_matrix = adata[:, genes_of_interest].X.toarray()

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(expr_matrix.T, cmap='viridis', aspect='auto')
ax.set_yticks(range(len(genes_of_interest)))
ax.set_yticklabels(genes_of_interest)
ax.set_xlabel('Cells')
ax.set_ylabel('Genes')
plt.colorbar(im, ax=ax, label='Expression')
plt.title('Gene Expression Heatmap')
plt.show()
```

## Clustering Visualization

### Cell Type Distribution

```python
# Plot cell type distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Spatial plot
sc.pl.spatial(adata, color='cell_type', ax=axes[0], show=False,
              title='Cell Type Distribution')
axes[0].set_aspect('equal')

# Bar plot
cell_counts = adata.obs['cell_type'].value_counts()
axes[1].bar(range(len(cell_counts)), cell_counts.values)
axes[1].set_xticks(range(len(cell_counts)))
axes[1].set_xticklabels(cell_counts.index, rotation=45)
axes[1].set_ylabel('Number of Cells')
axes[1].set_title('Cell Type Counts')

plt.tight_layout()
plt.show()
```

### UMAP with Cell Types

```python
# Compute UMAP if not already done
if 'X_umap' not in adata.obsm:
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

# Plot UMAP
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# UMAP colored by cell type
sc.pl.umap(adata, color='cell_type', ax=axes[0], show=False,
           title='UMAP - Cell Types')

# UMAP colored by sample
sc.pl.umap(adata, color='sample', ax=axes[1], show=False,
           title='UMAP - Sample')

plt.tight_layout()
plt.show()
```

## Spatial Analysis Visualization

### CLQ Heatmap

```python
from spex import clq_analysis, plot_clq_heatmap

# Perform CLQ analysis
clq_results = clq_analysis(
    adata,
    cell_types=['T_cells', 'B_cells', 'Macrophages', 'Endothelial'],
    radius=100
)

# Plot CLQ heatmap
fig, ax = plt.subplots(figsize=(10, 8))
plot_clq_heatmap(clq_results, ax=ax)
plt.title('CLQ Analysis Results')
plt.tight_layout()
plt.show()
```

### Spatial Niches

```python
from spex import niche_analysis, plot_niche_map

# Find spatial niches
niches = niche_analysis(adata, radius=50)

# Plot niches
fig, ax = plt.subplots(figsize=(12, 10))
plot_niche_map(adata, niches, ax=ax)
plt.title('Spatial Niches')
plt.show()
```

### Neighborhood Network

```python
from spex import neighborhood_analysis
import networkx as nx

# Analyze neighborhoods
neighbors = neighborhood_analysis(
    adata,
    radius=100,
    cell_types=['T_cells', 'B_cells', 'Macrophages']
)

# Create network
G = nx.Graph()
for _, row in neighbors.iterrows():
    G.add_edge(row['cell_type_1'], row['cell_type_2'], 
               weight=row['interaction_strength'])

# Plot network
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=2000, font_size=10, font_weight='bold',
        edge_color='gray', width=[G[u][v]['weight'] for u, v in G.edges()])
plt.title('Cell Type Neighborhood Network')
plt.show()
```

## Quality Control Visualization

### Segmentation Quality

```python
# Plot segmentation quality metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Cell size distribution
cell_areas = [len(np.where(labels == i)[0]) for i in range(1, labels.max() + 1)]
axes[0, 0].hist(cell_areas, bins=50, alpha=0.7)
axes[0, 0].set_xlabel('Cell Area (pixels)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Cell Size Distribution')

# Cell count over time
# (assuming you have multiple time points)
if 'time' in adata.obs.columns:
    time_counts = adata.obs.groupby('time').size()
    axes[0, 1].plot(time_counts.index, time_counts.values, 'o-')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Cell Count')
    axes[0, 1].set_title('Cell Count Over Time')

# Gene detection rate
detection_rate = (adata.X > 0).mean(axis=0)
axes[1, 0].hist(detection_rate, bins=50, alpha=0.7)
axes[1, 0].set_xlabel('Detection Rate')
axes[1, 0].set_ylabel('Number of Genes')
axes[1, 0].set_title('Gene Detection Rate')

# Spatial coverage
spatial_coords = adata.obsm['spatial']
axes[1, 1].scatter(spatial_coords[:, 0], spatial_coords[:, 1], 
                   c=adata.obs['cell_type'].cat.codes, cmap='tab10', s=1)
axes[1, 1].set_xlabel('X Coordinate')
axes[1, 1].set_ylabel('Y Coordinate')
axes[1, 1].set_title('Spatial Coverage')
axes[1, 1].set_aspect('equal')

plt.tight_layout()
plt.show()
```

## Interactive Visualizations

### Interactive Spatial Plot

```python
import plotly.express as px
import plotly.graph_objects as go

# Create interactive spatial plot
fig = px.scatter(
    x=adata.obsm['spatial'][:, 0],
    y=adata.obsm['spatial'][:, 1],
    color=adata.obs['cell_type'],
    title='Interactive Spatial Plot',
    labels={'x': 'X Coordinate', 'y': 'Y Coordinate'}
)

fig.update_layout(
    width=800,
    height=600,
    showlegend=True
)

fig.show()
```

### Interactive Gene Expression

```python
# Interactive gene expression plot
gene = 'CD3D'
if gene in adata.var_names:
    fig = px.scatter(
        x=adata.obsm['spatial'][:, 0],
        y=adata.obsm['spatial'][:, 1],
        color=adata[:, gene].X.toarray().flatten(),
        title=f'{gene} Expression',
        labels={'x': 'X Coordinate', 'y': 'Y Coordinate', 'color': 'Expression'},
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        width=800,
        height=600
    )
    
    fig.show()
```

## Custom Plotting Functions

### Multi-panel Figure

```python
def create_multi_panel_figure(adata, genes, figsize=(20, 15)):
    """
    Create a multi-panel figure with various visualizations.
    """
    n_genes = len(genes)
    n_cols = 4
    n_rows = (n_genes + 3) // n_cols  # +3 for additional panels
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot gene expression
    for i, gene in enumerate(genes):
        if gene in adata.var_names:
            sc.pl.spatial(adata, color=gene, ax=axes[i], show=False,
                         title=f'{gene} Expression')
            axes[i].set_aspect('equal')
    
    # Cell type distribution
    sc.pl.spatial(adata, color='cell_type', ax=axes[n_genes], show=False,
                  title='Cell Types')
    axes[n_genes].set_aspect('equal')
    
    # UMAP
    if 'X_umap' in adata.obsm:
        sc.pl.umap(adata, color='cell_type', ax=axes[n_genes + 1], show=False,
                   title='UMAP')
    
    # Quality metrics
    detection_rate = (adata.X > 0).mean(axis=0)
    axes[n_genes + 2].hist(detection_rate, bins=50, alpha=0.7)
    axes[n_genes + 2].set_xlabel('Detection Rate')
    axes[n_genes + 2].set_ylabel('Number of Genes')
    axes[n_genes + 2].set_title('Gene Detection Rate')
    
    # Hide unused subplots
    for i in range(n_genes + 3, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

# Create multi-panel figure
genes = ['CD3D', 'CD19', 'CD68', 'PECAM1', 'CD4', 'CD8A']
fig = create_multi_panel_figure(adata, genes)
plt.show()
```

## Exporting Plots

### Save High-Resolution Plots

```python
# Save plots in different formats
def save_plot(fig, filename, dpi=300):
    """
    Save plot in multiple formats.
    """
    # PNG
    fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches='tight')
    
    # PDF
    fig.savefig(f"{filename}.pdf", bbox_inches='tight')
    
    # SVG
    fig.savefig(f"{filename}.svg", bbox_inches='tight')
    
    print(f"Plots saved as {filename}.png, {filename}.pdf, {filename}.svg")

# Example usage
fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.spatial(adata, color='cell_type', ax=ax, show=False)
save_plot(fig, "cell_type_distribution")
```

## Best Practices

### Color Schemes

1. **Use colorblind-friendly palettes** for accessibility
2. **Consistent color schemes** across related plots
3. **Appropriate color scales** for different data types

### Layout and Design

1. **Clear titles and labels** for all plots
2. **Consistent figure sizes** for similar plot types
3. **Proper aspect ratios** for spatial plots
4. **Adequate spacing** between subplots

### Performance

1. **Downsample large datasets** for interactive plots
2. **Use appropriate file formats** for different use cases
3. **Optimize plot rendering** for large datasets

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce dataset size or use downsampling
2. **Slow rendering**: Use simpler plot types for large datasets
3. **Missing data**: Check for NaN values and handle appropriately
4. **Color issues**: Ensure colorblind-friendly palettes

### Performance Tips

1. Use vectorized operations when possible
2. Cache computed results for repeated plotting
3. Use appropriate data structures for large datasets
4. Consider using specialized plotting libraries for specific use cases

---

*This documentation is under development. More examples and advanced visualization techniques will be added.*
