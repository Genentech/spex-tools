# Frequently Asked Questions (FAQ)

This FAQ addresses common questions about SPEX library usage, troubleshooting, and best practices.

## General Questions

### What is SPEX?

SPEX is a comprehensive Python library for spatial transcriptomics analysis. It provides tools for:
- **Image segmentation** and cell detection
- **Feature extraction** from spatial data
- **Clustering analysis** with spatial awareness
- **Spatial analysis** including co-localization and niche analysis
- **Data preprocessing** and quality control

### What data formats does SPEX support?

SPEX supports multiple data formats:
- **Expression data**: H5AD files, CSV, TSV
- **Spatial coordinates**: CSV files with x,y coordinates
- **Images**: TIFF, PNG, JPEG formats
- **Segmentation masks**: TIFF, PNG formats

### How do I install SPEX?

```bash
pip install spex
```

For development installation:
```bash
git clone https://github.com/your-repo/spex.git
cd spex
pip install -e .
```

## Data Loading and Preprocessing

### How do I load my spatial transcriptomics data?

```python
import spex

# Load data with spatial coordinates
adata = spex.load_anndata(
    "expression_matrix.h5ad",
    spatial_data="spatial_coordinates.csv",
    image_path="tissue_image.tif"
)
```

### What should my spatial coordinates file look like?

Your spatial coordinates CSV should have this format:
```csv
cell_id,x,y
cell_1,100,200
cell_2,150,250
cell_3,200,300
```

### How do I handle missing spatial coordinates?

```python
# Load without spatial data
adata = spex.load_anndata("expression_matrix.h5ad")

# Add spatial coordinates later
spatial_coords = pd.read_csv("spatial_coordinates.csv", index_col=0)
adata.obsm['spatial'] = spatial_coords.values
```

### What preprocessing steps are recommended?

```python
# Basic preprocessing
adata = spex.preprocess(
    adata,
    min_genes=10,
    min_cells=3,
    max_counts_per_cell=5000,
    normalize=True,
    log_transform=True
)

# Dimensionality reduction
adata = spex.reduce_dimensionality(
    adata,
    method='pca',
    n_components=50
)
```

## Image Segmentation

### Which segmentation method should I use?

**Cellpose** (recommended for most cases):
- Works well with various cell types
- Automatic parameter detection
- Good for fluorescence images

**StarDist**:
- Excellent for nuclear segmentation
- Good for brightfield images
- Requires more parameter tuning

**Watershed**:
- Fast and simple
- Good for well-separated cells
- Less accurate for complex images

### How do I download Cellpose models?

```python
# Download default models
spex.download_cellpose_models()

# Download specific model
spex.download_cellpose_models(model_type='cyto')
```

### My segmentation is poor. What should I do?

1. **Preprocess the image**:
```python
image = spex.load_image("tissue.tif")
image_processed = spex.background_subtract(image)
image_processed = spex.median_denoise(image_processed)
```

2. **Adjust Cellpose parameters**:
```python
segmentation_mask = spex.cellpose_cellseg(
    image_processed,
    model_type='cyto',
    diameter=20,  # Manual diameter
    flow_threshold=0.3,  # Lower threshold
    cellprob_threshold=-2  # More permissive
)
```

3. **Post-process the segmentation**:
```python
segmentation_mask = spex.remove_small_objects(segmentation_mask, min_size=50)
segmentation_mask = spex.remove_large_objects(segmentation_mask, max_size=1000)
segmentation_mask = spex.rescue_cells(segmentation_mask, image_processed)
```

### How do I visualize segmentation results?

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(image_processed, cmap='gray')
axes[1].set_title('Preprocessed Image')
axes[1].axis('off')

axes[2].imshow(segmentation_mask, cmap='tab20')
axes[2].set_title('Cell Segmentation')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

## Clustering Analysis

### Which clustering method should I use?

**Leiden** (recommended):
- Fast and scalable
- Good for large datasets
- Consistent results

**Louvain**:
- Classic method
- Good for smaller datasets
- More sensitive to resolution parameter

**Phenograph**:
- Advanced method
- Good for complex datasets
- Slower but more robust

### How do I choose the right resolution parameter?

```python
# Test different resolutions
resolutions = [0.1, 0.3, 0.5, 0.7, 1.0]
results = {}

for res in resolutions:
    adata = spex.cluster(adata, method='leiden', resolution=res)
    results[res] = len(adata.obs['leiden'].unique())

print("Number of clusters per resolution:")
for res, n_clusters in results.items():
    print(f"Resolution {res}: {n_clusters} clusters")
```

### How do I validate clustering results?

```python
from sklearn.metrics import silhouette_score

# Calculate silhouette score
silhouette_avg = silhouette_score(adata.obsm['X_pca'], adata.obs['leiden'])
print(f"Silhouette Score: {silhouette_avg:.3f}")

# Visualize clustering
sc.pl.umap(adata, color='leiden', show=False)
plt.title(f'Clustering (Silhouette: {silhouette_avg:.3f})')
plt.show()
```

### How do I find marker genes for clusters?

```python
# Find marker genes
marker_genes = spex.differential_expression(
    adata,
    groupby='leiden',
    method='wilcoxon'
)

# Get top markers per cluster
for cluster in adata.obs['leiden'].unique():
    cluster_markers = marker_genes[marker_genes['cluster'] == cluster]
    top_markers = cluster_markers.head(5)['gene'].tolist()
    print(f"Cluster {cluster}: {', '.join(top_markers)}")
```

## Spatial Analysis

### What is Co-Localization Quotient (CLQ)?

CLQ measures the spatial association between different cell types:
- **CLQ > 1**: Cell types are co-localized
- **CLQ = 1**: Random spatial distribution
- **CLQ < 1**: Cell types avoid each other

### How do I perform CLQ analysis?

```python
# Calculate CLQ
clq_results = spex.CLQ_vec_numba(
    adata,
    cluster_key='leiden',
    spatial_key='spatial',
    n_permutations=1000
)

# Visualize results
import seaborn as sns
clq_matrix = clq_results.pivot(index='cluster1', columns='cluster2', values='clq')
sns.heatmap(clq_matrix, annot=True, cmap='RdBu_r', center=1)
plt.title('Co-Localization Quotient')
plt.show()
```

### What is niche analysis?

Niche analysis identifies spatial microenvironments where specific cell types are enriched:

```python
# Perform niche analysis
niche_results = spex.niche(
    adata,
    cluster_key='leiden',
    spatial_key='spatial',
    radius=100,
    min_cells=5
)

print("Niche analysis results:")
print(niche_results.head(10))
```

### How do I analyze spatial autocorrelation?

```python
# Calculate Moran's I for a gene
gene_expression = adata[:, 'gene_name'].X.flatten()
coords = adata.obsm['spatial']

# Calculate spatial weights
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=10).fit(coords)
distances, indices = nbrs.kneighbors(coords)

# Moran's I calculation (simplified)
def calculate_morans_i(expression, indices, distances):
    # Implementation here
    pass

moran_i = calculate_morans_i(gene_expression, indices, distances)
print(f"Moran's I: {moran_i:.3f}")
```

## Performance and Optimization

### My analysis is slow. How can I speed it up?

1. **Use appropriate data types**:
```python
adata.X = adata.X.astype(np.float32)
```

2. **Reduce dataset size**:
```python
sc.pp.subsample(adata, fraction=0.5, random_state=42)
```

3. **Use parallel processing**:
```python
import multiprocessing as mp

def process_chunk(chunk_data):
    # Your processing function
    pass

with mp.Pool(processes=4) as pool:
    results = pool.map(process_chunk, data_chunks)
```

4. **Reduce permutations for spatial analysis**:
```python
clq_results = spex.CLQ_vec_numba(
    adata,
    cluster_key='leiden',
    n_permutations=100  # Reduced from 1000
)
```

### How do I handle large datasets?

1. **Process in chunks**:
```python
chunk_size = 1000
for i in range(0, len(adata), chunk_size):
    chunk = adata[i:i+chunk_size]
    # Process chunk
```

2. **Use memory-efficient operations**:
```python
# Clear unnecessary data
del large_variable
import gc
gc.collect()
```

3. **Use caching**:
```python
import joblib

@joblib.cache
def expensive_computation(data):
    # Your computation here
    return result
```

## Troubleshooting

### I get a "No module named 'spex'" error

This means SPEX is not installed. Install it with:
```bash
pip install spex
```

### My segmentation fails with "CUDA out of memory"

1. **Reduce image size**:
```python
from skimage.transform import resize
image = resize(image, (image.shape[0]//2, image.shape[1]//2))
```

2. **Use CPU instead of GPU**:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU
```

3. **Process smaller regions**:
```python
# Split image into tiles
tile_size = 512
for i in range(0, image.shape[0], tile_size):
    for j in range(0, image.shape[1], tile_size):
        tile = image[i:i+tile_size, j:j+tile_size]
        # Process tile
```

### My clustering produces too many/few clusters

Adjust the resolution parameter:
```python
# For fewer clusters
adata = spex.cluster(adata, method='leiden', resolution=0.1)

# For more clusters
adata = spex.cluster(adata, method='leiden', resolution=1.0)
```

### My spatial analysis fails

1. **Check spatial coordinates**:
```python
print("Spatial coordinates shape:", adata.obsm['spatial'].shape)
print("Coordinate range:", adata.obsm['spatial'].min(), adata.obsm['spatial'].max())
```

2. **Remove duplicate coordinates**:
```python
coords = adata.obsm['spatial']
unique_coords = np.unique(coords, axis=0)
if len(unique_coords) < len(coords):
    print("Warning: Duplicate coordinates found")
```

3. **Check for NaN values**:
```python
if np.isnan(adata.obsm['spatial']).any():
    print("Warning: NaN values in spatial coordinates")
```

### My quality control removes too many cells

Adjust QC parameters:
```python
# More permissive parameters
qc_params = {
    'min_counts': 50,  # Lower minimum
    'max_counts': 15000,  # Higher maximum
    'min_genes': 5,  # Lower minimum
    'max_genes': 8000,  # Higher maximum
    'max_mito_ratio': 0.3,  # Higher threshold
    'min_cells_per_gene': 2  # Lower minimum
}
```

## Best Practices

### Data Organization

1. **Use consistent file naming**:
```
sample_001/
├── expression.h5ad
├── spatial_coordinates.csv
├── tissue_image.tif
└── metadata.json
```

2. **Document your analysis**:
```python
# Save analysis parameters
analysis_params = {
    'segmentation': {'method': 'cellpose', 'diameter': 20},
    'clustering': {'method': 'leiden', 'resolution': 0.5},
    'spatial_analysis': {'n_permutations': 1000}
}

import json
with open('analysis_parameters.json', 'w') as f:
    json.dump(analysis_params, f, indent=2)
```

### Reproducibility

1. **Set random seeds**:
```python
import numpy as np
np.random.seed(42)
```

2. **Save intermediate results**:
```python
# Save at key steps
adata.write('preprocessed_data.h5ad')
adata.write('clustered_data.h5ad')
adata.write('final_results.h5ad')
```

3. **Use version control**:
```bash
git add .
git commit -m "Analysis step: clustering completed"
```

### Visualization

1. **Use consistent color schemes**:
```python
# Define color palette
cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
```

2. **Save high-quality figures**:
```python
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
```

3. **Add proper labels and titles**:
```python
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('Cell Clustering Results')
```

## Getting Help

### Where can I find more examples?

- Check the `docs/examples/` directory
- Look at the Jupyter notebooks in `notebooks/`
- Visit the documentation website

### How do I report a bug?

1. Check if the issue is already reported
2. Create a minimal reproducible example
3. Include error messages and system information
4. Submit an issue on GitHub

### How do I contribute to SPEX?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Where can I ask questions?

- GitHub Issues for bug reports
- GitHub Discussions for questions
- Email the maintainers for specific issues

This FAQ covers the most common questions about SPEX. If you don't find your answer here, please check the documentation or ask in the GitHub discussions.
