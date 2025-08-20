# Practical Examples

This page provides comprehensive examples of using SPEX for image segmentation and analysis, based on real-world workflows.

## Complete Segmentation Pipeline

This example demonstrates a complete workflow from image loading to feature extraction and visualization.

### Step 1: Environment Setup

```python
import os
import glob
import numpy as np
from skimage import segmentation
from tifffile import TiffWriter, TiffFile
import matplotlib.pyplot as plt
import seaborn as sns
import spex as sp
```

### Step 2: Load Image

```python
# Load multi-channel microscopy image
img = 'TA459_multipleCores2_Run-4_Point1.tiff'
Image, channel = sp.load_image(img)

print(f"Image loaded with {len(channel)} channels: {channel}")
```

**Output:**
```
Image loaded with 44 channels: ['Au', 'Background', 'Beta_catenin', 'Ca', 'CD11b',
'CD11c', 'CD138', 'CD16', 'CD20', 'CD209', 'CD3', 'CD31', 'CD4', 'CD45', 'CD45RO',
'CD56', 'CD63', 'CD68', 'CD8', 'dsDNA', 'EGFR', 'Fe', 'FoxP3', 'H3K27me3', 'H3K9ac',
'HLA-DR', 'HLA_Class_1', 'IDO', 'Keratin17', 'Keratin6', 'Ki67', 'Lag3', 'MPO', 'Na',
'P', 'p53', 'Pan-Keratin', 'PD-L1', 'PD1', 'phospho-S6', 'Si', 'SMA', 'Ta', 'Vimentin']
```

### Step 3: Background Correction

```python
# Apply background subtraction to specific channels
index = channel.index('Au')
bgcorrect_Image = sp.background_subtract(Image, index, 10, 2)

index = channel.index('Background')
bgcorrect_Image = sp.background_subtract(bgcorrect_Image, index, 10, 2)
```

### Step 4: Optional Denoising

#### Non-Local Means Denoising
```python
# Apply NLM denoising for advanced noise reduction
nlm_Image = sp.nlm_denoise(bgcorrect_Image, 5, 6)
```

#### Median Filtering
```python
# Apply median filtering to specific channels
list_channels = ['dsDNA', 'H3K9ac', 'H3K27me3']

to_denoise = []
for channel_name in list_channels:
    index = channel.index(channel_name)
    to_denoise.append(index)
to_denoise.sort()

median_Image = sp.median_denoise(Image, 4, to_denoise)
```

### Step 5: Cell Segmentation

Choose one of the three segmentation methods:

#### Option A: StarDist Deep Learning

```python
# Prepare channels for segmentation
list_channels = ['dsDNA', 'H3K9ac', 'H3K27me3']

to_merge = []
for channel_name in list_channels:
    index = channel.index(channel_name)
    to_merge.append(index)
to_merge.sort()

# Perform StarDist segmentation
stardist_label = sp.stardist_cellseg(
    median_Image,
    to_merge,
    scaling=1,
    threshold=0.5,
    _min=1,
    _max=98.5
)
```

#### Option B: Cellpose Deep Learning

```python
# Perform Cellpose segmentation
cellpose_label = sp.cellpose_cellseg(
    median_Image,
    to_merge,
    diameter=12,
    scaling=1
)
```

#### Option C: Classic Watershed

```python
# Perform classical watershed segmentation
classic_label = sp.watershed_classic(median_Image, to_merge)
```

### Step 6: Visualization and Saving

```python
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from skimage.segmentation import expand_labels, mark_boundaries
from tifffile import imwrite

# Create nucleus channel for visualization
nuc = np.zeros((Image.shape[1], Image.shape[2]))
for i in to_merge:
    temp = Image[i]
    nuc = temp + nuc

# Save full resolution segmentation labels
imwrite(img.split('.')[0] + '_segmentationlabels.tif', stardist_label,
        photometric='minisblack')

# Create and save visualization
figure(figsize=(8, 8), dpi=80)
boundary = mark_boundaries(np.squeeze(nuc), stardist_label, (0, 0, 255)).astype('uint8')
plt.imsave(img.split('.')[0] + '_segmentation.jpg', boundary)

# Display segmentation
plt.imshow(mark_boundaries(np.squeeze(nuc), stardist_label, (0, 0, 255)))
plt.title('Cell Segmentation Results')
plt.axis('off')
plt.show()
```

### Step 7: Postprocessing

#### Rescue Missed Cells

```python
# Use traditional watershed to rescue cells missed by deep learning
new_label = sp.rescue_cells(Image, to_merge, stardist_label)
```

#### Remove Artifacts

```python
# Remove objects that are too small or too large
newlabel = sp.remove_small_objects(new_label, 8)
newlabel = sp.remove_large_objects(newlabel, 75)
```

#### Cell Expansion

```python
# Expand cell boundaries for better feature extraction
expanded_label = sp.simulate_cell(stardist_label, 10)
```

### Step 8: Feature Extraction

```python
# Extract features from segmented cells
anndata = sp.feature_extraction_adata(Image, expanded_label, channel)

# Convert to DataFrame for analysis
df = anndata.to_df()
df['centroid-0'] = anndata.obs['y_coordinate'].values
df['centroid-1'] = anndata.obs['x_coordinate'].values

print(f"Extracted features for {len(df)} cells")
```

### Step 9: Spatial Visualization

```python
import seaborn as sns

# Add coordinates to DataFrame
df[['x_coordinate', 'y_coordinate']] = anndata.obs[['x_coordinate', 'y_coordinate']]

# Create spatial plot colored by marker expression
g = sns.relplot(
    data=df,
    x='x_coordinate',
    y='y_coordinate',
    hue='CD20',  # Color by CD20 expression
    palette='plasma',
    s=12,
    alpha=0.8
)

g.facet_axis(0, 0).invert_yaxis()
plt.title('Spatial Distribution of CD20 Expression')
plt.show()
```

### Step 10: Data Export

```python
import anndata as ad
from anndata import AnnData
import scanpy as sc
from scipy.stats import zscore
import pandas as pd

# Prepare coordinates
coordinates = []
for k in range(len(df)):
    coordinates.append([df.loc[df.index[k], 'centroid-1'],
                       df.loc[df.index[k], 'centroid-0']])
coordinates = np.array(coordinates)

# Format cell labels
celltype = pd.DataFrame({'label': anndata.obs['Cell_ID'].astype(str)})
celltype['label'] = celltype['label'].astype('category')
celltype = celltype.rename(columns={'label': 'Cell_ID'})

# Prepare expression data
expression_data = df[channel]

# Create AnnData object
adata = AnnData(expression_data, obsm={"spatial": coordinates})
adata.obs['Cell_ID'] = [str(i) for i in celltype['Cell_ID'].tolist()]
adata.layers["zscored"] = expression_data.apply(zscore)

# Save to disk
adata.write(img.split('.')[0] + '.h5ad', compression="gzip")
print(f"Saved AnnData object with {adata.n_obs} cells and {adata.n_vars} markers")
```

## Batch Processing Example

Process multiple images automatically:

```python
import os
import glob

# Get all TIFF files in current directory
files = glob.glob('*.tiff', recursive=False)

for image in files:
    print(f"Processing {image}...")

    # Load Image
    Image, channel = sp.load_image(image)
    print(f"Channels: {channel}")

    # Denoise Image
    list_channels = ['dsDNA', 'H3K9ac', 'H3K27me3']
    to_denoise = []
    for channel_name in list_channels:
        index = channel.index(channel_name)
        to_denoise.append(index)
    to_denoise.sort()

    median_Image = sp.median_denoise(Image, 5, to_denoise)

    # Run Segmentation
    to_merge = []
    for channel_name in list_channels:
        index = channel.index(channel_name)
        to_merge.append(index)
    to_merge.sort()

    # Perform segmentation with multiple methods
    stardist_label = sp.stardist_cellseg(median_Image, to_merge, 1, 0.5, 1, 98.5)
    cellpose_label = sp.cellpose_cellseg(median_Image, to_merge, 12, 1)

    # Postprocessing
    new_label = sp.rescue_cells(Image, to_merge, stardist_label)
    expanded_label = sp.simulate_cell(new_label, 10)

    # Extract Features
    anndata = sp.feature_extraction_adata(Image, expanded_label, channel)
    df = anndata.to_df()
    df['centroid-0'] = anndata.obs['y_coordinate'].values
    df['centroid-1'] = anndata.obs['x_coordinate'].values

    # Save Results
    csvname = image.split(".tiff")[0] + '_stardist.csv'
    df.to_csv(csvname, index=False)

    print(f"Completed processing {image}")
```

## Clustering and Analysis

After feature extraction, you can perform clustering and analysis:

```python
import spex as sp
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

# Load AnnData object
adata = ad.read_h5ad("adata_ph.h5ad")

# Perform PhenoGraph clustering
adata = sp.phenograph_cluster(adata, channel_names=["CD3", "CD8", "CD20"], knn=2)

# Visualize clusters
umap_coords = adata.obsm['X_umap']
clusters = adata.obs['cluster_phenograph'].astype(int)

plt.figure(figsize=(7, 6))
scatter = plt.scatter(
    umap_coords[:, 0],
    umap_coords[:, 1],
    c=clusters,
    cmap="tab20",
    s=20
)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("PhenoGraph Clusters")

# Add cluster centers
for cluster_id in np.unique(clusters):
    mask = clusters == cluster_id
    x_mean, y_mean = umap_coords[mask, 0].mean(), umap_coords[mask, 1].mean()
    plt.text(x_mean, y_mean, str(cluster_id), fontsize=10, weight='bold')

plt.colorbar(scatter, label="Cluster")
plt.show()
```

## Advanced Clustering Workflow

### Step 1: Data Preprocessing

```python
import spex as sp
import anndata as ad
import scanpy as sc

# Load AnnData object
adata = sp.load_anndata(path="your_data.h5ad")

# Preprocess data for clustering
adata = sp.preprocess(
    adata,
    scale_max=10,
    size_factor=None,
    do_QC=True
)

print(f"Preprocessed data shape: {adata.shape}")
print(f"Highly variable genes: {adata.var.highly_variable.sum()}")
```

### Step 2: Dimensionality Reduction

```python
# Reduce dimensionality using PCA and UMAP
adata = sp.reduce_dimensionality(
    adata,
    n_components=50,
    method='pca'
)

# Compute neighborhood graph
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

# Run UMAP
sc.tl.umap(adata, min_dist=0.5)
```

### Step 3: Clustering with Spatial Information

```python
# Perform clustering with spatial weights
adata = sp.cluster(
    adata,
    spatial_weight=0.5,  # Weight for spatial neighbors
    resolution=1.0,      # Clustering resolution
    method='leiden'      # Clustering algorithm
)

print(f"Number of clusters: {adata.obs['leiden'].nunique()}")
```

### Step 4: Differential Expression Analysis

```python
# Find marker genes for each cluster
adata = sp.differential_expression(
    adata,
    groupby='leiden',
    method='wilcoxon'
)

# Get top markers for each cluster
markers = adata.uns['rank_genes_groups']
print("Top markers per cluster:")
for cluster in adata.obs['leiden'].unique():
    top_genes = markers['names'][cluster][:5]
    print(f"Cluster {cluster}: {list(top_genes)}")
```

## AnnData Workflow Examples

### Loading and Combining Multiple Datasets

```python
import spex as sp
import glob

# Load multiple AnnData files
files = glob.glob("path/to/data/*.h5ad")
combined_data = sp.load_anndata(files=files)

print(f"Combined dataset shape: {combined_data['adata'].shape}")
print(f"Files included: {combined_data['adata'].obs['filename'].unique()}")
```

### Working with Spatial Coordinates

```python
import spex as sp
import pandas as pd

# Extract spatial coordinates from AnnData
coordinates = pd.DataFrame({
    'x': adata.obsm['spatial'][:, 0],
    'y': adata.obsm['spatial'][:, 1],
    'cluster': adata.obs['leiden']
})

# Create spatial visualization
import seaborn as sns
import matplotlib.pyplot as plt

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

### Pathway Analysis

```python
# Annotate clusters based on marker genes
adata = sp.annotate_clusters(
    adata,
    groupby='leiden',
    reference_markers={
        'T_cells': ['CD3', 'CD8', 'CD4'],
        'B_cells': ['CD20', 'CD19'],
        'Macrophages': ['CD68', 'CD11b'],
        'Endothelial': ['CD31', 'Vimentin']
    }
)

# Analyze pathway enrichment
pathway_results = sp.analyze_pathways(
    adata,
    groupby='leiden',
    database='reactome'
)

print("Pathway enrichment results:")
for cluster in pathway_results.keys():
    print(f"\nCluster {cluster}:")
    top_pathways = pathway_results[cluster].head(3)
    for _, row in top_pathways.iterrows():
        print(f"  {row['pathway']}: p-value={row['p_value']:.3e}")
```

### Niche Analysis

```python
# Perform niche analysis to identify spatial patterns
niche_results = sp.niche(
    adata,
    groupby='leiden',
    spatial_key='spatial',
    radius=50
)

print("Niche analysis results:")
for niche_type, cells in niche_results.items():
    print(f"{niche_type}: {len(cells)} cells")
```

## Complete Spatial Analysis Pipeline

```python
import spex as sp
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

# Complete workflow from raw data to spatial analysis
def run_spatial_analysis(data_path, output_path):
    """
    Complete spatial analysis pipeline.
    
    Parameters
    ----------
    data_path : str
        Path to AnnData file
    output_path : str
        Path to save results
    """
    
    # 1. Load data
    print("Loading data...")
    adata = sp.load_anndata(path=data_path)['adata']
    
    # 2. Preprocess
    print("Preprocessing...")
    adata = sp.preprocess(adata, scale_max=10, do_QC=True)
    
    # 3. Dimensionality reduction
    print("Reducing dimensionality...")
    adata = sp.reduce_dimensionality(adata, n_components=50)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    sc.tl.umap(adata)
    
    # 4. Clustering
    print("Clustering...")
    adata = sp.cluster(adata, spatial_weight=0.3, resolution=1.0)
    
    # 5. Differential expression
    print("Finding marker genes...")
    adata = sp.differential_expression(adata, groupby='leiden')
    
    # 6. Pathway analysis
    print("Analyzing pathways...")
    adata = sp.annotate_clusters(adata, groupby='leiden')
    pathway_results = sp.analyze_pathways(adata, groupby='leiden')
    
    # 7. Niche analysis
    print("Performing niche analysis...")
    niche_results = sp.niche(adata, groupby='leiden')
    
    # 8. Save results
    print("Saving results...")
    adata.write(f"{output_path}_processed.h5ad")
    
    # 9. Create visualizations
    create_visualizations(adata, pathway_results, niche_results, output_path)
    
    return adata

def create_visualizations(adata, pathway_results, niche_results, output_path):
    """Create comprehensive visualizations."""
    
    # UMAP plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    sc.pl.umap(adata, color='leiden', show=False, title='Clusters')
    
    plt.subplot(1, 3, 2)
    sc.pl.umap(adata, color='spatial', show=False, title='Spatial')
    
    plt.subplot(1, 3, 3)
    sc.pl.umap(adata, color='annotation', show=False, title='Annotation')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}_umap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Spatial plot
    plt.figure(figsize=(10, 8))
    coordinates = pd.DataFrame({
        'x': adata.obsm['spatial'][:, 0],
        'y': adata.obsm['spatial'][:, 1],
        'cluster': adata.obs['leiden']
    })
    
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
    plt.savefig(f"{output_path}_spatial.png", dpi=300, bbox_inches='tight')
    plt.close()

# Run the complete pipeline
if __name__ == "__main__":
    adata = run_spatial_analysis("input_data.h5ad", "results")
    print("Analysis complete!")
```

## Advanced Spatial Analysis with CLQ

### Co-Localization Quotient (CLQ) Analysis

CLQ analysis helps identify spatial relationships between different cell types:

```python
import spex as sp
import anndata as ad
import pandas as pd

# Load your AnnData object with spatial coordinates
adata = ad.read_h5ad("your_data.h5ad")

# Ensure spatial coordinates are available
if 'spatial' not in adata.obsm:
    adata.obsm['spatial'] = adata.obs[['x_coordinate', 'y_coordinate']].to_numpy()

# Perform CLQ analysis
adata_out, results = sp.CLQ_vec_numba(
    adata,
    clust_col='leiden',      # Column with cluster labels
    radius=50,              # Analysis radius
    n_perms=1000            # Number of permutations for significance testing
)

# Access results
print("Global CLQ matrix:")
print(results['global_clq'])

print("\nPermutation test results:")
print(results['permute_test'])

# Local CLQ values for each cell
local_clq = adata_out.obsm['local_clq']
neighborhood_vectors = adata_out.obsm['NCV']

print(f"Local CLQ shape: {local_clq.shape}")
print(f"Neighborhood vectors shape: {neighborhood_vectors.shape}")
```

### CLQ Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot global CLQ matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    results['global_clq'],
    annot=True,
    cmap='RdBu_r',
    center=1.0,
    square=True
)
plt.title('Global Co-Localization Quotient')
plt.xlabel('Cell Type')
plt.ylabel('Cell Type')
plt.show()

# Plot permutation test results
plt.figure(figsize=(8, 6))
sns.heatmap(
    results['permute_test'],
    annot=True,
    cmap='RdBu_r',
    center=0.5,
    square=True
)
plt.title('CLQ Permutation Test P-values')
plt.xlabel('Cell Type')
plt.ylabel('Cell Type')
plt.show()
```

## Complete Segmentation and Analysis Workflow

### Step-by-Step Complete Pipeline

```python
import spex as sp
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tifffile import imwrite
import pandas as pd

def complete_analysis_pipeline(image_path, output_dir):
    """
    Complete analysis pipeline from image to spatial analysis.
    
    Parameters
    ----------
    image_path : str
        Path to multi-channel microscopy image
    output_dir : str
        Directory to save results
    """
    
    # Step 1: Load and preprocess image
    print("Loading image...")
    Image, channel = sp.load_image(image_path)
    print(f"Loaded {len(channel)} channels: {channel}")
    
    # Step 2: Background correction
    print("Applying background correction...")
    bg_channels = ['Au', 'Background']
    for bg_channel in bg_channels:
        if bg_channel in channel:
            index = channel.index(bg_channel)
            Image = sp.background_subtract(Image, index, 10, 2)
    
    # Step 3: Denoising
    print("Applying denoising...")
    denoise_channels = ['dsDNA', 'H3K9ac', 'H3K27me3']
    to_denoise = [channel.index(ch) for ch in denoise_channels if ch in channel]
    if to_denoise:
        Image = sp.median_denoise(Image, 4, to_denoise)
    
    # Step 4: Cell segmentation
    print("Performing cell segmentation...")
    seg_channels = ['dsDNA', 'H3K9ac', 'H3K27me3']
    to_merge = [channel.index(ch) for ch in seg_channels if ch in channel]
    
    # Try multiple segmentation methods
    segmentation_results = {}
    
    # StarDist
    try:
        stardist_label = sp.stardist_cellseg(
            Image, to_merge, scaling=1, threshold=0.5, _min=1, _max=98.5
        )
        segmentation_results['stardist'] = stardist_label
        print(f"StarDist detected {stardist_label.max()} cells")
    except Exception as e:
        print(f"StarDist failed: {e}")
    
    # Cellpose
    try:
        cellpose_label = sp.cellpose_cellseg(Image, to_merge, diameter=12, scaling=1)
        segmentation_results['cellpose'] = cellpose_label
        print(f"Cellpose detected {cellpose_label.max()} cells")
    except Exception as e:
        print(f"Cellpose failed: {e}")
    
    # Watershed
    try:
        watershed_label = sp.watershed_classic(Image, to_merge)
        segmentation_results['watershed'] = watershed_label
        print(f"Watershed detected {watershed_label.max()} cells")
    except Exception as e:
        print(f"Watershed failed: {e}")
    
    # Step 5: Post-processing
    print("Post-processing segmentation...")
    best_label = None
    best_method = None
    
    for method, label in segmentation_results.items():
        # Rescue missed cells
        rescued_label = sp.rescue_cells(Image, to_merge, label)
        
        # Remove artifacts
        cleaned_label = sp.remove_small_objects(rescued_label, 8)
        cleaned_label = sp.remove_large_objects(cleaned_label, 75)
        
        # Expand cell boundaries
        expanded_label = sp.simulate_cell(cleaned_label, 10)
        
        if best_label is None or expanded_label.max() > best_label.max():
            best_label = expanded_label
            best_method = method
    
    print(f"Using {best_method} segmentation with {best_label.max()} cells")
    
    # Step 6: Feature extraction
    print("Extracting features...")
    anndata = sp.feature_extraction_adata(Image, best_label, channel)
    
    # Step 7: Clustering
    print("Performing clustering...")
    marker_channels = ['CD3', 'CD8', 'CD20', 'CD68']
    available_markers = [ch for ch in marker_channels if ch in channel]
    
    if available_markers:
        anndata = sp.phenograph_cluster(
            anndata,
            channel_names=available_markers,
            knn=30,
            transformation='arcsin',
            scaling='z-score'
        )
    
    # Step 8: Spatial analysis
    print("Performing spatial analysis...")
    
    # Add spatial coordinates
    coords = np.column_stack([
        anndata.obs['x_coordinate'].values,
        anndata.obs['y_coordinate'].values
    ])
    anndata.obsm['spatial'] = coords
    
    # CLQ analysis
    try:
        anndata_out, clq_results = sp.CLQ_vec_numba(
            anndata,
            clust_col='cluster_phenograph',
            radius=50,
            n_perms=100
        )
        print("CLQ analysis completed")
    except Exception as e:
        print(f"CLQ analysis failed: {e}")
    
    # Step 9: Save results
    print("Saving results...")
    
    # Save AnnData object
    output_name = image_path.split('/')[-1].split('.')[0]
    anndata.write(f"{output_dir}/{output_name}_analysis.h5ad")
    
    # Save segmentation labels
    imwrite(f"{output_dir}/{output_name}_segmentation.tif", best_label)
    
    # Save feature table
    df = anndata.to_df()
    df['x_coordinate'] = anndata.obs['x_coordinate']
    df['y_coordinate'] = anndata.obs['y_coordinate']
    df['cluster'] = anndata.obs.get('cluster_phenograph', 'unknown')
    df.to_csv(f"{output_dir}/{output_name}_features.csv", index=False)
    
    # Step 10: Create visualizations
    print("Creating visualizations...")
    create_comprehensive_plots(anndata, best_label, Image, channel, output_dir, output_name)
    
    return anndata, best_label

def create_comprehensive_plots(anndata, labels, Image, channel, output_dir, output_name):
    """Create comprehensive visualization plots."""
    
    # 1. Segmentation visualization
    plt.figure(figsize=(12, 4))
    
    # Original image (nucleus channel)
    nucleus_ch = [i for i, ch in enumerate(channel) if 'DNA' in ch or 'DAPI' in ch]
    if nucleus_ch:
        nucleus_img = Image[nucleus_ch[0]]
    else:
        nucleus_img = Image[0]  # First channel as fallback
    
    plt.subplot(1, 3, 1)
    plt.imshow(nucleus_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Segmentation overlay
    plt.subplot(1, 3, 2)
    from skimage.segmentation import mark_boundaries
    boundaries = mark_boundaries(nucleus_img, labels, color=(1, 0, 0))
    plt.imshow(boundaries)
    plt.title('Segmentation')
    plt.axis('off')
    
    # Cluster visualization
    plt.subplot(1, 3, 3)
    if 'cluster_phenograph' in anndata.obs.columns:
        coords = anndata.obsm['spatial']
        clusters = anndata.obs['cluster_phenograph'].astype('category')
        
        scatter = plt.scatter(
            coords[:, 0], coords[:, 1],
            c=clusters.cat.codes,
            cmap='tab20',
            s=20,
            alpha=0.8
        )
        plt.title('Cell Clusters')
        plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{output_name}_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. UMAP plot
    if 'X_umap' in anndata.obsm:
        plt.figure(figsize=(8, 6))
        umap_coords = anndata.obsm['X_umap']
        clusters = anndata.obs['cluster_phenograph'].astype('category')
        
        scatter = plt.scatter(
            umap_coords[:, 0], umap_coords[:, 1],
            c=clusters.cat.codes,
            cmap='tab20',
            s=20,
            alpha=0.8
        )
        plt.title('UMAP Clustering')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.colorbar(scatter, label='Cluster')
        plt.savefig(f"{output_dir}/{output_name}_umap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Spatial distribution
    plt.figure(figsize=(10, 8))
    coords = anndata.obsm['spatial']
    clusters = anndata.obs['cluster_phenograph'].astype('category')
    
    scatter = plt.scatter(
        coords[:, 0], coords[:, 1],
        c=clusters.cat.codes,
        cmap='tab20',
        s=20,
        alpha=0.8
    )
    plt.title('Spatial Distribution of Clusters')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig(f"{output_dir}/{output_name}_spatial.png", dpi=300, bbox_inches='tight')
    plt.close()

# Example usage
if __name__ == "__main__":
    # Run complete analysis
    adata, labels = complete_analysis_pipeline(
        "TA459_multipleCores2_Run-4_Point1.tiff",
        "results"
    )
    print("Analysis complete!")
```

## Tips and Best Practices

!!! tip "Channel Selection"
    Choose channels with clear cell boundaries for segmentation. Nuclear markers like DAPI, dsDNA, or histone modifications work well.

!!! tip "Parameter Tuning"
    - Adjust `diameter` parameter based on your cell size
    - Use `threshold` values between 0.3-0.7 for optimal results
    - Scale images appropriately for your data

!!! tip "Performance"
    - Use GPU acceleration when available for faster processing
    - Consider downsampling large images for initial testing
    - Batch processing is more efficient than processing individual files

!!! warning "Memory Usage"
    Large multi-channel images can consume significant memory. Consider processing in chunks for very large datasets.

!!! tip "Clustering Best Practices"
    - Use `spatial_weight` parameter to incorporate spatial information in clustering
    - Adjust `resolution` parameter to control number of clusters
    - Consider using PhenoGraph for large datasets with complex cell populations

!!! tip "AnnData Workflow"
    - Always save intermediate results during long workflows
    - Use `adata.raw` to preserve original data
    - Leverage `adata.layers` for storing different data transformations
    - Use `adata.uns` for storing metadata and analysis results

!!! tip "Spatial Analysis"
    - Ensure spatial coordinates are properly scaled
    - Use appropriate radius for niche analysis based on your tissue type
    - Consider batch effects when combining multiple datasets
    - Validate clustering results with known marker genes

!!! tip "CLQ Analysis"
    - Use radius parameter based on your tissue density and cell size
    - Increase n_perms for more robust statistical testing
    - Interpret CLQ values: >1 indicates attraction, <1 indicates avoidance
    - Consider cell type frequencies when interpreting results

!!! tip "Error Handling"
    - Always wrap segmentation methods in try-except blocks
    - Check for required dependencies before running analysis
    - Validate input data formats and dimensions
    - Use fallback methods when primary methods fail
