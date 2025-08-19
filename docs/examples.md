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
