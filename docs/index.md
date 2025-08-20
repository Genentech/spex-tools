# SPEX - Spatial Omics Analysis Library

SPEX is a spatial transcriptomics analysis library that implements methods developed for the [SPEX](https://www.biorxiv.org/content/10.1101/2022.08.22.504841v2) software platform. The library enables users to apply state-of-the-art tissue segmentation techniques on their own image data.



## 🚀 Key Features

### Image Segmentation
- **Cellpose** - Deep learning for cell segmentation
- **StarDist** - Star-convex object-based segmentation
- **Watershed** - Classical watershed segmentation
- **Filtering** - Median filtering and non-local means
- **Preprocessing** - Background subtraction and noise removal

### Spatial Transcriptomics Analysis
- **Clustering** - PhenoGraph and other clustering algorithms
- **Niche Analysis** - Cell niche identification and interactions
- **Differential Expression** - Analysis of differences between groups
- **Pathway Analysis** - Cluster annotation and signaling pathway analysis
- **CLQ Analysis** - Co-localization quotient for spatial relationships

### Utilities
- **Data Loading** - Support for OME-TIFF, OME-ZARR, AnnData formats
- **Preprocessing** - Normalization and batch correction
- **Visualization** - Tools for displaying results

## 📦 Installation

### System Requirements

Before using OpenCV-related features, install the required system libraries:

```bash
sudo apt install -y libgl1-mesa-glx libjpeg-dev zlib1g-dev libpng-dev libgl1 libfftw3-dev build-essential python3-dev
```

### Package Installation

```bash
# Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel packaging
pip install pytest

# Install the package locally
pip install .
```

## 📚 Documentation

- **[Image Segmentation](segmentation.md)** - Detailed segmentation documentation
- **[Clustering and Spatial Transcriptomics](clustering.md)** - Clustering and spatial analysis
- **[Practical Examples](examples.md)** - Complete workflows and tutorials
- **[API Reference](api.md)** - Complete API documentation

## 📂 Examples

Use the methods directly in your own analysis pipelines. Example notebooks are available:

- ▶️ **Google Colab**
  [Run on Colab](https://colab.research.google.com/drive/1Qlc3pgN9SlZPUa8kUBu0ePrLG5dj2rd8?usp=sharing)

- 🖥️ **JupyterLab Server**
  [View on Server](http://65.108.226.226:2266/lab/workspaces/auto-j/tree/work/notebook/Segmentation.ipynb)
  password "spexspex"

Notebooks include:
- Model downloading (in case Cellpose server access fails)
- Visualization examples
- End-to-end segmentation pipelines

## ⚙️ Compatibility

- ✅ Tested with **Python 3.11**
- ⚠️ Compatibility with other Python versions is not guaranteed
- ⚙️ Includes integrated **Cellpose** support, with fallback model handling

## 🔧 Quick Start

```python
from spex import load_image, cellpose_cellseg

# Load an image
array, channels = load_image("path/to/image.ome.tiff")

# Perform segmentation
labels = cellpose_cellseg(array, seg_channels=[0], diameter=30, scaling=1)

print(f"Detected {labels.max()} cells")
```

## 🎯 Getting Started

1. **Install SPEX** - Follow the installation instructions above
2. **Load your data** - Use `load_image()` for microscopy images
3. **Segment cells** - Choose from Cellpose, StarDist, or Watershed
4. **Extract features** - Get per-cell measurements
5. **Analyze spatially** - Perform clustering and spatial analysis

## 📖 What's New

- **Complete segmentation pipelines** - From image loading to feature extraction
- **Advanced clustering** - PhenoGraph with spatial awareness
- **CLQ analysis** - Co-localization quotient for spatial relationships
- **Comprehensive documentation** - Step-by-step tutorials and examples
- **Error handling** - Robust pipelines with fallback methods