# SPEX - Spatial Omics Analysis Library

SPEX is a comprehensive spatial omics analysis library supporting both spatial proteomics and spatial transcriptomics. It implements state-of-the-art methods for tissue segmentation, clustering, and spatial analysis. The library enables researchers to apply advanced image processing and spatial analysis techniques to their own microscopy data, providing a complete pipeline from image loading to biological insights.



## 🚀 Key Features

### Image Segmentation
- **Cellpose** - Deep learning for cell segmentation
- **StarDist** - Star-convex object-based segmentation
- **Watershed** - Classical watershed segmentation
- **Filtering** - Median filtering and non-local means
- **Preprocessing** - Background subtraction and noise removal

### Spatial Proteomics Analysis
- **Image Preprocessing** - Background subtraction and noise removal methods
- **Segmentation** - Cellpose, StarDist, and Watershed for protein marker images
- **Post-processing** - Cell rescue, filtering, and feature extraction
- **Clustering** - PhenoGraph and other clustering algorithms for protein profiles
- **CLQ Analysis** - Co-localization quotient for spatial protein relationships
- **Niche Analysis** - Protein-based cell niche identification and interactions

### Spatial Transcriptomics Analysis
- **Clustering** - PhenoGraph and other clustering algorithms
- **Niche Analysis** - Cell niche identification and interactions
- **Differential Expression** - Analysis of differences between groups
- **Pathway Analysis** - Cluster annotation and signaling pathway analysis
- **CLQ Analysis** - Co-localization quotient for spatial relationships

### Utilities
- **Data Loading** - Support for OME-TIFF, OME-ZARR, AnnData formats
- **Preprocessing** - Normalization, batch correction, dimensionality reduction
- **Visualization** - Comprehensive plotting and analysis tools
- **Quality Control** - Data validation and quality assessment

## 📦 Installation

### System Requirements

Before using OpenCV-related features, install the required system libraries:

#### Using Conda (Recommended)
```bash
# Load Miniforge3/Anaconda (if available)
module load Miniforge3  # or module load Anaconda3

# Create environment
conda create -n py311 python=3.11 -c conda-forge -y
conda activate py311

# Install dependencies (optional, but recommended)
conda install -c conda-forge libjpeg-turbo zlib libpng fftw compilers make cmake imagecodecs -y
```

#### Using System Package Manager (Alternative)
```bash
# Ubuntu/Debian
sudo apt install -y libgl1-mesa-glx libjpeg-dev zlib1g-dev libpng-dev libgl1 libfftw3-dev build-essential python3-dev

# macOS
brew install libjpeg zlib libpng fftw
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

- **[Image Segmentation](api-reference/segmentation.md)** - Detailed segmentation documentation
- **[Clustering and Spatial Transcriptomics](api-reference/clustering.md)** - Clustering and spatial analysis
- **[Practical Examples](examples/complete-pipeline.md)** - Complete workflows and tutorials
- **[API Reference](api-reference/)** - Complete API documentation

## 📂 Examples

Comprehensive examples and tutorials are available in our documentation:

- 🎓 **[Tutorials](tutorials/)** - Step-by-step guides for all major workflows
- 📊 **[Examples](examples/)** - Complete pipelines and practical use cases
- 🔧 **[API Reference](api-reference/)** - Detailed function documentation

### Interactive Examples

- ▶️ **Google Colab**
  [Run on Colab](https://colab.research.google.com/drive/1Qlc3pgN9SlZPUa8kUBu0ePrLG5dj2rd8?usp=sharing)

- 🖥️ **JupyterLab Server**
  [View on Server](http://65.108.226.226:2266/lab/workspaces/auto-j/tree/work/notebook/Segmentation.ipynb)
  password "spexspex"

## ⚙️ System Requirements

- **Python**: 3.11 (recommended), other versions may work
- **Memory**: 8GB+ RAM recommended for large images
- **GPU**: Optional, for faster Cellpose processing
- **Dependencies**: OpenCV, NumPy, SciPy, Scanpy, AnnData

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
2. **Load your data** - Use `load_image()` for microscopy images or `scanpy.read_h5ad()` for AnnData
3. **Segment cells** - Choose from Cellpose, StarDist, or Watershed methods
4. **Extract features** - Get per-cell measurements and characteristics
5. **Analyze spatially** - Perform clustering and spatial analysis
6. **Visualize results** - Create publication-ready plots and figures

## 📖 What's New

- **Complete segmentation pipelines** - From image loading to feature extraction
- **Advanced clustering** - PhenoGraph, Leiden, and Louvain with spatial awareness
- **CLQ analysis** - Co-localization quotient for spatial relationships
- **Niche analysis** - Cell niche identification and interaction analysis
- **Comprehensive documentation** - Step-by-step tutorials and examples
- **Robust error handling** - Fallback methods and quality control
- **Performance optimization** - Efficient algorithms for large datasets