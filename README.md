# SPEX - Spatial Omics Analysis Library

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://genentech.github.io/spex-tools/)

SPEX is a comprehensive spatial omics analysis library supporting both spatial proteomics and spatial transcriptomics. It implements state-of-the-art methods for tissue segmentation, clustering, and spatial analysis. The library enables researchers to apply advanced image processing and spatial analysis techniques to their own microscopy data.

## 🚀 Key Features

### 🖼️ Image Segmentation
- **Cellpose** - Deep learning-based cell segmentation
- **StarDist** - Star-convex object-based segmentation  
- **Watershed** - Classical watershed segmentation
- **Preprocessing** - Background subtraction, noise removal, filtering
- **Post-processing** - Cell rescue, object filtering, feature extraction

### 🔬 Spatial Proteomics Analysis
- **Image Preprocessing** - Background subtraction and noise removal methods
- **Segmentation** - Cellpose, StarDist, and Watershed for protein marker images
- **Post-processing** - Cell rescue, filtering, and feature extraction
- **Clustering** - PhenoGraph and other clustering algorithms for protein profiles
- **CLQ Analysis** - Co-localization quotient for spatial protein relationships
- **Niche Analysis** - Protein-based cell niche identification and interactions

### 🧬 Spatial Transcriptomics Analysis
- **Clustering** - PhenoGraph, Leiden, Louvain algorithms
- **Niche Analysis** - Cell niche identification and interactions
- **Differential Expression** - Spatial-aware differential analysis
- **Pathway Analysis** - Cluster annotation and signaling pathways
- **CLQ Analysis** - Co-localization quotient for spatial relationships

### 🔧 Utilities
- **Data Loading** - OME-TIFF, OME-ZARR, AnnData formats
- **Preprocessing** - Normalization, batch correction, dimensionality reduction
- **Visualization** - Comprehensive plotting and analysis tools

## 📦 Installation

### Quick Start

```bash
# Install from PyPI
pip install spex-tools
```

### Advanced Installation

For optimal performance, we recommend installing system dependencies:

#### Option 1: Using Conda (Recommended)

```bash
# Load Miniforge3/Anaconda (if available on your system)
module load Miniforge3  # or module load Anaconda3

# Create and activate conda environment
conda create -n py311 python=3.11 -c conda-forge -y
conda activate py311

# Install system dependencies (optional, but recommended)
conda install -c conda-forge libjpeg-turbo zlib libpng fftw compilers make cmake imagecodecs -y

# Install SPEX
pip install spex-tools
```

#### Option 2: Using System Package Manager

```bash
# Ubuntu/Debian
sudo apt install -y libgl1-mesa-glx libjpeg-dev zlib1g-dev libpng-dev libgl1 libfftw3-dev build-essential python3-dev

# Install SPEX
pip install spex-tools
```

### From Source

```bash
# Clone repository
git clone https://github.com/genentech/spex-tools.git
cd spex-tools

# Set up environment
conda create -n py311 python=3.11 -c conda-forge -y
conda activate py311

# Install
pip install .
```

## 🔧 Quick Start

```python
import spex as sp

# Load an image
array, channels = sp.load_image("path/to/image.ome.tiff")

# Perform cell segmentation
labels = sp.cellpose_cellseg(array, seg_channels=[0], diameter=30, scaling=1)

# Extract features
features = sp.feature_extraction(array, labels)

print(f"Detected {labels.max()} cells")
```

## 📚 Documentation

- **[Installation Guide](https://genentech.github.io/spex-tools/getting-started/installation/)** - Detailed setup instructions
- **[Tutorials](https://genentech.github.io/spex-tools/tutorials/segmentation-basics/)** - Step-by-step guides
- **[API Reference](https://genentech.github.io/spex-tools/api-reference/)** - Complete function documentation
- **[Examples](https://genentech.github.io/spex-tools/examples/)** - Practical workflows and use cases

## 📂 Example Workflows

### Complete Segmentation Pipeline

```python
import spex as sp

# 1. Load and preprocess image
array, channels = sp.load_image("image.ome.tiff")
array = sp.background_subtract(array)
array = sp.median_denoise(array)

# 2. Segment cells
labels = sp.cellpose_cellseg(array, seg_channels=[0], diameter=30)

# 3. Post-process
labels = sp.remove_small_objects(labels, min_size=50)
labels = sp.remove_large_objects(labels, max_size=1000)

# 4. Extract features
features = sp.feature_extraction(array, labels)
```

### Spatial Analysis Workflow

```python
import spex as sp
import scanpy as sc

# 1. Load AnnData
adata = sc.read_h5ad("spatial_data.h5ad")

# 2. Preprocess
adata = sp.preprocess(adata)

# 3. Cluster
adata = sp.cluster(adata, method='leiden', resolution=0.5)

# 4. Spatial analysis
clq_results = sp.CLQ_vec_numba(adata, cluster_key='leiden')
niche_results = sp.niche(adata, cluster_key='leiden')

# 5. Differential expression
de_results = sp.differential_expression(adata, groupby='leiden')
```

### Interactive Examples

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

## 🎯 Getting Started

1. **Install SPEX** - Follow the installation instructions above
2. **Load your data** - Use `load_image()` for microscopy images or `scanpy.read_h5ad()` for AnnData
3. **Segment cells** - Choose from Cellpose, StarDist, or Watershed methods
4. **Extract features** - Get per-cell measurements and characteristics
5. **Analyze spatially** - Perform clustering and spatial analysis

## ⚙️ System Requirements

- **Python**: 3.11 (recommended), other versions may work
- **Memory**: 8GB+ RAM recommended for large images
- **GPU**: Optional, for faster Cellpose processing
- **Dependencies**: OpenCV, NumPy, SciPy, Scanpy, AnnData
- **Platform Compatibility**: 
  - ✅ **Linux (x86_64)**: Fully supported
  - ✅ **Windows (x86_64)**: Fully supported
  - ❌ **macOS**: Not supported due to compatibility issues with core dependencies
  - ❌ **ARM64/Apple Silicon**: Not supported
  - ❌ **Other architectures**: Not tested or supported

## 🔍 Troubleshooting

### Common Issues

1. **OpenCV Import Error**
   - Ensure system dependencies are installed
   - Try: `conda install -c conda-forge opencv`

2. **Cellpose Model Download Issues**
   - Models download automatically on first use
   - Check internet connection
   - See documentation for manual download

3. **Memory Issues**
   - Large images may require significant RAM
   - Consider using smaller image tiles

For more help, see our [Troubleshooting Guide](https://genentech.github.io/spex-tools/reference/troubleshooting/).



## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE.md) file for details.

## 🔗 Links

- 📚 [Full Documentation](https://genentech.github.io/spex-tools/)
- 🎓 [Tutorials](https://genentech.github.io/spex-tools/tutorials/)

---

**SPEX** - Empowering spatial omics research with advanced analysis tools for proteomics and transcriptomics.
