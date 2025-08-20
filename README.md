# SPEX - Spatial Omics Analysis Library

This library implements methods developed for the [SPEX](https://www.biorxiv.org/content/10.1101/2022.08.22.504841v2) software platform, enabling users to apply state-of-the-art tissue segmentation techniques on their own image data.

## 📚 Documentation

- **📖 API Reference**: [https://genentech.github.io/spex-tools/api/](https://genentech.github.io/spex-tools/api/)
- **🔧 Installation Guide**: See installation section below
- **📋 Examples**: See examples section below

## 📦 Installation

### From PyPI (recommended)

```bash
pip install spex-tools
```

### From source

Upgrade pip and install dependencies:

```bash
pip install --upgrade pip setuptools wheel packaging
pip install pytest
```

### 📚 Install with documentation

To install with documentation dependencies:

```bash
pip install spex-tools[docs]
```

### 🛠️ System Requirements

Before using OpenCV-related features, install the required system libraries:

```bash
sudo apt install -y libgl1-mesa-glx libjpeg-dev zlib1g-dev libpng-dev libgl1 libfftw3-dev build-essential python3-dev

```

Install the package locally:

```bash
pip install .
```

## 🚀 Quick Start

```python
import spex as sp

# Load image
Image, channel = sp.load_image('your_image.tiff')

# Perform watershed segmentation
labels = sp.watershed_classic(Image, [0])

# Extract features
features = sp.feature_extraction_adata(Image, labels, channel)
```

## 🔧 Key Features

- **🖼️ Image Processing**: Load and process multi-channel images
- **🔍 Segmentation**: Watershed, Cellpose, and StarDist segmentation
- **📊 Feature Extraction**: Extract per-cell expression data
- **🎯 Clustering**: PhenoGraph clustering for cell type identification
- **🧬 Spatial Analysis**: CLQ analysis and niche detection
- **📈 Differential Expression**: Multiple statistical methods for DE analysis
- **🔄 Preprocessing**: Comprehensive data preprocessing pipeline

## 📂 Examples

Use the methods directly in your own analysis pipelines. Example notebooks are available:

- ▶️ **Google Colab**
  [Run on Colab](https://colab.research.google.com/drive/1Qlc3pgN9SlZPUa8kUBu0ePrLG5dj2rd8?usp=sharing)

- 🖥️ **JupyterLab Server**
  [View on Server](http://65.108.226.226:2266/lab/workspaces/auto-j/tree/work/notebook/Segmentation.ipynb)
  password "spexspex"

### Notebooks include:

- Model downloading (in case Cellpose server access fails)
- Visualization examples
- End-to-end segmentation pipelines
- Clustering and spatial transcriptomics analysis

## ⚙️ Compatibility

- ✅ Tested with **Python 3.11**
- ⚠️ Compatibility with other Python versions is not guaranteed
- ⚙️ Includes integrated **Cellpose** support, with fallback model handling (in notebooks)

## 🤝 Support

- **📖 Documentation**: [https://genentech.github.io/spex-tools/api/](https://genentech.github.io/spex-tools/api/)
- **🐛 Issues**: Report bugs and feature requests on GitHub
- **💬 Questions**: Use GitHub Discussions for questions and help
