# SPEX - Spatial Omics Analysis Library

This library implements methods developed for the [SPEX](https://www.biorxiv.org/content/10.1101/2022.08.22.504841v2) software platform, enabling users to apply state-of-the-art tissue segmentation techniques on their own image data.

## 📦 Installation

Upgrade pip and install dependencies:

```bash
pip install --upgrade pip setuptools wheel packaging
pip install pytest
```

### 🛠️ System Requirements

Before using OpenCV-related features, install the required system libraries:

```bash
sudo apt install -y libgl1-mesa-glx libjpeg-dev zlib1g-dev libpng-dev libgl1
```

Install the package locally:

```bash
pip install .
```

## 📂 Examples

Use the methods directly in your own analysis pipelines. Example notebooks are available:

- ▶️ **Google Colab**
  [Run on Colab](https://colab.research.google.com/drive/1Qlc3pgN9SlZPUa8kUBu0ePrLG5dj2rd8?usp=sharing)

- 🖥️ **JupyterLab Server**
  [View on Server](http://65.108.226.226:2266/lab/workspaces/auto-j/tree/notebook/Segmentation.ipynb)
  password "spexspex"

Notebooks include:
- Model downloading (in case Cellpose server access fails)
- Visualization examples
- End-to-end segmentation pipelines

## ⚙️ Compatibility

- ✅ Tested with **Python 3.10** and **3.11**
- ⚠️ Compatibility with other Python versions is not guaranteed
- ⚙️ Includes integrated **Cellpose** support, with fallback model handling (in notebooks)
