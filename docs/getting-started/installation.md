# 📦 Installation Guide

## System Requirements

### Python Version
- **Recommended**: Python 3.11
- **Compatibility**: Other Python versions may work but are not guaranteed

### System Dependencies

Before using OpenCV-related features, install the required system libraries:

#### Option 1: Using Conda (Recommended)

```bash
# Load Miniforge3/Anaconda (if available on your system)
# For HPC systems with module system:
module load Miniforge3  # or module load Anaconda3

# Create and activate conda environment
conda create -n py311 python=3.11 -c conda-forge -y
conda activate py311

# Install system dependencies via conda-forge (optional, but recommended)
conda install -c conda-forge libjpeg-turbo zlib libpng fftw compilers make cmake imagecodecs -y
```

#### Option 2: Using System Package Manager (Alternative)

```bash
# Ubuntu/Debian
sudo apt install -y libgl1-mesa-glx libjpeg-dev zlib1g-dev libpng-dev libgl1 libfftw3-dev build-essential python3-dev

# CentOS/RHEL
sudo yum install -y mesa-libGL libjpeg-devel zlib-devel libpng-devel mesa-libGL-devel fftw3-devel gcc gcc-c++ python3-devel

# macOS (using Homebrew)
brew install libjpeg zlib libpng fftw
```

## Installation Methods

### From PyPI (Recommended)

```bash
pip install spex-tools
```

**Note**: This minimal installation includes core functionality. For optimal performance with large images, consider installing system dependencies as shown in the advanced installation options below.

### Install with Documentation Dependencies

To install with all documentation dependencies:

```bash
pip install spex-tools[docs]
```

### Install with Development Dependencies

For contributors and developers:

```bash
pip install spex-tools[dev]
```

### From Source

1. **Set up environment** (choose one):
```bash
# Option A: Using conda (recommended)
conda create -n py311 python=3.11 -c conda-forge -y
conda activate py311

# Option B: Using virtual environment
python -m venv spex-env
source spex-env/bin/activate  # On Windows: spex-env\Scripts\activate
```

2. **Upgrade pip and install dependencies**:
```bash
pip install --upgrade pip setuptools wheel packaging
pip install pytest
```

3. **Clone the repository**:
```bash
git clone https://github.com/genentech/spex-tools.git
cd spex-tools
```

4. **Install locally**:
```bash
pip install .
```

## Verification

Test the installation:

```python
import spex as sp
print("SPEX library imported successfully!")
```

## Troubleshooting

### Common Issues

1. **OpenCV Import Error**
   - Ensure system dependencies are installed
   - If using conda: `conda install -c conda-forge opencv`
   - If using pip: `pip uninstall opencv-python && pip install opencv-python`

2. **Conda Environment Issues**
   - Ensure conda environment is activated: `conda activate py311`
   - Verify conda-forge dependencies: `conda list | grep -E "(libjpeg|zlib|libpng|fftw)"`
   - Reinstall if needed: `conda install -c conda-forge libjpeg-turbo zlib libpng fftw compilers make cmake imagecodecs -y`

3. **Cellpose Model Download Issues**
   - Check internet connection
   - Models will be downloaded automatically on first use
   - See notebooks for manual model download instructions

4. **Memory Issues**
   - Large images may require significant RAM
   - Consider using smaller image tiles for processing

### Getting Help

- 📚 [Full Documentation](https://genentech.github.io/spex-tools/api/)
- 🐛 [GitHub Issues](https://github.com/genentech/spex-tools/issues)
- 📖 [Examples and Tutorials](../tutorials/)
