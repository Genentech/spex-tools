"""
Tests for universal tiling functionality.

This module tests the universal tiling system that works with all segmentation methods.
"""

import numpy as np
import pytest
import os
from PIL import Image
import time
from spex.core.segmentation import cellpose_cellseg, stardist_cellseg, watershed_classic


class TestUniversalTiling:
    """Test universal tiling functionality."""

    def test_cellpose_tiling_basic(self):
        """Test basic tiling functionality with cellpose."""
        # Create synthetic image
        img = np.random.rand(1, 200, 200).astype(np.float32)

        # Test without tiling
        labels_no_tiling = cellpose_cellseg(img, [0], 30, 1)
        assert labels_no_tiling.shape == (200, 200)

        # Test with tiling
        labels_tiled = cellpose_cellseg(img, [0], 30, 1, num_tiles=4, overlap=32)
        assert labels_tiled.shape == (200, 200)

        # Both should produce valid segmentation
        assert np.issubdtype(labels_no_tiling.dtype, np.integer)
        assert np.issubdtype(labels_tiled.dtype, np.integer)

    def test_stardist_tiling_basic(self):
        """Test basic tiling functionality with stardist."""
        # Create synthetic image
        img = np.random.rand(1, 200, 200).astype(np.float32)

        # Test without tiling
        labels_no_tiling = stardist_cellseg(img, [0], 1, 0.5, 1, 99)
        assert labels_no_tiling.shape == (200, 200)

        # Test with tiling
        labels_tiled = stardist_cellseg(img, [0], 1, 0.5, 1, 99, num_tiles=4, overlap=32)
        assert labels_tiled.shape == (200, 200)

        # Both should produce valid segmentation
        assert np.issubdtype(labels_no_tiling.dtype, np.integer)
        assert np.issubdtype(labels_tiled.dtype, np.integer)

    def test_watershed_tiling_basic(self):
        """Test basic tiling functionality with watershed."""
        # Create synthetic image
        img = np.random.rand(1, 200, 200).astype(np.float32)

        # Test without tiling
        labels_no_tiling = watershed_classic(img, [0])
        assert labels_no_tiling.shape == (200, 200)

        # Test with tiling
        labels_tiled = watershed_classic(img, [0], num_tiles=4, overlap=32)
        assert labels_tiled.shape == (200, 200)

        # Both should produce valid segmentation
        assert np.issubdtype(labels_no_tiling.dtype, np.integer)
        assert np.issubdtype(labels_tiled.dtype, np.integer)

    def test_real_tiff_image_cellpose(self):
        """Test cellpose segmentation with real TIFF image from project root."""
        # Check if test image exists
        test_image_path = "TA459_multipleCores2_Run-4_Point1.tiff"
        if not os.path.exists(test_image_path):
            pytest.skip(f"Test image {test_image_path} not found")

        # Try different methods to load image
        img_array = None
        try:
            # Method 1: PIL
            with Image.open(test_image_path) as img:
                width, height = img.size
                img_array = np.array(img)
        except Exception:
            try:
                # Method 2: tifffile
                import tifffile
                img_array = tifffile.imread(test_image_path)
                height, width = img_array.shape[-2:]
            except Exception:
                try:
                    # Method 3: skimage
                    from skimage import io
                    img_array = io.imread(test_image_path)
                    height, width = img_array.shape[-2:]
                except Exception as e:
                    pytest.skip(f"Could not load image with any method: {e}")

        # Handle different image shapes
        if len(img_array.shape) == 3:
            if img_array.shape[0] < img_array.shape[1]:  # (C, H, W) format
                if img_array.shape[0] > 1:
                    img_array = np.mean(img_array, axis=0)
                else:
                    img_array = img_array[0]
            else:  # (H, W, C) format
                img_array = np.mean(img_array, axis=2)
        elif len(img_array.shape) == 2:
            pass  # Already 2D
        else:
            pytest.skip(f"Unexpected image shape: {img_array.shape}")

        # Add channel dimension
        img_array = img_array[np.newaxis, :, :].astype(np.float32)

        # Test without tiling
        print(f"Testing cellpose without tiling on image shape: {img_array.shape}")
        start_time = time.time()
        labels_no_tiling = cellpose_cellseg(img_array, [0], 30, 1)
        no_tiling_time = time.time() - start_time
        print(f"Cellpose without tiling: {no_tiling_time:.2f}s")

        # Test with tiling
        print(f"Testing cellpose with tiling on image shape: {img_array.shape}")
        start_time = time.time()
        labels_tiled = cellpose_cellseg(img_array, [0], 30, 1, num_tiles=4, overlap=64)
        tiled_time = time.time() - start_time
        print(f"Cellpose with tiling: {tiled_time:.2f}s")

        # Both should produce valid segmentation
        assert labels_no_tiling.shape == (height, width)
        assert labels_tiled.shape == (height, width)
        assert np.issubdtype(labels_no_tiling.dtype, np.integer)
        assert np.issubdtype(labels_tiled.dtype, np.integer)

        # Count unique labels
        unique_no_tiling = len(np.unique(labels_no_tiling[labels_no_tiling > 0]))
        unique_tiled = len(np.unique(labels_tiled[labels_tiled > 0]))
        print(f"Unique labels without tiling: {unique_no_tiling}")
        print(f"Unique labels with tiling: {unique_tiled}")

        # Both should detect some cells
        assert unique_no_tiling > 0
        assert unique_tiled > 0

    def test_real_tiff_image_watershed(self):
        """Test watershed segmentation with real TIFF image from project root."""
        # Check if test image exists
        test_image_path = "TA459_multipleCores2_Run-4_Point1.tiff"
        if not os.path.exists(test_image_path):
            pytest.skip(f"Test image {test_image_path} not found")

        # Try different methods to load image
        img_array = None
        try:
            # Method 1: PIL
            with Image.open(test_image_path) as img:
                width, height = img.size
                img_array = np.array(img)
        except Exception:
            try:
                # Method 2: tifffile
                import tifffile
                img_array = tifffile.imread(test_image_path)
                height, width = img_array.shape[-2:]
            except Exception:
                try:
                    # Method 3: skimage
                    from skimage import io
                    img_array = io.imread(test_image_path)
                    height, width = img_array.shape[-2:]
                except Exception as e:
                    pytest.skip(f"Could not load image with any method: {e}")

        # Handle different image shapes
        if len(img_array.shape) == 3:
            if img_array.shape[0] < img_array.shape[1]:  # (C, H, W) format
                if img_array.shape[0] > 1:
                    img_array = np.mean(img_array, axis=0)
                else:
                    img_array = img_array[0]
            else:  # (H, W, C) format
                img_array = np.mean(img_array, axis=2)
        elif len(img_array.shape) == 2:
            pass  # Already 2D
        else:
            pytest.skip(f"Unexpected image shape: {img_array.shape}")

        # Add channel dimension
        img_array = img_array[np.newaxis, :, :].astype(np.float32)

        # Test without tiling
        print(f"Testing watershed without tiling on image shape: {img_array.shape}")
        start_time = time.time()
        labels_no_tiling = watershed_classic(img_array, [0])
        no_tiling_time = time.time() - start_time
        print(f"Watershed without tiling: {no_tiling_time:.2f}s")

        # Test with tiling
        print(f"Testing watershed with tiling on image shape: {img_array.shape}")
        start_time = time.time()
        labels_tiled = watershed_classic(img_array, [0], num_tiles=4, overlap=64)
        tiled_time = time.time() - start_time
        print(f"Watershed with tiling: {tiled_time:.2f}s")

        # Both should produce valid segmentation
        assert labels_no_tiling.shape == (height, width)
        assert labels_tiled.shape == (height, width)
        assert np.issubdtype(labels_no_tiling.dtype, np.integer)
        assert np.issubdtype(labels_tiled.dtype, np.integer)

        # Count unique labels
        unique_no_tiling = len(np.unique(labels_no_tiling[labels_no_tiling > 0]))
        unique_tiled = len(np.unique(labels_tiled[labels_tiled > 0]))
        print(f"Unique labels without tiling: {unique_no_tiling}")
        print(f"Unique labels with tiling: {unique_tiled}")

        # Both should detect some cells
        assert unique_no_tiling > 0
        assert unique_tiled > 0

    def test_tiling_parameters_validation(self):
        """Test parameter validation for tiling."""
        img = np.random.rand(1, 100, 100).astype(np.float32)

        # Test invalid num_tiles
        with pytest.raises(ValueError, match="Number of tiles must be positive"):
            cellpose_cellseg(img, [0], 30, 1, num_tiles=0)

        # Test invalid overlap
        with pytest.raises(ValueError, match="Overlap must be non-negative"):
            cellpose_cellseg(img, [0], 30, 1, num_tiles=4, overlap=-10)

    def test_performance_comparison(self):
        """Test performance comparison between tiled and non-tiled segmentation."""
        # Create larger synthetic image
        img = np.random.rand(1, 400, 400).astype(np.float32)

        # Test cellpose performance
        print("\\n=== Cellpose Performance Comparison ===")

        # Without tiling
        start_time = time.time()
        labels_no_tiling = cellpose_cellseg(img, [0], 30, 1)
        no_tiling_time = time.time() - start_time

        # With tiling
        start_time = time.time()
        labels_tiled = cellpose_cellseg(img, [0], 30, 1, num_tiles=4, overlap=32)
        tiled_time = time.time() - start_time

        print(f"Without tiling: {no_tiling_time:.2f}s")
        print(f"With tiling: {tiled_time:.2f}s")
        print(f"Speedup: {no_tiling_time/tiled_time:.2f}x")

        # Both should produce valid results
        assert labels_no_tiling.shape == (400, 400)
        assert labels_tiled.shape == (400, 400)
        assert np.issubdtype(labels_no_tiling.dtype, np.integer)
        assert np.issubdtype(labels_tiled.dtype, np.integer)
