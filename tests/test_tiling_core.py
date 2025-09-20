"""
Tests for tiling core utilities.

This module tests the basic tiling functions: compute_tiles, crop_core, place_core.
"""

import numpy as np
import pytest
import os
from PIL import Image
from spex.core.tiling.core import (
    compute_tiles, crop_core, place_core,
    crop_core_safe, get_core_coordinates_safe
)


class TestComputeTiles:
    """Test compute_tiles function."""

    def test_normal_case(self):
        """Test normal case with overlap."""
        tiles = compute_tiles((100, 100), (50, 50), 10)
        # With step = 50 - 10 = 40, we get:
        # Height: 0, 40, 80 -> 3 tiles
        # Width: 0, 40, 80 -> 3 tiles
        # Total: 3x3 = 9 tiles
        assert len(tiles) == 9
        # Check first tile
        assert tiles[0] == (slice(0, 50), slice(0, 50))
        # Check last tile
        assert tiles[-1] == (slice(80, 100), slice(80, 100))

    def test_no_overlap(self):
        """Test case with no overlap."""
        tiles = compute_tiles((100, 100), (50, 50), 0)
        # With step = 50 - 0 = 50, we get:
        # Height: 0, 50 -> 2 tiles
        # Width: 0, 50 -> 2 tiles
        # Total: 2x2 = 4 tiles
        assert len(tiles) == 4
        # Check first tile
        assert tiles[0] == (slice(0, 50), slice(0, 50))
        # Check last tile
        assert tiles[-1] == (slice(50, 100), slice(50, 100))

    def test_image_smaller_than_tile(self):
        """Test when image is smaller than tile."""
        tiles = compute_tiles((30, 30), (50, 50), 10)
        expected = [(slice(0, 30), slice(0, 30))]
        assert len(tiles) == 1
        assert tiles == expected

    def test_non_multiple_dimensions(self):
        """Test when dimensions are not multiples of tile size."""
        tiles = compute_tiles((120, 120), (50, 50), 10)
        # With step = 50 - 10 = 40, we get:
        # Height: 0, 40, 80 -> 3 tiles
        # Width: 0, 40, 80 -> 3 tiles
        # Total: 3x3 = 9 tiles
        assert len(tiles) == 9
        # Check first tile
        assert tiles[0] == (slice(0, 50), slice(0, 50))
        # Check last tile (should be cropped)
        assert tiles[-1] == (slice(80, 120), slice(80, 120))

    def test_single_tile(self):
        """Test when only one tile fits."""
        tiles = compute_tiles((40, 40), (50, 50), 0)
        expected = [(slice(0, 40), slice(0, 40))]
        assert len(tiles) == 1
        assert tiles == expected

    def test_negative_overlap_error(self):
        """Test error for negative overlap."""
        with pytest.raises(ValueError, match="Overlap must be non-negative"):
            compute_tiles((100, 100), (50, 50), -5)

    def test_overlap_too_large_error(self):
        """Test error when overlap >= min(tile)."""
        with pytest.raises(ValueError, match="Overlap .* must be < min\\(tile\\)"):
            compute_tiles((100, 100), (50, 50), 50)

    def test_zero_tile_error(self):
        """Test error for zero tile size."""
        with pytest.raises(ValueError, match="Tile dimensions must be positive"):
            compute_tiles((100, 100), (0, 50), 10)

    def test_real_image_dimensions(self):
        """Test with real image dimensions (2048x2048)."""
        tiles = compute_tiles((2048, 2048), (512, 512), 64)
        # With step = 512 - 64 = 448, we get:
        # Height: 0, 448, 896, 1344, 1792 -> 5 tiles
        # Width: 0, 448, 896, 1344, 1792 -> 5 tiles
        # Total: 5x5 = 25 tiles
        assert len(tiles) == 25
        # Check first tile
        assert tiles[0] == (slice(0, 512), slice(0, 512))
        # Check last tile
        assert tiles[-1] == (slice(1792, 2048), slice(1792, 2048))

    def test_real_tiff_image(self):
        """Test with actual TIFF image from project root."""
        # Check if test image exists
        test_image_path = "TA459_multipleCores2_Run-4_Point1.tiff"
        if not os.path.exists(test_image_path):
            pytest.skip(f"Test image {test_image_path} not found")

        try:
            # Try to load with PIL first
            with Image.open(test_image_path) as img:
                # Get dimensions without loading full image
                width, height = img.size
                image_array = None  # We don't need the full array for this test
        except Exception as e:
            # If PIL fails, try alternative approach
            try:
                import tifffile
                with tifffile.TiffFile(test_image_path) as tif:
                    shape = tif.series[0].shape
                    if len(shape) == 3:
                        height, width = shape[1], shape[2]  # Skip channel dimension
                    else:
                        height, width = shape
            except ImportError:
                pytest.skip("Neither PIL nor tifffile can load the image")

        # Test tiling with actual dimensions
        tiles = compute_tiles((height, width), (512, 512), 64)

        # Verify tiles are valid
        assert len(tiles) > 0
        for tile in tiles:
            assert tile[0].start >= 0
            assert tile[1].start >= 0
            assert tile[0].stop <= height
            assert tile[1].stop <= width


class TestCropCore:
    """Test crop_core function."""

    def test_normal_crop(self):
        """Test normal core cropping."""
        # Create a tile with overlap
        tile = np.random.rand(60, 60)  # 50 + 10 overlap
        core = crop_core(tile, 10)
        expected_shape = (40, 40)  # 50 - 10 overlap
        assert core.shape == expected_shape
        # Core should be from the center
        assert np.array_equal(core, tile[10:50, 10:50])

    def test_no_overlap(self):
        """Test cropping with no overlap."""
        tile = np.random.rand(50, 50)
        core = crop_core(tile, 0)
        assert core.shape == (50, 50)
        assert np.array_equal(core, tile)

    def test_negative_overlap_error(self):
        """Test error for negative overlap."""
        tile = np.random.rand(50, 50)
        with pytest.raises(ValueError, match="Overlap must be non-negative"):
            crop_core(tile, -5)

    def test_wrong_dimensions_error(self):
        """Test error for wrong array dimensions."""
        tile = np.random.rand(50, 50, 3)  # 3D array
        with pytest.raises(ValueError, match="Expected 2D array"):
            crop_core(tile, 10)


class TestPlaceCore:
    """Test place_core function."""

    def test_normal_placement(self):
        """Test normal core placement."""
        dst = np.zeros((100, 100))
        core = np.ones((30, 30))
        place_core(dst, core, (10, 20))

        # Check that core was placed correctly
        assert np.array_equal(dst[10:40, 20:50], core)
        # Check that other areas are still zero
        assert np.all(dst[:10, :] == 0)
        assert np.all(dst[40:, :] == 0)
        assert np.all(dst[:, :20] == 0)
        assert np.all(dst[:, 50:] == 0)

    def test_edge_placement(self):
        """Test placement at edges."""
        dst = np.zeros((50, 50))
        core = np.ones((20, 20))
        place_core(dst, core, (0, 0))

        assert np.array_equal(dst[:20, :20], core)
        assert np.all(dst[20:, :] == 0)
        assert np.all(dst[:, 20:] == 0)

    def test_negative_coordinates_error(self):
        """Test error for negative coordinates."""
        dst = np.zeros((50, 50))
        core = np.ones((20, 20))
        with pytest.raises(ValueError, match="Coordinates must be non-negative"):
            place_core(dst, core, (-5, 10))

    def test_out_of_bounds_error(self):
        """Test error for out of bounds placement."""
        dst = np.zeros((50, 50))
        core = np.ones((30, 30))
        with pytest.raises(ValueError, match="Core would extend beyond destination"):
            place_core(dst, core, (30, 30))


class TestPropertyReassembly:
    """Test property: reassemble tiles equals source."""

    def test_reassembly_property(self):
        """Test that reassembling tiles gives original image."""
        # Create synthetic image
        original = np.random.rand(100, 100)

        # Compute tiles
        tiles = compute_tiles((100, 100), (50, 50), 10)

        # Extract cores using safe cropping
        cores = []
        for tile in tiles:
            core = crop_core_safe(original[tile], 10, tile, (100, 100), (50, 50))
            cores.append(core)

        # Reassemble using safe placement
        reconstructed = np.zeros_like(original)
        for tile, core in zip(tiles, cores):
            y, x = get_core_coordinates_safe(tile, 10, (100, 100), (50, 50))
            place_core(reconstructed, core, (y, x))

        # Should be identical
        assert np.allclose(original, reconstructed)

    def test_reassembly_real_image_dimensions(self):
        """Test reassembly with real image dimensions."""
        # Create synthetic image with real dimensions
        original = np.random.rand(2048, 2048)

        # Compute tiles
        tiles = compute_tiles((2048, 2048), (512, 512), 64)

        # Extract cores using safe cropping
        cores = []
        for tile in tiles:
            core = crop_core_safe(original[tile], 64, tile, (2048, 2048), (512, 512))
            cores.append(core)

        # Reassemble using safe placement
        reconstructed = np.zeros_like(original)
        for tile, core in zip(tiles, cores):
            y, x = get_core_coordinates_safe(tile, 64, (2048, 2048), (512, 512))
            place_core(reconstructed, core, (y, x))

        # Should be identical
        assert np.allclose(original, reconstructed)

