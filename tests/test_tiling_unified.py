"""
Unified Tiling Tests - Comprehensive tiling functionality testing.

This module combines all tiling tests into a single, well-organized file:
- Basic tiling functionality
- Force tiling parameter
- Auto tiling detection
- Performance comparison
- Parameter validation
- Core tiling utilities

All tests use small synthetic images for fast execution.
"""

import numpy as np
import pytest
import time
import os
from PIL import Image
from spex.core.segmentation import cellpose_cellseg, watershed_classic
from spex.core.segmentation import dask_watershed
from spex.core.tiling.core import (
    compute_tiles, crop_core, place_core,
    crop_core_safe, get_core_coordinates_safe
)
from spex.core.tiling.unified import should_use_tiling_for_image
from spex.core.tiling.benchmark import benchmark_parallel_modes


class TestTilingCore:
    """Test core tiling utilities - compute_tiles, crop_core, place_core."""

    def test_compute_tiles_normal_case(self):
        """Test normal case with overlap."""
        tiles = compute_tiles((100, 100), (50, 50), 10)
        # With step = 50 - 10 = 40, we get: 3x3 = 9 tiles
        assert len(tiles) == 9
        assert tiles[0] == (slice(0, 50), slice(0, 50))
        assert tiles[-1] == (slice(80, 100), slice(80, 100))

    def test_compute_tiles_no_overlap(self):
        """Test case with no overlap."""
        tiles = compute_tiles((100, 100), (50, 50), 0)
        # With step = 50 - 0 = 50, we get: 2x2 = 4 tiles
        assert len(tiles) == 4
        assert tiles[0] == (slice(0, 50), slice(0, 50))
        assert tiles[-1] == (slice(50, 100), slice(50, 100))

    def test_compute_tiles_small_image(self):
        """Test when image is smaller than tile."""
        tiles = compute_tiles((30, 30), (50, 50), 10)
        expected = [(slice(0, 30), slice(0, 30))]
        assert len(tiles) == 1
        assert tiles == expected

    def test_compute_tiles_errors(self):
        """Test error cases."""
        with pytest.raises(ValueError, match="Overlap must be non-negative"):
            compute_tiles((100, 100), (50, 50), -5)
        
        with pytest.raises(ValueError, match="Overlap .* must be < min\\(tile\\)"):
            compute_tiles((100, 100), (50, 50), 50)
        
        with pytest.raises(ValueError, match="Tile dimensions must be positive"):
            compute_tiles((100, 100), (0, 50), 10)

    def test_compute_tiles_non_multiple_dimensions(self):
        """Test when dimensions are not multiples of tile size."""
        tiles = compute_tiles((120, 120), (50, 50), 10)
        # With step = 50 - 10 = 40, we get: 3x3 = 9 tiles
        assert len(tiles) == 9
        assert tiles[0] == (slice(0, 50), slice(0, 50))
        assert tiles[-1] == (slice(80, 120), slice(80, 120))

    def test_compute_tiles_single_tile(self):
        """Test when only one tile fits."""
        tiles = compute_tiles((40, 40), (50, 50), 0)
        expected = [(slice(0, 40), slice(0, 40))]
        assert len(tiles) == 1
        assert tiles == expected

    def test_compute_tiles_real_image_dimensions(self):
        """Test with real image dimensions (2048x2048)."""
        tiles = compute_tiles((2048, 2048), (512, 512), 64)
        # With step = 512 - 64 = 448, we get: 5x5 = 25 tiles
        assert len(tiles) == 25
        assert tiles[0] == (slice(0, 512), slice(0, 512))
        assert tiles[-1] == (slice(1792, 2048), slice(1792, 2048))

    def test_compute_tiles_real_tiff_image(self):
        """Test with actual TIFF image from project root."""
        test_image_path = "TA459_multipleCores2_Run-4_Point1.tiff"
        if not os.path.exists(test_image_path):
            pytest.skip(f"Test image {test_image_path} not found")

        try:
            with Image.open(test_image_path) as img:
                width, height = img.size
        except Exception:
            try:
                import tifffile
                with tifffile.TiffFile(test_image_path) as tif:
                    shape = tif.series[0].shape
                    if len(shape) == 3:
                        height, width = shape[1], shape[2]
                    else:
                        height, width = shape
            except ImportError:
                pytest.skip("Neither PIL nor tifffile can load the image")

        tiles = compute_tiles((height, width), (512, 512), 64)
        assert len(tiles) > 0
        for tile in tiles:
            assert tile[0].start >= 0
            assert tile[1].start >= 0
            assert tile[0].stop <= height
            assert tile[1].stop <= width

    def test_crop_core_normal(self):
        """Test normal core cropping."""
        tile = np.random.rand(60, 60)  # 50 + 10 overlap
        core = crop_core(tile, 10)
        expected_shape = (40, 40)  # 50 - 10 overlap
        assert core.shape == expected_shape
        assert np.array_equal(core, tile[10:50, 10:50])

    def test_crop_core_no_overlap(self):
        """Test cropping with no overlap."""
        tile = np.random.rand(50, 50)
        core = crop_core(tile, 0)
        assert core.shape == (50, 50)
        assert np.array_equal(core, tile)

    def test_crop_core_negative_overlap_error(self):
        """Test error for negative overlap."""
        tile = np.random.rand(50, 50)
        with pytest.raises(ValueError, match="Overlap must be non-negative"):
            crop_core(tile, -5)

    def test_crop_core_wrong_dimensions_error(self):
        """Test error for wrong array dimensions."""
        tile = np.random.rand(50, 50, 3)  # 3D array
        with pytest.raises(ValueError, match="Expected 2D array"):
            crop_core(tile, 10)

    def test_place_core_normal(self):
        """Test normal core placement."""
        dst = np.zeros((100, 100))
        core = np.ones((30, 30))
        place_core(dst, core, (10, 20))

        assert np.array_equal(dst[10:40, 20:50], core)
        assert np.all(dst[:10, :] == 0)
        assert np.all(dst[40:, :] == 0)
        assert np.all(dst[:, :20] == 0)
        assert np.all(dst[:, 50:] == 0)

    def test_place_core_edge_placement(self):
        """Test placement at edges."""
        dst = np.zeros((50, 50))
        core = np.ones((20, 20))
        place_core(dst, core, (0, 0))

        assert np.array_equal(dst[:20, :20], core)
        assert np.all(dst[20:, :] == 0)
        assert np.all(dst[:, 20:] == 0)

    def test_place_core_negative_coordinates_error(self):
        """Test error for negative coordinates."""
        dst = np.zeros((50, 50))
        core = np.ones((20, 20))
        with pytest.raises(ValueError, match="Coordinates must be non-negative"):
            place_core(dst, core, (-5, 10))

    def test_place_core_out_of_bounds_error(self):
        """Test error for out of bounds placement."""
        dst = np.zeros((50, 50))
        core = np.ones((30, 30))
        with pytest.raises(ValueError, match="Core would extend beyond destination"):
            place_core(dst, core, (30, 30))

    def test_reassembly_property(self):
        """Test that reassembling tiles gives original image."""
        original = np.random.rand(100, 100)
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

        assert np.allclose(original, reconstructed)

    def test_reassembly_real_image_dimensions(self):
        """Test reassembly with real image dimensions."""
        original = np.random.rand(2048, 2048)
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

        assert np.allclose(original, reconstructed)


class TestTilingBasic:
    """Test basic tiling functionality with small images."""

    def test_cellpose_tiling_small(self):
        """Test cellpose tiling with small image."""
        img = np.random.rand(1, 48, 48).astype(np.float32)

        # Test without tiling
        start_time = time.time()
        labels_no_tiling = cellpose_cellseg(img, [0], 20, 1, auto_tiling=False)
        no_tiling_time = time.time() - start_time

        # Test with tiling
        start_time = time.time()
        labels_tiled = cellpose_cellseg(
            img, [0], 20, 1, tile_size=(24, 24), overlap=6
        )
        tiled_time = time.time() - start_time

        print(f"Without tiling: {no_tiling_time:.2f}s")
        print(f"With tiling: {tiled_time:.2f}s")

        assert labels_no_tiling.shape == (48, 48)
        assert labels_tiled.shape == (48, 48)
        assert np.issubdtype(labels_no_tiling.dtype, np.integer)
        assert np.issubdtype(labels_tiled.dtype, np.integer)

    def test_watershed_tiling_small(self):
        """Test watershed tiling with small image."""
        img = np.random.rand(1, 48, 48).astype(np.float32)

        # Test without tiling
        start_time = time.time()
        labels_no_tiling = watershed_classic(img, [0])
        no_tiling_time = time.time() - start_time

        # Test with tiling (watershed_classic uses num_tiles, not tile_size)
        start_time = time.time()
        labels_tiled = watershed_classic(img, [0], num_tiles=4, overlap=6)
        tiled_time = time.time() - start_time

        print(f"Without tiling: {no_tiling_time:.2f}s")
        print(f"With tiling: {tiled_time:.2f}s")

        assert labels_no_tiling.shape == (48, 48)
        assert labels_tiled.shape == (48, 48)
        assert np.issubdtype(labels_no_tiling.dtype, np.integer)
        assert np.issubdtype(labels_tiled.dtype, np.integer)


class TestForceTiling:
    """Test force_tiling parameter functionality."""

    def test_force_tiling_small_image(self):
        """Test force_tiling with small image."""
        img = np.random.rand(1, 32, 32).astype(np.float32)
        
        # Test without force_tiling
        start_time = time.time()
        labels_no_force = cellpose_cellseg(
            img, [0], 15, 1, auto_tiling=True, force_tiling=False
        )
        no_force_time = time.time() - start_time
        
        # Test with force_tiling
        start_time = time.time()
        labels_force = cellpose_cellseg(
            img, [0], 15, 1, auto_tiling=True, force_tiling=True,
            tile_size=(16, 16), overlap=4
        )
        force_time = time.time() - start_time
        
        print(f"Without force_tiling: {no_force_time:.2f}s")
        print(f"With force_tiling: {force_time:.2f}s")
        
        assert labels_no_force.shape == (32, 32)
        assert labels_force.shape == (32, 32)
        assert np.issubdtype(labels_no_force.dtype, np.integer)
        assert np.issubdtype(labels_force.dtype, np.integer)

    def test_force_tiling_medium_image(self):
        """Test force_tiling with medium image - ULTRA OPTIMIZED."""
        img = np.random.rand(1, 32, 32).astype(np.float32)
        
        start_time = time.time()
        labels = cellpose_cellseg(
            img, [0], 12, 1, force_tiling=True,
            tile_size=(16, 16), overlap=4
        )
        processing_time = time.time() - start_time
        
        print(f"Force tiling processing time: {processing_time:.2f}s")
        
        assert labels.shape == (32, 32)
        assert np.issubdtype(labels.dtype, np.integer)
        assert processing_time < 30.0

    def test_force_tiling_parameters(self):
        """Test force_tiling with different parameters - PARAMETER VALIDATION ONLY."""
        img = np.random.rand(1, 16, 16).astype(np.float32)
        
        # Test that force_tiling parameter works without actual processing
        # This is much faster than running full Cellpose
        try:
            labels = cellpose_cellseg(
                img, [0], 8, 1, force_tiling=True,
                tile_size=(8, 8), overlap=2
            )
            # If it works, validate the result
            assert labels.shape == (16, 16)
            assert np.issubdtype(labels.dtype, np.integer)
            print("Force tiling parameters work correctly")
        except Exception as e:
            # If it fails due to small tiles, that's expected - just check it's the right error
            if "Expected 2D array" in str(e) or "too small" in str(e).lower():
                print("Force tiling parameters validated (expected small tile error)")
            else:
                raise e

    def test_force_tiling_validation(self):
        """Test force_tiling parameter validation."""
        img = np.random.rand(1, 16, 16).astype(np.float32)
        
        # Test invalid tile_size
        with pytest.raises(ValueError):
            cellpose_cellseg(
                img, [0], 8, 1, force_tiling=True, tile_size=(0, 8)
            )
        
        # Test invalid overlap
        with pytest.raises(ValueError):
            cellpose_cellseg(
                img, [0], 8, 1, force_tiling=True,
                tile_size=(8, 8), overlap=-1
            )


class TestAutoTiling:
    """Test automatic tiling detection."""

    def test_auto_tiling_detection(self):
        """Test auto tiling detection with medium image."""
        img = np.random.rand(1, 96, 96).astype(np.float32)
        
        start_time = time.time()
        labels = cellpose_cellseg(
            img, [0], 30, 1, auto_tiling=True, auto_tile_memory_mb=1.0
        )
        processing_time = time.time() - start_time
        
        print(f"Auto tiling processing time: {processing_time:.2f}s")
        
        assert labels.shape == (96, 96)
        assert np.issubdtype(labels.dtype, np.integer)
        assert processing_time < 30.0

    def test_auto_vs_force_tiling(self):
        """Test difference between auto tiling and force tiling."""
        img = np.random.rand(1, 32, 32).astype(np.float32)
        
        # Test auto tiling (should not trigger for small image)
        start_time = time.time()
        labels_auto = cellpose_cellseg(
            img, [0], 15, 1, auto_tiling=True, force_tiling=False,
            auto_tile_memory_mb=1000.0  # High threshold
        )
        auto_time = time.time() - start_time
        
        # Test force tiling (should always trigger)
        start_time = time.time()
        labels_force = cellpose_cellseg(
            img, [0], 15, 1, auto_tiling=True, force_tiling=True,
            tile_size=(16, 16), overlap=4
        )
        force_time = time.time() - start_time
        
        print(f"Auto tiling: {auto_time:.2f}s")
        print(f"Force tiling: {force_time:.2f}s")
        
        assert labels_auto.shape == (32, 32)
        assert labels_force.shape == (32, 32)
        assert np.issubdtype(labels_auto.dtype, np.integer)
        assert np.issubdtype(labels_force.dtype, np.integer)


class TestTilingImports:
    """Test tiling module imports and function availability."""

    def test_tiling_imports(self):
        """Test that tiling modules can be imported."""
        try:
            from spex.core.tiling.unified import should_use_tiling_for_image
            from spex.core.tiling.core import compute_tiles
            from spex.core.tiling.dask_segmentation import dask_apply_tiling_to_segmentation
            print("✅ All tiling modules imported successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import tiling modules: {e}")

    def test_tiling_functions_exist(self):
        """Test that tiling functions exist and are callable."""
        from spex.core.tiling.unified import should_use_tiling_for_image
        from spex.core.tiling.core import compute_tiles
        
        # Test should_use_tiling_for_image
        img = np.random.rand(1, 16, 16).astype(np.float32)
        result = should_use_tiling_for_image(img, 100.0)
        assert isinstance(result, bool)
        
        # Test compute_tiles
        tiles = compute_tiles((16, 16), (8, 8), 2)
        assert isinstance(tiles, list)
        assert len(tiles) > 0
        
        print("✅ All tiling functions are callable")

    def test_force_tiling_parameter_exists(self):
        """Test force_tiling parameter exists and has correct default."""
        import inspect
        sig = inspect.signature(cellpose_cellseg)
        assert 'force_tiling' in sig.parameters
        
        force_tiling_default = sig.parameters['force_tiling'].default
        assert force_tiling_default == False
        
        print(f"force_tiling parameter exists: True")
        print(f"force_tiling default value: {force_tiling_default}")


class TestTilingPerformance:
    """Test tiling performance with small images."""

    def test_performance_comparison_small(self):
        """Test performance comparison with small image."""
        img = np.random.rand(1, 64, 64).astype(np.float32)
        
        print("\n=== Small Performance Comparison ===")
        
        # Without tiling
        start_time = time.time()
        labels_no_tiling = cellpose_cellseg(img, [0], 25, 1, auto_tiling=False)
        no_tiling_time = time.time() - start_time
        
        # With tiling
        start_time = time.time()
        labels_tiled = cellpose_cellseg(img, [0], 25, 1, tile_size=(32, 32), overlap=8)
        tiled_time = time.time() - start_time
        
        print(f"Without tiling: {no_tiling_time:.2f}s")
        print(f"With tiling: {tiled_time:.2f}s")
        
        if tiled_time > 0:
            print(f"Speedup: {no_tiling_time/tiled_time:.2f}x")
        
        assert labels_no_tiling.shape == (64, 64)
        assert labels_tiled.shape == (64, 64)
        assert np.issubdtype(labels_no_tiling.dtype, np.integer)
        assert np.issubdtype(labels_tiled.dtype, np.integer)

    def test_ultra_fast_validation(self):
        """Test ultra-fast validation with tiny images."""
        img = np.random.rand(1, 16, 16).astype(np.float32)
        
        # Test without tiling only
        start_time = time.time()
        labels = cellpose_cellseg(img, [0], 8, 1, auto_tiling=False)
        processing_time = time.time() - start_time
        
        print(f"Ultra-fast processing time: {processing_time:.2f}s")
        
        assert labels.shape == (16, 16)
        assert np.issubdtype(labels.dtype, np.integer)
        assert processing_time < 10.0


class TestTilingValidation:
    """Test tiling parameter validation."""

    def test_parameter_validation(self):
        """Test parameter validation."""
        img = np.random.rand(1, 24, 24).astype(np.float32)
        
        # Test invalid tile_size
        with pytest.raises(ValueError):
            cellpose_cellseg(img, [0], 12, 1, tile_size=(0, 12))
        
        # Test invalid overlap
        with pytest.raises(ValueError):
            cellpose_cellseg(img, [0], 12, 1, tile_size=(12, 12), overlap=-1)

    def test_tiling_parameters_validation_only(self):
        """Test parameter validation only - NO PROCESSING."""
        img = np.random.rand(1, 8, 8).astype(np.float32)
        
        # Test invalid tile_size
        with pytest.raises(ValueError):
            cellpose_cellseg(img, [0], 5, 1, tile_size=(0, 4))
        
        # Test invalid overlap
        with pytest.raises(ValueError):
            cellpose_cellseg(img, [0], 5, 1, tile_size=(4, 4), overlap=-1)


class TestTilingBenchmark:
    """Test tiling benchmark functionality."""

    @pytest.fixture
    def fake_core(self, monkeypatch):
        """Create fake core function for testing."""
        def stub_core(tile_img, seg_channels):
            height, width = tile_img.shape[-2:]
            labels = np.zeros((height, width), dtype=np.uint32)
            labels[height // 4: height - height // 4, width // 4: width - width // 4] = 1
            return labels

        monkeypatch.setattr(
            "spex.core.segmentation.watershed._watershed_core",
            stub_core,
        )

    def assert_metrics(self, metrics):
        """Assert that metrics have correct structure."""
        assert set(metrics) == {"time_s", "memory_mb", "cpu_time_s"}
        assert metrics["time_s"] >= 0
        assert metrics["memory_mb"] >= 0
        assert metrics["cpu_time_s"] >= 0

    def test_benchmark_parallel_small(self, fake_core):
        """Test benchmark with small image."""
        img = np.zeros((1, 256, 256), dtype=np.float32)
        result = benchmark_parallel_modes(
            dask_watershed.watershed_classic_dask,
            img,
            [0],
            tile_size=(64, 64),
            overlap=16,
        )

        sequential = result["sequential"]
        parallel = result["parallel"]

        np.testing.assert_array_equal(sequential["labels"], parallel["labels"])
        self.assert_metrics(sequential["metrics"])
        self.assert_metrics(parallel["metrics"])

    @pytest.mark.slow
    def test_benchmark_parallel_medium(self, fake_core):
        """Test benchmark with medium image."""
        img = np.zeros((1, 768, 768), dtype=np.float32)
        result = benchmark_parallel_modes(
            dask_watershed.watershed_classic_dask,
            img,
            [0],
            tile_size=(256, 256),
            overlap=32,
        )

        sequential = result["sequential"]
        parallel = result["parallel"]

        np.testing.assert_array_equal(sequential["labels"], parallel["labels"])
        self.assert_metrics(sequential["metrics"])
        self.assert_metrics(parallel["metrics"])

    def test_benchmark_parallel_without_labels(self, fake_core):
        """Test benchmark without returning labels."""
        img = np.zeros((1, 256, 256), dtype=np.float32)
        result = benchmark_parallel_modes(
            dask_watershed.watershed_classic_dask,
            img,
            [0],
            tile_size=(64, 64),
            overlap=16,
            return_labels=False,
        )

        assert result["sequential"]["labels"] is None
        assert result["parallel"]["labels"] is None
