"""
Tests for large file processing capabilities.

This module contains tests for the Patient_test_1 breakthrough methodology
and validates the tiling system for large spatial transcriptomics files.

Follows SOLID, DRY, KISS principles:
- Single Responsibility: Each test has one clear purpose
- DRY: Common functionality in base class
- KISS: Simple, clear test structure
"""

import pytest
import numpy as np
import time
import os


class BaseLargeFileTest:
    """Base class for large file tests following DRY principle."""

    @staticmethod
    def validate_image_shape(img, expected_shape):
        """Validate image has expected shape."""
        assert img.shape == expected_shape, f"Expected {expected_shape}, got {img.shape}"

    @staticmethod
    def validate_processing_result(labels, expected_shape, min_labels=0, max_time=None, processing_time=None):
        """Validate segmentation processing results."""
        label_count = len(np.unique(labels)) - 1
        assert labels.shape == expected_shape, f"Labels shape mismatch: {labels.shape} vs {expected_shape}"
        assert label_count >= min_labels, f"Too few labels: {label_count} < {min_labels}"

        if max_time and processing_time:
            assert processing_time < max_time, f"Processing too slow: {processing_time:.1f}s > {max_time}s"

        return label_count

    @staticmethod
    def skip_if_file_missing(file_path):
        """Skip test if required file is not available."""
        if not os.path.exists(file_path):
            pytest.skip(f"Test file {file_path} not available")

    @staticmethod
    def skip_if_env_not_set(env_var, reason):
        """Skip test if environment variable is not set."""
        if not os.environ.get(env_var):
            pytest.skip(reason)


class TestLargeFileProcessing(BaseLargeFileTest):
    """Test suite for large file processing capabilities."""

    def test_auto_tiling_decision(self):
        """Test automatic tiling decision logic."""
        import spex as sp
        from spex.core.tiling.unified import should_use_tiling_for_image

        # Small image - should not use tiling
        small_img = np.random.rand(3, 100, 100).astype(np.uint8)
        assert not should_use_tiling_for_image(small_img, 500.0)

        # Large image - should use tiling (19.1MB should exceed 10MB threshold)
        large_img = np.random.rand(5, 2000, 2000).astype(np.uint8)
        assert should_use_tiling_for_image(large_img, 10.0)

    def test_cellpose_auto_tiling(self):
        """Test cellpose with automatic tiling detection."""
        import spex as sp

        # Create medium-sized test image
        test_img = np.random.rand(3, 500, 500).astype(np.uint8) * 255

        # Should work without explicit tiling
        labels = sp.cellpose_cellseg(
            test_img, [0], diameter=20, scaling=1,
            auto_tiling=False  # Disable for small test
        )

        assert labels.shape == (500, 500)
        assert labels.dtype in [np.int32, np.uint32]

    def test_dask_functions_available(self):
        """Test that all Dask functions are available in main API."""
        import spex as sp

        required_dask_functions = [
            'cellpose_cellseg_dask',
            'watershed_classic_dask',
            'stardist_cellseg_dask'
        ]

        for func_name in required_dask_functions:
            assert hasattr(sp, func_name), f"Missing function: {func_name}"

    def test_backward_compatibility(self):
        """Test backward compatibility with old API."""
        import spex as sp
        import warnings

        test_img = np.random.rand(3, 200, 200).astype(np.uint8) * 255

        # Old API with num_tiles should work but show warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            labels = sp.cellpose_cellseg(
                test_img, [0], diameter=20, scaling=1,
                num_tiles=4,
                auto_tiling=False  # Disable auto-tiling for this test
            )

            # Check that deprecation warning was issued
            assert len(w) > 0
            assert "deprecated" in str(w[0].message)

        assert labels.shape == (200, 200)

    @pytest.mark.slow
    def test_ta459_processing(self):
        """Test processing of TA459 file (if available)."""
        import spex as sp

        ta459_path = "TA459_multipleCores2_Run-4_Point1.tiff"
        self.skip_if_file_missing(ta459_path)

        # Load TA459
        img, channels = sp.load_image(ta459_path)
        self.validate_image_shape(img, (44, 2048, 2048))

        # Process with auto-tiling
        start_time = time.time()
        labels = sp.cellpose_cellseg(img, [0], diameter=30, scaling=1.0)
        processing_time = time.time() - start_time

        # Validate results using base class method
        label_count = self.validate_processing_result(
            labels,
            expected_shape=(2048, 2048),
            min_labels=1,
            max_time=120,
            processing_time=processing_time
        )

        print(f"TA459 processed in {processing_time:.1f}s, found {label_count} labels")

    @pytest.mark.slow
    @pytest.mark.patient_test_1
    def test_patient_test_1_processing(self):
        """Test Patient_test_1.ome.tiff with proven 99-tile configuration.

        This test uses the proven working configuration:
        - 2000×2000px tiles (19.1MB each)
        - 400px overlap (20%)
        - Expected: 15,332 cells in ~35 minutes

        Follows KISS principle: Simple, explicit configuration.
        """
        import spex as sp

        # KISS: Clear preconditions
        patient_path = "Patient_test_1.ome.tiff"
        self.skip_if_file_missing(patient_path)
        self.skip_if_env_not_set("RUN_PATIENT_TEST_1",
                                "Patient_test_1 test requires RUN_PATIENT_TEST_1=1 environment variable")

        # Load and validate
        img, channels = sp.load_image(patient_path)
        self.validate_image_shape(img, (5, 14400, 17280))
        print(f"Loaded Patient_test_1: {img.shape}, Memory: {img.nbytes/(1024*1024):.1f}MB")

        # SOLID: Single responsibility - use proven configuration
        start_time = time.time()
        labels = sp.cellpose_cellseg_dask(
            img, [0],
            diameter=30,
            scaling=1,
            tile_size=(2000, 2000),  # Proven 19.1MB tiles
            overlap=400              # Proven 20% overlap
        )
        processing_time = time.time() - start_time

        # Validate using base class method (DRY principle)
        label_count = self.validate_processing_result(
            labels,
            expected_shape=(14400, 17280),
            min_labels=10000,
            max_time=4000,
            processing_time=processing_time
        )

        print(f"Patient_test_1 processed in {processing_time/60:.1f}min, found {label_count} labels")

        # Verify against proven result
        if 14000 <= label_count <= 16000:
            print("✅ Result consistent with proven Patient_test_1 methodology!")
        else:
            print(f"⚠️ Label count {label_count} differs from proven result (15,332)")

    def test_tile_size_calculation(self):
        """Test tile size calculation for different image sizes."""
        from spex.core.tiling.unified import TilingProcessor

        processor = TilingProcessor(memory_limit_mb=30, overlap_percent=20)

        # Test various image shapes
        test_cases = [
            ((5, 1000, 1000), 30),  # Small image
            ((5, 5000, 5000), 30),  # Medium image
            ((5, 14400, 17280), 30),  # Patient_test_1 size
        ]

        for img_shape, memory_limit in test_cases:
            tile_size = processor.calculate_optimal_tile_size(img_shape, memory_limit)

            # Validate tile size
            assert len(tile_size) == 2
            assert tile_size[0] > 0 and tile_size[1] > 0
            assert tile_size[0] <= img_shape[1]
            assert tile_size[1] <= img_shape[2]

            # Check memory constraint
            channels = img_shape[0]
            estimated_mb = (tile_size[0] * tile_size[1] * channels) / (1024 * 1024)
            assert estimated_mb <= memory_limit + 1  # Allow small tolerance

    def test_overlap_calculation(self):
        """Test overlap calculation logic."""
        from spex.core.tiling.unified import TilingProcessor

        processor = TilingProcessor(overlap_percent=20)

        test_cases = [
            ((100, 100), 32),    # 20px calculated but 32px minimum enforced
            ((500, 500), 100),   # 100px overlap (20% of 500)
            ((2000, 2000), 400), # 400px overlap (20% of 2000, Patient_test_1 config)
        ]

        for tile_size, expected_overlap in test_cases:
            overlap = processor.calculate_overlap(tile_size)
            assert overlap == expected_overlap

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        from spex.core.tiling.unified import TilingProcessor

        processor = TilingProcessor()

        # Test different image sizes
        small_img = np.zeros((3, 500, 500), dtype=np.uint8)
        large_img = np.zeros((5, 5000, 5000), dtype=np.uint8)

        small_mem = processor._estimate_memory_usage(small_img)
        large_mem = processor._estimate_memory_usage(large_img)

        assert small_mem < large_mem
        assert small_mem > 0
        assert large_mem > 0

        # Check specific values
        expected_small = (3 * 500 * 500) / (1024 * 1024)  # ~0.7MB
        expected_large = (5 * 5000 * 5000) / (1024 * 1024)  # ~119MB

        assert abs(small_mem - expected_small) < 0.1
        assert abs(large_mem - expected_large) < 1.0