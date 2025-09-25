import os

os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
import pytest

pytest.importorskip("dask")

from spex.core.segmentation import dask_watershed
from spex.core.tiling import dask_segmentation


def test_watershed_dask_parallel_matches_sequential(monkeypatch):
    """Ensure parallel tiling produces the same result as sequential processing."""

    def fake_watershed_core(tile_img, seg_channels):
        """Return a simple, deterministic label mask for testing."""
        height, width = tile_img.shape[-2:]
        labels = np.zeros((height, width), dtype=np.uint32)
        labels[height // 4:height - height // 4, width // 4:width - width // 4] = 1
        return labels

    monkeypatch.setattr(
        "spex.core.segmentation.watershed._watershed_core",
        fake_watershed_core,
    )

    parallel_calls = {"count": 0}
    original_parallel = dask_segmentation._process_tiles_parallel

    def tracking_parallel(*args, **kwargs):
        parallel_calls["count"] += 1
        return original_parallel(*args, **kwargs)

    monkeypatch.setattr(
        dask_segmentation,
        "_process_tiles_parallel",
        tracking_parallel,
    )

    img = np.zeros((1, 128, 128), dtype=np.float32)

    labels_sequential = dask_watershed.watershed_classic_dask(
        img,
        [0],
        tile_size=(64, 64),
        overlap=16,
        parallel=False,
    )

    labels_parallel = dask_watershed.watershed_classic_dask(
        img,
        [0],
        tile_size=(64, 64),
        overlap=16,
        parallel=True,
    )

    np.testing.assert_array_equal(labels_sequential, labels_parallel)
    assert parallel_calls["count"] > 0, "Parallel path was not exercised"


def test_watershed_dask_auto_parallel(monkeypatch):
    calls = {"parallel": 0}

    def fake_watershed_core(tile_img, seg_channels):
        return np.zeros(tile_img.shape[-2:], dtype=np.uint32)

    monkeypatch.setattr(
        "spex.core.segmentation.watershed._watershed_core",
        fake_watershed_core,
    )

    original_parallel = dask_segmentation._process_tiles_parallel

    def tracking_parallel(*args, **kwargs):
        calls["parallel"] += 1
        return original_parallel(*args, **kwargs)

    monkeypatch.setattr(
        dask_segmentation,
        "_process_tiles_parallel",
        tracking_parallel,
    )

    img = np.zeros((1, 512, 512), dtype=np.float32)
    # With 128x128 tiles we expect 16 tiles -> auto parallel kicks in
    dask_watershed.watershed_classic_dask(
        img,
        [0],
        tile_size=(128, 128),
        overlap=32,
    )

    assert calls["parallel"] > 0


def test_watershed_dask_auto_parallel_disabled_for_single_tile(monkeypatch):
    def fake_watershed_core(tile_img, seg_channels):
        return np.zeros(tile_img.shape[-2:], dtype=np.uint32)

    monkeypatch.setattr(
        "spex.core.segmentation.watershed._watershed_core",
        fake_watershed_core,
    )

    calls = {"parallel": 0}

    def tracking_parallel(*args, **kwargs):
        calls["parallel"] += 1
        raise AssertionError("Parallel path should not be used")

    monkeypatch.setattr(
        dask_segmentation,
        "_process_tiles_parallel",
        tracking_parallel,
    )

    img = np.zeros((1, 128, 128), dtype=np.float32)
    # Tile large enough to cover entire image -> only one tile, no parallel
    dask_watershed.watershed_classic_dask(
        img,
        [0],
        tile_size=(256, 256),
        overlap=32,
    )

    assert calls["parallel"] == 0
