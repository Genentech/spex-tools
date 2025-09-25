"""Utilities for benchmarking sequential vs parallel tiling modes."""

from __future__ import annotations

import time
import os
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import psutil

from .dask_segmentation import dask_apply_tiling_to_segmentation

MB = 1024 * 1024


def benchmark_parallel_modes(
    seg_func,
    img: np.ndarray,
    *args: Iterable[Any],
    tile_size: Tuple[int, int],
    overlap: int,
    return_labels: bool = True,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Compare sequential and parallel tiling execution for a segmentation function.

    Args:
        seg_func: Segmentation entrypoint accepting ``parallel`` keyword.
        img: Input image array with shape ``(C, H, W)``.
        *args: Positional arguments forwarded to ``seg_func``.
        tile_size: Tile size used for both runs.
        overlap: Overlap in pixels.
        **kwargs: Additional keyword arguments forwarded to ``seg_func``.

    Returns:
        Dictionary with keys ``"sequential"`` and ``"parallel"``.
        Each entry contains ``labels`` (``np.ndarray`` or ``None`` when ``return_labels``
        is False) and ``metrics`` dict with
        ``time_s``, ``memory_mb``, ``cpu_time_s``.
    """

    result = {}
    for parallel in (False, True):
        mode = "parallel" if parallel else "sequential"
        labels, metrics = _run_benchmark_mode(
            seg_func,
            img,
            args,
            tile_size,
            overlap,
            parallel,
            kwargs,
        )
        result[mode] = {
            "labels": labels if return_labels else None,
            "metrics": metrics,
        }

    return result


def _run_benchmark_mode(
    seg_func,
    img: np.ndarray,
    args: Iterable[Any],
    tile_size: Tuple[int, int],
    overlap: int,
    parallel: bool,
    kwargs: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Execute ``seg_func`` under a specific ``parallel`` flag collecting metrics."""

    process = psutil.Process(os.getpid())

    cpu_before = process.cpu_times()
    mem_before = process.memory_info().rss
    start = time.perf_counter()

    labels = seg_func(
        img,
        *args,
        tile_size=tile_size,
        overlap=overlap,
        parallel=parallel,
        **kwargs,
    )

    elapsed = time.perf_counter() - start
    cpu_after = process.cpu_times()
    mem_after = process.memory_info().rss

    metrics = {
        "time_s": float(elapsed),
        "cpu_time_s": float((cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)),
        "memory_mb": float(max(mem_after - mem_before, 0) / MB),
    }

    return labels, metrics


__all__ = ["benchmark_parallel_modes"]
