"""
Segmentation module for SPEX.

This module provides various segmentation methods including Cellpose, StarDist,
Watershed, and their tiled versions for memory-efficient processing.
"""

from .cellpose_cellseg import cellpose_cellseg
from .stardist import stardist_cellseg
from .watershed import watershed_classic
from .background_subtract import background_subtract
from .filters import median_denoise, nlm_denoise
from .postprocessing import (
    rescue_cells,
    simulate_cell,
    remove_large_objects,
    remove_small_objects
)
from .io import load_image

__all__ = [
    "cellpose_cellseg",
    "stardist_cellseg",
    "watershed_classic",
    "background_subtract",
    "median_denoise",
    "nlm_denoise",
    "rescue_cells",
    "simulate_cell",
    "remove_large_objects",
    "remove_small_objects",
    "load_image",
]
