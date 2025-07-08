from .core.segmentation.io import load_image
from .core.segmentation.filters import median_denoise, nlm_denoise
from .core.segmentation.stardist import stardist_cellseg
from .core.segmentation.background_subtract import background_subtract
from .core.segmentation.watershed import watershed_classic
from .core.segmentation.postprocessing import (
    rescue_cells,
    simulate_cell,
    remove_large_objects,
    remove_small_objects,
    feature_extraction_adata
)
from .core.spatial_transcriptomics.clq import CLQ_vec_numba
from .core.spatial_transcriptomics.niche import niche
from .core.clustering.phenograph import phenograph_cluster
from .core.spatial_transcriptomics.preprocessing import (
    preprocess,
    MAD_threshold,
    should_batch_correct
)
from .core.spatial_transcriptomics.reduce_dimensionality import (
    reduce_dimensionality
)
from .core.spatial_transcriptomics.clustering import cluster
from importlib.metadata import version

__version__ = version("spex")

from .core.segmentation.cellpose_cellseg import cellpose_cellseg

__all__ = [
    "load_image",
    "median_denoise",
    "nlm_denoise",
    "stardist_cellseg",
    "watershed_classic",
    "background_subtract",
    "cellpose_cellseg",
    "rescue_cells",
    "simulate_cell",
    "remove_large_objects",
    "remove_small_objects",
    "feature_extraction_adata",
    "phenograph_cluster",
    "niche",
    "preprocess",
    "MAD_threshold",
    "should_batch_correct",
    "reduce_dimensionality",
    "cluster"
]

try:
    from spex.core.utils import download_cellpose_models
    download_cellpose_models()
except Exception as e:
    print(f"[spex] ⚠️ Model download skipped: {e}")