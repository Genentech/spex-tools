import os
import importlib.util
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API",
    category=UserWarning,
)
import pkg_resources  # needed for Pegasus, deprecated but still in use

# vendored chex
vendored_chex_path = os.path.join(os.path.dirname(__file__), "chex")

spec = importlib.util.spec_from_file_location(
    "chex", os.path.join(vendored_chex_path, "__init__.py")
)
chex_module = importlib.util.module_from_spec(spec)
sys.modules["chex"] = chex_module
spec.loader.exec_module(chex_module)
# vendored chex

# fix np.bool8
import numpy as np

np.bool8 = np.bool_
# fix np.bool8

# === FIX: JAX CPU ONLY ===
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import jax.numpy as jnp
import jax

if not hasattr(jax, "interpreters"):
    jax.interpreters = type("fake", (), {})()
if not hasattr(jax.interpreters, "xla"):
    jax.interpreters.xla = type("fake", (), {})()
jax.interpreters.xla.DeviceArray = type(jnp.array(0))
# === FIX: JAX CPU ONLY ===

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
    feature_extraction,
    feature_extraction_adata,
)
from .core.spatial_transcriptomics.clq import CLQ_vec_numba
from .core.spatial_transcriptomics.niche import niche
from .core.clustering.phenograph import phenograph_cluster
from .core.spatial_transcriptomics.preprocessing import (
    preprocess,
    MAD_threshold,
    should_batch_correct,
)
from .core.spatial_transcriptomics.reduce_dimensionality import reduce_dimensionality
from .core.spatial_transcriptomics.clustering import cluster
from .core.spatial_transcriptomics.differential_expression import (
    differential_expression,
)
from .core.spatial_transcriptomics.analyze_pathways import (
    annotate_clusters,
    analyze_pathways,
)
from .core.spatial_transcriptomics.load_anndata import load_anndata
from .core.utils import download_cellpose_models, delete_cellpose_models, to_uint8
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
    "feature_extraction",
    "feature_extraction_adata",
    "phenograph_cluster",
    "niche",
    "preprocess",
    "MAD_threshold",
    "should_batch_correct",
    "reduce_dimensionality",
    "cluster",
    "differential_expression",
    "annotate_clusters",
    "analyze_pathways",
    "CLQ_vec_numba",
    "load_anndata",
    "download_cellpose_models",
    "delete_cellpose_models",
    "to_uint8",
]

try:
    from spex.core.utils import download_cellpose_models

    download_cellpose_models()
except Exception as e:
    print(f"[spex] ⚠️ Model download skipped: {e}")
