import phenograph
import anndata as ad
import re
from typing import List
import scanpy as sc
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
import numpy as np
import pandas as pd

def phenograph_cluster(
    adata: ad.AnnData,
    channel_names: List[str],
    knn: int = 30,
    transformation: str = 'arcsin',
    scaling: str = 'z-score',
    cofactor: float = 5.0,
    umap_min_dist: float = 0.5
) -> ad.AnnData:
    """
    Performs PhenoGraph clustering and adds results to .obs and .obsm of the AnnData object.
    Marker channels are specified by name; their indices are resolved via adata.uns['channel_index_map'].

    Parameters
    ----------
    adata : AnnData
        AnnData object containing single-cell features.
    channel_names : List[str]
        List of channel names (e.g., ['CD3', 'CD8']) to be used for clustering.
    knn : int
        Number of neighbors for graph construction (used in PhenoGraph and UMAP).
    transformation : str
        Data transformation to apply before clustering. Options: 'arcsin', 'log', 'none'.
    scaling : str
        Feature scaling strategy. Options: 'z-score', 'winsorize', 'none'.
    cofactor : float
        Cofactor for arcsinh transformation (if used).
    umap_min_dist : float
        Minimum distance parameter for UMAP projection.

    Returns
    -------
    AnnData
        Updated AnnData object with:
        - adata.obs['cluster_phenograph']: cluster labels (as strings)
        - adata.obsm['X_umap']: 2D UMAP coordinates
    """

    def normalize(name: str) -> str:
        return re.sub(r"[^0-9a-zA-Z]", "", name).lower().replace("target", "")

    if 'channel_index_map' not in adata.uns:
        adata.uns['channel_index_map'] = {
            normalize(ch): i for i, ch in enumerate(adata.var_names)
        }

    channel_map = adata.uns['channel_index_map']

    marker_indices = []
    for target in channel_names:
        target_norm = normalize(target)
        found = [idx for name, idx in channel_map.items() if target_norm in name]
        if found:
            marker_indices.extend(found)

    if not marker_indices:
        raise ValueError(f"No valid channel names found for {channel_names}")

    df = adata.to_df()
    data_for_calc = df.iloc[:, marker_indices]

    if transformation == 'arcsin':
        data_for_calc = np.arcsinh(data_for_calc / cofactor)
    elif transformation == 'log':
        data_for_calc = data_for_calc.apply(
            lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x
        )

    if scaling == 'z-score':
        data_for_calc = data_for_calc.apply(zscore)
    elif scaling == 'winsorize':
        arr_data_frame_winsorized = winsorize(
            data_for_calc.to_numpy(), limits=(0, 0.01)
        ).data
        data_for_calc.iloc[:, :] = arr_data_frame_winsorized

    communities, graph, Q = phenograph.cluster(
        data_for_calc.values.tolist(),
        k=knn,
        clustering_algo='leiden'
    )

    # UMAP
    bdata = ad.AnnData(data_for_calc)
    sc.pp.neighbors(bdata, n_neighbors=knn)
    sc.tl.umap(bdata, min_dist=umap_min_dist)

    adata.obsm['X_umap'] = bdata.obsm['X_umap']
    adata.obs['cluster_phenograph'] = [str(i) for i in communities]

    return adata