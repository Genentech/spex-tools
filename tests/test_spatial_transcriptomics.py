import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spex import CLQ_vec_numba


def test_clq_vec_numba_basic():
    n_cells = 100
    n_clusters = 3
    coords = np.random.rand(n_cells, 2) * 100  # coord
    clusters = np.random.choice(['A', 'B', 'C'], size=n_cells)  # clasters

    obs = pd.DataFrame({
        'x_coordinate': coords[:, 0],
        'y_coordinate': coords[:, 1],
        'leiden': clusters
    })

    adata = AnnData(obs=obs)

    bdata, adata_out = CLQ_vec_numba(adata, clust_col='leiden', radius=20, n_perms=10)

    # out
    assert 'NCV' in adata_out.obsm
    assert 'local_clq' in adata_out.obsm
    assert 'CLQ' in adata_out.uns
    assert 'global_clq' in adata_out.uns['CLQ']
    assert 'permute_test' in adata_out.uns['CLQ']

    # dims
    k = len(np.unique(clusters))
    assert adata_out.uns['CLQ']['global_clq'].shape == (k, k)
    assert adata_out.uns['CLQ']['permute_test'].shape == (k, k)
    assert adata_out.obsm['NCV'].shape == (n_cells, k)
    assert adata_out.obsm['local_clq'].shape == (n_cells, k)