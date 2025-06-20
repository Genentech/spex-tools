import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spex import CLQ_vec_numba
from scipy.sparse import csr_matrix
from spex import niche


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


@pytest.mark.parametrize("method", ["leiden", "louvain"])
def test_cluster_creates_expected_labels(method):
    X = np.random.rand(5, 3)
    adata = AnnData(X)

    conn = np.ones((5, 5)) - np.eye(5)
    adata.obsp["connectivities"] = csr_matrix(conn)

    clustered = niche(adata.copy(), resolution=0.5, method=method)

    assert "leiden" in clustered.obs.columns or "louvain" in clustered.obs.columns
    labels = clustered.obs[method]
    assert labels.nunique() > 0
    assert len(labels) == adata.n_obs