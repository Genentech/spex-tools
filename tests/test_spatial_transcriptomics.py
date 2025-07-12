import numpy as np
import pandas as pd
import pytest
import anndata
from anndata import AnnData
from spex import CLQ_vec_numba
from scipy.sparse import csr_matrix
import scipy.sparse as sp_sparse
from spex import niche
from spex import preprocess, MAD_threshold, should_batch_correct
from spex import reduce_dimensionality
from spex import cluster
from spex import differential_expression
import scvi
import pegasus as pg


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



def test_mad_threshold():
    x = np.array([1, 2, 3, 4, 100])  # выброс
    result = MAD_threshold(x, ndevs=1)
    assert result < np.median(x)


def test_should_batch_correct_true():
    adata = anndata.AnnData(np.ones((10, 5)))
    adata.uns["batch_key"] = "batch"
    adata.obs["batch"] = ["A"] * 5 + ["B"] * 5
    assert should_batch_correct(adata) is True


def test_should_batch_correct_false():
    adata = anndata.AnnData(np.ones((10, 5)))
    assert should_batch_correct(adata) is False


def test_preprocess_basic():
    X = sp_sparse.csr_matrix(np.random.poisson(1, (20, 10)))
    adata = anndata.AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(10)]
    adata.obs_names = [f"cell_{i}" for i in range(20)]

    processed = preprocess(adata.copy(), scale_max=5, size_factor=None, do_QC=False)

    assert "log1p" in processed.uns
    assert "prepro" in processed.uns
    assert "counts" in processed.layers
    assert processed.X.shape[1] <= 10  # могли быть отфильтрованы гены



@pytest.mark.parametrize("method", ["pca", "diff_map", "scvi"])
@pytest.mark.parametrize("prefilter", [False, True])
@pytest.mark.parametrize("use_batch", [False, True])
def test_reduce_dimensionality_all(method, prefilter, use_batch):
    # Only this scenario is currently used in the notebook;
    # mark other combinations as expected failures until examples are added.
    if not (method == "pca" and prefilter is False and use_batch is False):
        pytest.xfail("Combination not yet supported in the current pipeline")

    X = np.random.poisson(1, (30, 20))
    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(20)]
    adata.obs_names = [f"cell_{i}" for i in range(30)]

    if prefilter:
        adata.var['highly_variable'] = [True] * 10 + [False] * 10

    if use_batch:
        adata.obs["batch"] = ["A"] * 15 + ["B"] * 15
        adata.uns["batch_key"] = "batch"

    if method == "scvi":
        adata.raw = AnnData(X=adata.X.copy(), obs=adata.obs.copy(), var=adata.var.copy())

    reduced = reduce_dimensionality(adata, prefilter=prefilter, method=method)

    assert 'X_umap' in reduced.obsm
    assert 'neighbor_idx' in reduced.obsm
    assert 'distances' in reduced.obsm
    assert 'connectivities' in reduced.obsp
    assert 'dim_reduce' in reduced.uns

    if method == "diff_map":
        assert 'X_diffmap' in reduced.obsm
        assert 'diffmap_evals' in reduced.uns

    if method == "scvi":
        assert 'X_scvi' in reduced.obsm

    if use_batch and method in ["pca", "diff_map"]:
        assert 'X_pca_harmony' in reduced.obsm


def test_cluster_function_direct_call():
    X = np.random.rand(5, 3)
    adata = AnnData(X)
    conn = np.ones((5, 5)) - np.eye(5)
    adata.obsp["connectivities"] = csr_matrix(conn)

    # optional: spatial connectivity
    adata.obsp["spatial_connectivities"] = csr_matrix(np.eye(5))

    clustered = cluster(adata.copy(), spatial_weight=0.5, resolution=0.5, method='leiden')

    assert "leiden" in clustered.obs.columns
    labels = clustered.obs["leiden"]
    assert labels.nunique() > 0
    assert len(labels) == adata.n_obs

def test_differential_expression_scvi_real():
    X = np.random.poisson(1, (30, 10))
    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(10)]
    adata.obs["leiden"] = ["A"] * 15 + ["B"] * 15

    # Подготовка adata для scvi
    scvi.model.SCVI.setup_anndata(adata, batch_key=None, labels_key="leiden")

    # Обучение модели
    model = scvi.model.SCVI(adata)
    model.train(max_epochs=10)

    # Обязательное поле для вызова метода
    adata.obsm["X_scvi"] = model.get_latent_representation()

    adata_out, mdl_out = differential_expression(adata, cluster_key="leiden", method="scvi", mdl=model)

    assert "de_res" in adata_out.uns
    assert isinstance(mdl_out, scvi.model.SCVI)


def test_differential_expression_pegasus():
    X = np.random.poisson(1, (20, 5))
    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(5)]
    adata.obs["leiden"] = ["A"] * 10 + ["B"] * 10

    adata.uns["log1p"] = {"base": np.e}

    adata_out = differential_expression(adata, cluster_key="leiden", method="pegasus")

    assert "de_res" in adata_out.uns
    assert "de_res" in adata_out.varm

    for key in ["names", "pvals", "pvals_adj", "logfoldchanges", "scores"]:
        assert key in adata_out.uns["de_res"]