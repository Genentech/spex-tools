import numpy as np
import anndata
import re
from spex import phenograph_cluster


def test_phenograph_cluster_basic():
    X_cluster1 = np.random.normal(loc=0, scale=0.1, size=(30, 3))
    X_cluster2 = np.random.normal(loc=5, scale=0.1, size=(30, 3))
    X = np.vstack([X_cluster1, X_cluster2]).astype(np.float32)

    adata = anndata.AnnData(X)
    adata.var_names = ["Target:CD3", "Cd8", "cd20_extra"]

    clustered = phenograph_cluster(
        adata=adata,
        channel_names=["cd3", "CD8", "CD20"],
        knn=6,
        transformation="arcsin",
        scaling="z-score"
    )

    labels = clustered.obs["cluster_phenograph"]
    print(labels.value_counts())

    unique_clusters = set(labels) - {"-1"}
    assert len(unique_clusters) >= 2