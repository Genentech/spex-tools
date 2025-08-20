import numpy as np
import anndata
import re
from spex import phenograph_cluster


def test_phenograph_cluster_basic():
    # Create more separated clusters with larger distance
    X_cluster1 = np.random.normal(loc=0, scale=0.1, size=(30, 3))
    X_cluster2 = np.random.normal(
        loc=10, scale=0.1, size=(30, 3)
    )  # Increased distance from 5 to 10
    X = np.vstack([X_cluster1, X_cluster2]).astype(np.float32)

    adata = anndata.AnnData(X)
    adata.var_names = ["Target:CD3", "Cd8", "cd20_extra"]

    clustered = phenograph_cluster(
        adata=adata,
        channel_names=["cd3", "CD8", "CD20"],
        knn=6,
        transformation="none",  # Disable transformation to preserve cluster separation
        scaling="none",  # Disable scaling to preserve cluster separation
    )

    labels = clustered.obs["cluster_phenograph"]
    print(labels.value_counts())

    unique_clusters = set(labels) - {"-1"}
    assert len(unique_clusters) >= 2
