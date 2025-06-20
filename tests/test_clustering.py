import numpy as np
import anndata
import re
from spex import phenograph_cluster

def test_phenograph_cluster_basic():
    X_cluster1 = np.random.normal(loc=0, scale=0.1, size=(30, 3))
    X_cluster2 = np.random.normal(loc=5, scale=0.1, size=(30, 3))
    X = np.vstack([X_cluster1, X_cluster2]).astype(np.float32)

    adata = anndata.AnnData(X)
    adata.var_names = ["CD3", "CD8", "CD20"]

    def normalize(name: str) -> str:
        return re.sub(r"[^0-9a-zA-Z]", "", name).lower().replace("target", "")

    adata.uns["channel_index_map"] = {normalize(ch): i for i, ch in enumerate(adata.var_names)}

    clustered = phenograph_cluster(
        adata=adata,
        channel_names=["CD3", "CD8", "CD20"],
        knn=6,
        transformation="arcsin",
        scaling="z-score"
    )

    labels = clustered.obs["cluster_phenograph"]
    print(labels.value_counts())
    unique_clusters = set(labels) - {"-1"}
    assert len(unique_clusters) >= 2
