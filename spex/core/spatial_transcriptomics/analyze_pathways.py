import decoupler as dc
import pandas as pd
import pegasus as pg
from pegasusio import UnimodalData
import os
import importlib.resources as pkg_resources
from spex import resources


def annotate_clusters(adata, marker_db=None, cluster_key='leiden', method='pegasus'):
    #Args:
    #adata: the data
    #cluster_key: which key in adata.obs tells you the cluster a cell belongs to
    #marker_db: Either a string for pegasus, or a DataFrame that labels source ('gene') to target ('cell-type').
    #method: Pegasus will use DEGs; scanpy data will use decouplr

    if marker_db is None:
        with pkg_resources.path(resources, "progeny.parquet") as p:
            markers = pd.read_parquet(p)
            markers = markers.rename(columns={"pathway": "source", "genesymbol": "target"})

    if method == 'pegasus':
        if isinstance(adata, UnimodalData):
            pdat = adata
        else:
            pdat = UnimodalData(adata)
        ctypes = pg.infer_cell_types(pdat,markers=marker_db)

        adata.obs['cell_type'] = 'Unknown'
        for cluster in ctypes:
            if len(ctypes[cluster]) == 0:
                continue
            try:
                adata.obs.loc[adata.obs[cluster_key] == cluster,'cell_type'] = ctypes[cluster][0].name
            except:
                cluster_key = 'louvain'
                adata.obs.loc[adata.obs[cluster_key] == cluster,'cell_type'] = ctypes[cluster][0].name

    else:
        dc.decouple(
            adata,
            marker_db,
            source='src',
            target='genesymbol',
            weight='wgt',
            min_n=3,
            verbose=False,
            methods=[method]
        )

    return adata



def analyze_pathways(adata, pathway_file=None):
    import os
    import pandas as pd
    import decoupler as dc

    if pathway_file is None:
        with pkg_resources.path(resources, "progeny.parquet") as p:
            markers = pd.read_parquet(p)
    else:
        if pathway_file.endswith(".csv"):
            markers = pd.read_csv(pathway_file)
        else:
            markers = pd.read_parquet(pathway_file)

    markers = markers.rename(columns={"pathway": "source", "genesymbol": "target"})

    result = dc.mt.mlm(
        adata,
        markers,
        tmin=3,
        verbose=True,
    )

    acts = adata.obsm['score_mlm']
    adata.obsm['pathway_scores'] = acts

    mean_acts = acts.groupby(adata.obs['cell_type']).mean()

    return adata


# def run(**kwargs):
#     adata = kwargs.get('adata')

#     adata = annotate_clusters(adata, marker_db='human_immune')
#     adata = analyze_pathways(adata)

#     return {'adata': adata}
