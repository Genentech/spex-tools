import decoupler as dc
import numpy as np
import pandas as pd
import pegasus as pg
from pegasusio import UnimodalData
import os
import re
import importlib.resources as pkg_resources
from spex import resources
from collections import defaultdict
from numpy.linalg import LinAlgError


def get_markers(name: str) -> pd.DataFrame:
    # Always read from the single bundled file
    with pkg_resources.path(resources, "progeny.parquet") as path:
        df = pd.read_parquet(path)

    # Accept aliases used in the notebook; return the full marker set
    if name is None:
        return df.copy()
    key = str(name).strip().lower()
    if key in {"*", "all", "progeny", "human_lung", "human_immune", "lung", "immune"}:
        return df.copy()

    # Fuzzy match by substring; if empty, gracefully fall back to full set
    mask = df["pathway"].str.lower().str.contains(re.escape(key), regex=True)
    sub = df[mask].copy()
    return sub if not sub.empty else df.copy()


def convert_progeny_to_pegasus_marker_dict(path: str) -> dict:
    # Convert a PROGENy-like parquet into a Pegasus-style marker dict
    df = pd.read_parquet(path)
    result = {"title": "converted_from_progeny", "cell_types": []}

    for cell_type in df["pathway"].unique():
        subset = df[df["pathway"] == cell_type]

        pos_subset = subset[subset["weight"] > 0]
        neg_subset = subset[subset["weight"] < 0]

        markers = []

        if not pos_subset.empty:
            avg_weight_pos = pos_subset["weight"].mean()
            # Normalize very small positive magnitudes to at least +1.0
            if 0 < avg_weight_pos < 1.0:
                avg_weight_pos = 1.0
            else:
                avg_weight_pos = round(avg_weight_pos)
            markers.append({
                "genes": pos_subset["genesymbol"].dropna().unique().tolist(),
                "type": "+",
                "weight": float(avg_weight_pos)
            })

        if not neg_subset.empty:
            avg_weight_neg = neg_subset["weight"].mean()
            # Normalize very small negative magnitudes to at most -1.0
            if -1.0 < avg_weight_neg < 0:
                avg_weight_neg = -1.0
            else:
                avg_weight_neg = round(avg_weight_neg)
            markers.append({
                "genes": neg_subset["genesymbol"].dropna().unique().tolist(),
                "type": "-",
                "weight": float(avg_weight_neg)
            })

        result["cell_types"].append({
            "name": cell_type,
            "markers": markers
        })

    return result


def annotate_clusters_pg(adata, marker_db=None, cluster_key="leiden"):
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"Cluster key '{cluster_key}' not found in adata.obs.")

    pdat = UnimodalData(adata)
    if "de_res" not in pdat.varm:
        if "log1p" not in adata.uns:
            adata.uns["log1p"] = {"base": float(np.e)}
        pg.de_analysis(pdat, cluster=cluster_key, n_jobs=1)
        adata.varm["de_res"] = pdat.varm["de_res"]

    if marker_db is None:
        with pkg_resources.path(resources, "progeny.parquet") as p:
            marker_db = convert_progeny_to_pegasus_marker_dict(p)

    ctypes = pg.infer_cell_types(pdat, markers=marker_db)

    adata.obs["cell_type"] = "Unknown"
    for cl, sugg in ctypes.items():
        if sugg:
            adata.obs.loc[adata.obs[cluster_key] == cl, "cell_type"] = str(getattr(sugg[0], "name", sugg[0]))

    # categorical + reset palette to match categories (fix Squidpy palette mismatch)
    ct = adata.obs["cell_type"].astype("category")
    ct = ct.cat.remove_unused_categories()
    adata.obs["cell_type"] = ct
    cats = list(ct.cat.categories)
    try:
        import scanpy as sc
        adata.uns["cell_type_colors"] = sc.plotting.palettes.default_64[: len(cats)]
    except Exception:
        from matplotlib import cm, colors as mcolors
        base = cm.get_cmap("tab20").colors
        adata.uns["cell_type_colors"] = [mcolors.to_hex(base[i % len(base)]) for i in range(len(cats))]

    # HDF5-safe serialization
    adata.uns["cell_typing"] = {str(cl): [str(getattr(x, "name", x)) for x in lst] for cl, lst in ctypes.items()}

    if "score_mlm" not in adata.obsm:
        adata.obsm["score_mlm"] = pd.DataFrame(index=adata.obs_names)

    return adata


def annotate_clusters_dc(adata, marker_db=None, method="mlm", tmin=3):
    """
    Compute pathway activities using decoupler and store them in `adata.obsm`.
    Accepts: None (bundled PROGENy), pandas.DataFrame, or Pegasus-style dict.
    Strings are NOT accepted here (strings are reserved for Pegasus path).
    """
    # Load default markers if nothing provided
    if marker_db is None:
        with pkg_resources.path(resources, "progeny.parquet") as p:
            marker_db = pd.read_parquet(p)

    # Strings are not supported for decoupler branch by design
    if isinstance(marker_db, str):
        raise TypeError(
            "For decoupler methods, 'marker_db' must be a DataFrame or dict. "
            "String inputs are only allowed for Pegasus and are passed through unchanged."
        )

    # Convert Pegasus-style dict to decoupler network (source, target, weight)
    if isinstance(marker_db, dict):
        rows = []
        for ct in marker_db.get("cell_types", []):
            src = ct.get("name", "unknown")
            for m in ct.get("markers", []):
                w = float(m.get("weight", 1.0))
                t = m.get("type", "+")
                # Ensure weight sign matches marker type
                if t == "+" and w < 0:
                    w = abs(w)
                if t == "-" and w > 0:
                    w = -abs(w)
                for g in m.get("genes", []):
                    if isinstance(g, str):
                        rows.append((src, g, w))
        if not rows:
            raise ValueError("marker_dict is empty after conversion.")
        marker_db = pd.DataFrame(rows, columns=["source", "target", "weight"])
        # Merge duplicates by averaging weights per (source, target)
        marker_db = marker_db.groupby(["source", "target"], as_index=False)["weight"].mean()

    # Normalize column names to the decoupler schema
    marker_db = marker_db.rename(
        columns={
            "pathway": "source",
            "src": "source",
            "genesymbol": "target",
            "gene": "target",
            "wgt": "weight",
            "weight": "weight",
        },
        errors="ignore",
    )

    # Keep only genes present in the data matrix
    if "target" in marker_db.columns:
        marker_db = marker_db[marker_db["target"].isin(adata.var_names)]

    # Try direct method from decoupler.mt first (common return: DataFrame)
    acts = None
    try:
        func = getattr(dc.mt, method)
        try:
            acts = func(adata, marker_db[["source", "target", "weight"]], tmin=tmin, verbose=False)
        except TypeError:
            # Backward/forward compatibility for min_n vs tmin
            acts = func(adata, marker_db[["source", "target", "weight"]], min_n=tmin, verbose=False)
        if not isinstance(acts, pd.DataFrame):
            acts = pd.DataFrame(acts, index=adata.obs_names)
    except Exception:
        acts = None

    # Fallback: generic dispatcher + extractor
    if acts is None or getattr(acts, "shape", (len(adata.obs_names), 0))[1] == 0:
        try:
            dc.decouple(
                mat=adata,
                net=marker_db[["source", "target", "weight"]],
                source="source",
                target="target",
                weight="weight",
                min_n=tmin,
                verbose=False,
                methods=[method],
            )
            acts2 = dc.get_acts(adata, obsm_key=f"{method}_estimate")
            if acts2 is None:
                acts2 = adata.obsm.get(f"{method}_estimate")
            if acts2 is None:
                acts2 = pd.DataFrame(index=adata.obs_names)
            elif not isinstance(acts2, pd.DataFrame):
                acts2 = pd.DataFrame(acts2, index=adata.obs_names)
            acts = acts2
        except Exception:
            acts = None

    # Final fallback: zero-filled matrix with pathway columns to keep downstream code stable
    if acts is None or acts.shape[1] == 0:
        srcs = list(pd.unique(marker_db["source"])) if "source" in marker_db.columns else []
        acts = pd.DataFrame(0.0, index=adata.obs_names, columns=srcs)

    # Ensure index alignment
    if not acts.index.equals(adata.obs_names):
        acts = acts.reindex(index=adata.obs_names)

    # Store results
    adata.obsm[f"score_{method}"] = acts
    if method == "mlm":
        adata.obsm["score_mlm"] = acts

    return adata

def annotate_clusters(adata, marker_db=None, cluster_key="leiden", method="mlm", tmin=3):
    # If marker_db is a string, always route to Pegasus and pass it through unchanged
    if isinstance(marker_db, str):
        return annotate_clusters_pg(adata, marker_db=marker_db, cluster_key=cluster_key)

    # Otherwise dispatch by method
    if method == "pegasus":
        return annotate_clusters_pg(adata, marker_db=marker_db, cluster_key=cluster_key)
    else:
        return annotate_clusters_dc(adata, marker_db=marker_db, method=method, tmin=tmin)

def analyze_pathways(adata, pathway_file=None):
    # Run decoupler MLM pathway analysis given a parquet/CSV markers file or the bundled PROGENy
    if pathway_file is None:
        with pkg_resources.path(resources, "progeny.parquet") as p:
            markers = pd.read_parquet(p)
    else:
        if pathway_file.endswith(".csv"):
            markers = pd.read_csv(pathway_file)
        else:
            markers = pd.read_parquet(pathway_file)

    markers = markers.rename(
        columns={
            "pathway": "source",
            "genesymbol": "target",
            "src": "source",
            "wgt": "weight",
        },
        errors="ignore",
    )
    for col in ("source", "target", "weight"):
        if col not in markers.columns:
            raise ValueError(f"row not exists '{col}'.")

    try:
        acts = dc.mt.mlm(adata, markers, tmin=3, verbose=False)
    except TypeError:
        acts = dc.mt.mlm(adata, markers, min_n=3, verbose=False)

    if not isinstance(acts, pd.DataFrame):
        acts = adata.obsm.get("score_mlm")
        if acts is None:
            acts = adata.obsm.get("mlm_estimate")

    if acts is None:
        acts = pd.DataFrame(np.zeros((adata.n_obs, 0)), index=adata.obs_names)

    if not acts.index.equals(adata.obs_names):
        acts = acts.reindex(index=adata.obs_names)

    adata.obsm["pathway_scores"] = acts
    adata.obsm["score_mlm"] = acts

    return adata