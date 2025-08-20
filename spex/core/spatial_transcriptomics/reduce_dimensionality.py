import scanpy as sc
import numpy as np
import pandas as pd
import scipy as sp

# Dependencies
import scvi
import pegasus as pg
from pegasusio import UnimodalData
from anndata import AnnData
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# This function reduces dimensionality of transcriptomic data and constructs a k-NN graph.
def reduce_dimensionality(adata, prefilter=False, method='pca', mdist=0.5, n_neighbors=None, latent_dim=None):
    """
    Reduce dimensionality of AnnData object.
    
    This function performs dimensionality reduction using various methods
    including PCA, UMAP, and t-SNE.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to reduce dimensionality.
    prefilter : bool, optional
        Whether to prefilter highly variable genes.
    method : str, optional
        Dimensionality reduction method. Options: 'pca', 'umap', 'tsne'.
    mdist : float, optional
        Minimum distance for UMAP (if method='umap').
    n_neighbors : int, optional
        Number of neighbors for UMAP/t-SNE. If None, uses default.
    latent_dim : int, optional
        Number of latent dimensions. If None, uses default.
        
    Returns
    -------
    AnnData
        Updated AnnData object with dimensionality reduction results.
        
    Notes
    -----
    - For PCA: results stored in adata.obsm['X_pca']
    - For UMAP: results stored in adata.obsm['X_umap']
    - For t-SNE: results stored in adata.obsm['X_tsne']
    - Computes neighborhood graph for UMAP/t-SNE methods
    """
    #Args:
    #adata: AnnData (after preprocessing)
    #prefilter: Whether to do PCA on only HVGs or use all genes.
    #method: PCA, scVI, and diffusion maps supported. (Diffusion map will run PCA first.)
    #n_neighbors: number of neighbors for KNN graph construction
    #latent_dim: Number of components/dimensions in reduced representation

    #Prefiltering genes and cells

    if method == "pca" or method == "diff_map":
        if latent_dim is None or latent_dim == 0:
            latent_dim = min(adata.shape[1], 20)

    if prefilter:
        print('Prefiltering variable genes...')
        adata = adata[:,adata.var.highly_variable]
        if 'blank_genes' in adata.obsm: #The original method created a "blank_threshold" equal to the 95th percentile of blank gene counts (in a cell), and thresholded all cells on that threshold.
            print('Prefiltering low expressing cells...')
            noise_counts = adata.obsm['blank_genes'].sum(1)
            adata = adata[adata.X.sum(1) > noise_counts, :] #Filter out cells with less expression than blank genes.

    #Automatic neighbor estimation
    if pd.isnull(n_neighbors):
        n_neighbors = int(np.round(np.sqrt(adata.shape[0])))

    #Component estimation from https://arxiv.org/abs/1305.5870. Use the polynomial approximation for speed.
    if pd.isnull(latent_dim):
        s = np.linalg.svd(adata.X,compute_uv=False)

        b = np.sort(adata.X.shape)
        b = b[0]/b[1]
        omega = 0.56*b**3 - 0.95*b**2 + 1.82*b + 1.43
        thresh = np.median(s)*omega*0.8 #Fudge factor to be more gentle.

        latent_dim = np.argmax(s < thresh) - 1

    adata.uns['dim_reduce'] = {
        'n_neighbors': n_neighbors,
        'latent_dim': latent_dim,
        'min_dist': mdist
    }

    print('Doing dimensionality reduction...')
    if method=='scvi':
        counts = adata.raw.to_adata()
        counts.layers["counts"] = counts.X

        if should_batch_correct(adata):
            scvi.model.SCVI.setup_anndata(counts, layer='counts', batch_key=adata.uns['batch_key'])
        else:
            scvi.model.SCVI.setup_anndata(counts, layer='counts')


        vae = scvi.model.SCVI(counts, n_layers=2, n_latent=latent_dim, gene_likelihood='nb')
        vae.train()

        adata.obsm['X_scvi'] = vae.get_latent_representation()

    elif method=='pca' or method=='diff_map':
        if latent_dim >= min(adata.shape):
            latent_dim = min(adata.shape) - 1
        sc.pp.pca(adata, n_comps=latent_dim, use_highly_variable=False)

    pdat = UnimodalData(adata)

    if should_batch_correct(adata):
        print('Batch key detected. Performing batch correcting...')
        pk = pg.run_harmony(pdat,batch=adata.uns['batch_key'])
        adata.obsm['X_pca_harmony'] = pdat.obsm['X_' + pk]
        #sc.external.pp.harmony_integrate(adata, key=adata.uns['batch_key'])
        #sc.external.pp.scanorama_integrate(adata, key=adata.uns['batch_key'])
        if method=='pca':
            method = 'pca_harmony'

    print('Calculating neighborhood graph...')
    if should_batch_correct(adata):
        #sc.pp.neighbors(adata,n_neighbors,use_rep='X_pca_harmony')
        #pg.neighbors(pdat, K=n_neighbors, rep='pca_harmony')
        pg.neighbors(pdat, K=n_neighbors, rep='pca_harmony')
    else:
        #sc.pp.neighbors(adata,n_neighbors,use_rep='X_pca')
        #pg.neighbors(pdat, K=n_neighbors, rep='pca')
        pg.tools.neighbors(pdat, K=n_neighbors, rep='pca')


    #Use pegasus to run diffusion map.
    if method == 'diff_map':
        pg.diffmap(pdat, n_components=latent_dim)
        adata.obsm['X_diffmap'] = pdat.obsm['X_diffmap']
        adata.uns['diffmap_evals'] = pdat.uns['diffmap_evals']
        adata.obsp['connectivities'] = pdat.obsp['W_pca']
    else:
        adata.obsp['connectivities'] = pdat.obsp['W_' + method]

    rep = 'pca' if method == 'diff_map' else method
    adata.obsm['neighbor_idx'] = pdat.obsm[rep + '_knn_indices']

    #Pegasus to scanpy format for neighbors
    z = sp.sparse.lil_matrix((pdat.shape[0], pdat.shape[0]))
    for i in range(pdat.shape[0]):
        z[i, pdat.obsm[rep + '_knn_indices'][i, :]] = pdat.obsm[rep + '_knn_distances'][i, :]
    z = z.tocsr()

    adata.obsm['distances'] = z

    print('UMAP embedding...')
    #sc.tl.umap(adata, min_dist=mdist)
    pg.umap(pdat, min_dist=mdist)
    adata.obsm['X_umap'] = pdat.obsm['X_umap']

    return adata


def should_batch_correct(adata):
    """
    Check if batch correction should be performed.
    
    This function checks if batch correction is needed by looking
    for batch information in the AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to check for batch information.
        
    Returns
    -------
    bool
        True if batch correction should be performed, False otherwise.
        
    Notes
    -----
    Checks for 'batch_key' in adata.uns and ensures it's not None.
    """
    if 'batch_key' in adata.uns:
        if adata.uns['batch_key']:
            return True
    return False


# def run(**kwargs):
#     adata = kwargs.get('adata')
#     prefilter = kwargs.get('prefilter', False)
#     method = kwargs.get('method', 'pca')
#     mdist = kwargs.get('mdist', 0.5)
#     n_neighbors = kwargs.get('n_neighbors')
#     latent_dim = kwargs.get('latent_dim')
#     
#     return {'adata': reduce_dimensionality(adata, prefilter, method, mdist, n_neighbors, latent_dim)}
