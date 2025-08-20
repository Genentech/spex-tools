import scanpy as sc


def cluster(adata, spatial_weight=0.0, resolution=1.0, method='leiden'):
    """
    Perform clustering with optional spatial weights.
    
    This function performs clustering on AnnData objects with optional spatial
    neighbor weights. It supports both Leiden and Louvain clustering algorithms.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object after preprocessing and dimensionality reduction.
    spatial_weight : float, optional
        Weight for spatial neighbors in adjacency calculation. 
        If > 0, spatial neighbors are forced to be close.
    resolution : float, optional
        Resolution of the modularity cost function. 
        Lower values result in fewer clusters, higher values in more clusters.
    method : str, optional
        Clustering algorithm to use. Options: 'leiden', 'louvain'.
        
    Returns
    -------
    AnnData
        Updated AnnData object with clustering results in adata.obs.
        
    Notes
    -----
    - Requires 'connectivities' in adata.obsp
    - For spatial clustering, requires 'spatial_connectivities' in adata.obsp
    - Results are stored in adata.obs with the method name as column
    """
    #Clustering
    adjacency = adata.obsp['connectivities']

    #Force spatial neighbors to be close.
    if spatial_weight > 0:
        if 'spatial_connectivities' in adata.obsp:
            adjacency += adata.obsp['spatial_connectivities']*spatial_weight

    #Pegasus
    #pdat = UnimodalData(adata)
    #pdat.obsp['W_pca'] = pdat.obsp['connectivities']
    #pg.cluster(pdat,algo=method)

    #Scanpy
    if method == 'leiden':
        sc.tl.leiden(adata, resolution=resolution, adjacency=adjacency)
    elif method == 'louvain':
        sc.tl.louvain(adata, resolution=resolution, adjacency=adjacency)

    return adata


# def run(**kwargs):
#     adata = kwargs.get('adata')

#     swgt = kwargs.get('spatial_weight')
#     method = kwargs.get('method')
#     res = kwargs.get('resolution')

#     return {'adata': cluster(adata, swgt, res, method)}