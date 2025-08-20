import numpy as np
import pandas as pd
from anndata import AnnData
import squidpy as sq
from numba import njit, prange
import time

@njit(parallel=True, fastmath=True)
def _count_neighborhood_vectors(indices, indptr, cell_types, n_clust):
    """
    Count neighborhood vectors for each cell type.
    
    This is a Numba-optimized function that counts the number of neighbors
    of each cell type for every cell.
    
    Parameters
    ----------
    indices : array
        Indices of neighbors for each cell.
    indptr : array
        Pointer array for sparse neighbor matrix.
    cell_types : array
        Cell type labels for each cell.
    n_clust : int
        Number of unique cell types.
        
    Returns
    -------
    array
        Neighborhood count matrix of shape (n_cells, n_clust).
    """
    n_cells = indptr.shape[0] - 1
    result = np.zeros((n_cells, n_clust), dtype=np.float32)

    for i in prange(n_cells):
        start, end = indptr[i], indptr[i+1]
        neighbors = indices[start:end]

        if len(neighbors) > 0:
            for neighbor in neighbors:
                cell_type = cell_types[neighbor]
                result[i, cell_type] += 1

    return result

@njit(parallel=True)
def _calculate_global_clq(local_clq, cell_types, n_clust):
    """
    Calculate global CLQ values.
    
    This function computes the global co-localization quotient
    by averaging local CLQ values for each cell type.
    
    Parameters
    ----------
    local_clq : array
        Local CLQ values for each cell.
    cell_types : array
        Cell type labels for each cell.
    n_clust : int
        Number of unique cell types.
        
    Returns
    -------
    array
        Global CLQ matrix of shape (n_clust, n_clust).
    """
    global_clq = np.zeros((n_clust, n_clust), dtype=np.float32)

    for cell_type in prange(n_clust):
        count = 0
        sum_values = np.zeros(n_clust, dtype=np.float32)

        for cell in range(local_clq.shape[0]):
            if cell_types[cell] == cell_type:
                sum_values += local_clq[cell]
                count += 1

        if count > 0:
            global_clq[cell_type] = sum_values / count

    return global_clq


def CLQ_vec_numba(adata, clust_col='leiden', clust_uniq=None, radius=50, n_perms=1000):
    """
    Calculate Co-Localization Quotient (CLQ) using Numba optimization.
    
    This function computes the co-localization quotient between different cell types
    in spatial data. CLQ measures whether cell types are attracted to or avoid each other
    compared to random spatial distribution.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing spatial data and cluster labels.
    clust_col : str, optional
        Column name in adata.obs containing cluster/cell type labels.
    clust_uniq : array-like, optional
        Unique cluster labels. If None, will be inferred from adata.obs[clust_col].
    radius : float, optional
        Radius for spatial neighbor calculation.
    n_perms : int, optional
        Number of permutations for significance testing.
        
    Returns
    -------
    tuple
        A tuple containing:
        - adata_out : AnnData
            Updated AnnData object with CLQ results in adata.obsm and adata.uns
        - results : dict
            Dictionary containing CLQ results:
            - 'global_clq': Global CLQ matrix
            - 'permute_test': Permutation test p-values
            - 'local_clq': Local CLQ values for each cell
            - 'NCV': Neighborhood count vectors
            
    Notes
    -----
    - CLQ > 1 indicates attraction between cell types
    - CLQ < 1 indicates avoidance between cell types
    - CLQ = 1 indicates random spatial distribution
    - Results are stored in adata.obsm['local_clq'] and adata.obsm['NCV']
    - Global results are stored in adata.uns['CLQ']
    """
    start_time = time.time()
    if 'spatial' not in adata.obsm:
        # Create spatial coordinates from x_coordinate and y_coordinate
        adata.obsm['spatial'] = adata.obs[['x_coordinate', 'y_coordinate']].to_numpy()

    # Preprocess spatial neighbors
    radius = float(radius)
    sq.gr.spatial_neighbors(adata, coord_type='generic', radius=radius)
    neigh_idx = adata.obsp['spatial_connectivities'].tocsr()
    indices = neigh_idx.indices.astype(np.int32)
    indptr = neigh_idx.indptr.astype(np.int32)

    # Map and prepare cell types
    global_cluster_freq = adata.obs[clust_col].value_counts(normalize=True)
    label_dict = {x: i for i, x in enumerate(global_cluster_freq.index)}
    n_clust = len(label_dict)
    n_cells = adata.shape[0]
    cell_types = np.array([label_dict[x] for x in adata.obs[clust_col]], dtype=np.int32)

    # Calculate observed neighborhood vectors and local CLQ
    observed_ncv = _count_neighborhood_vectors(indices, indptr, cell_types, n_clust)
    neighborhood_sizes = np.sum(observed_ncv, axis=1, keepdims=True)
    norm_ncv = np.divide(observed_ncv, neighborhood_sizes, out=np.zeros_like(observed_ncv), where=neighborhood_sizes > 0)
    global_freqs = np.array([global_cluster_freq[x] for x in label_dict])
    global_freqs_adj = np.where(global_freqs > 0, global_freqs, 1.0)
    local_clq = norm_ncv / global_freqs_adj

    # Calculate observed global CLQ
    global_clq = _calculate_global_clq(local_clq, cell_types, n_clust)

    # Initialize permutation test results
    permute_counts = np.zeros((n_clust, n_clust), dtype=np.int32)

    # Process permutations in batches
    batch_size = 100  # Adjust batch size based on available memory and performance
    for batch_start in range(0, n_perms, batch_size):
        batch_end = min(batch_start + batch_size, n_perms)
        for perm in range(batch_start, batch_end):
            permuted_cell_types = np.random.permutation(cell_types)
            permuted_ncv = _count_neighborhood_vectors(indices, indptr, permuted_cell_types, n_clust)
            perm_neighborhood_sizes = np.sum(permuted_ncv, axis=1, keepdims=True)
            perm_norm_ncv = np.divide(permuted_ncv, perm_neighborhood_sizes, out=np.zeros_like(permuted_ncv), where=perm_neighborhood_sizes > 0)
            perm_local_clq = perm_norm_ncv / global_freqs_adj

            # Calculate global CLQ for this permutation
            perm_global_clq = _calculate_global_clq(perm_local_clq, permuted_cell_types, n_clust)

            # Update permutation counts
            for cell_type in range(n_clust):
                permute_counts[cell_type] += (perm_global_clq[cell_type] < global_clq[cell_type]).astype(np.int32)

    idx = list(label_dict.keys())

    # Normalize permutation counts and convert to labels
    clq_perm_avoid = permute_counts / n_perms
    clq_perm_attr = 1 - clq_perm_avoid

    # Store results in AnnData object
    adata.obsm['local_clq'] = local_clq
    adata.obsm['NCV'] = observed_ncv
    adata.uns['CLQ'] = {
        'global_clq': global_clq,
        'permute_test': clq_perm_attr,
        'cell_types': idx
    }

    results = {
        'global_clq': global_clq,
        'permute_test': clq_perm_attr,
        'local_clq': local_clq,
        'NCV': observed_ncv,
        'cell_types': idx
    }

    return adata, results