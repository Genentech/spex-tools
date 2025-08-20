import scanpy as sc
import numpy as np
import pandas as pd
import scipy as sp


def MAD_threshold(variable, ndevs=2.5):
    """
    Calculate threshold using Median Absolute Deviation (MAD).
    
    This function calculates a threshold value using the MAD method,
    which is robust to outliers.
    
    Parameters
    ----------
    variable : array-like
        Input data array.
    ndevs : float, optional
        Number of MAD deviations from the median.
        
    Returns
    -------
    float
        Threshold value calculated as median - ndevs * MAD.
        
    Notes
    -----
    MAD is calculated as median(|x - median(x)|)
    """
    mad = sp.stats.median_abs_deviation(variable, nan_policy='omit')
    return np.median(variable) - mad * ndevs


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


def preprocess(adata, scale_max=10, size_factor=None, do_QC=False):
    """
    Preprocess AnnData object for analysis.
    
    This function performs comprehensive preprocessing including quality control,
    normalization, and feature selection.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to preprocess.
    scale_max : float, optional
        Maximum value for scaling (clips larger values).
    size_factor : float, optional
        Size factor for normalization. If None, uses median library size.
    do_QC : bool, optional
        Whether to perform quality control filtering.
        
    Returns
    -------
    AnnData
        Preprocessed AnnData object.
        
    Notes
    -----
    Preprocessing steps include:
    1. Quality control (if do_QC=True)
    2. Normalization
    3. Log transformation
    4. Highly variable gene selection
    5. Scaling
    """
    if size_factor == 0:
        size_factor = None

    adata.uns['prepro'] = {}
    adata.raw = adata.copy()

    if 'blank_genes' in adata.obsm:
        print('Filtering on blank genes...')
        noise_floor = adata.obsm['blank_genes'].max(axis=1) #We assume that each "blank gene" is a n-bit barcode error and not a bit flip error.
        adata.obs['blank_counts'] = noise_floor

        adata.X = adata.X.tolil(copy=False)

        #Zero out all counts below the maximum blank gene count in the cell.
        #A more principled approach: for n blank genes, calculate distribution of FP counts per gene.
        #Then we calculate P(count = obs.) for FP distribution and discard counts with P > 0.05
        cix,_ = adata.X.nonzero()
        for c in np.unique(cix):
            if noise_floor[c] < 1:
                continue
            crow = adata.X.getrowview(c)
            adata.X[c,:] = crow.multiply(crow > noise_floor[c])

        adata.X = adata.X.tocsr(copy=False)

    print('Excluding bad cells and genes (singletons)...')
    sc.pp.filter_cells(adata, min_genes=2)
    sc.pp.filter_genes(adata, min_cells=2)

    #Total counts, number of genes by counts
    sc.pp.calculate_qc_metrics(adata,percent_top=None,log1p=False,inplace=True)
    #Automatically exclude genes that have a low amount of transcripts.
    if do_QC:
        adata.uns['prepro']['total_count_thresh'] = MAD_threshold(adata.obs.total_counts,1)
        adata.uns['prepro']['n_genes_by_count_thresh'] = MAD_threshold(adata.obs.n_genes_by_counts,1.5)
        adata = adata[adata.obs.n_genes_by_counts > adata.uns['prepro']['n_genes_by_count_thresh'],:]
        adata = adata[adata.obs.total_counts > adata.uns['prepro']['total_count_thresh'],:]

    print('Normalizing data...')

    adata.layers['counts'] = adata.X.copy()
    if pd.isnull(size_factor):
        adata.uns['prepro']['size_factor'] = np.median(adata.obs.total_counts)
        sc.pp.normalize_total(adata,target_sum=None)
    else:
        sc.pp.normalize_total(adata,target_sum=size_factor)

    sc.pp.log1p(adata)
    adata.uns['log1p']['base'] = None

    #Annotate highly variable genes (take batch correction into account)
    if should_batch_correct(adata):
        adata.obs[adata.uns['batch_key']] = pd.Categorical(adata.obs[adata.uns['batch_key']])
        sc.pp.highly_variable_genes(adata,batch_key=adata.uns['batch_key'])
    else:
        sc.pp.highly_variable_genes(adata)

    #Z-transform data and truncate.
    sc.pp.scale(adata, max_value=scale_max)

    return adata


# def run(**kwargs):
#     adata = kwargs.get('adata')
#     scale_max = kwargs.get('scale_max', 10)
#     size_factor = kwargs.get('size_factor')
#     do_QC = True if kwargs.get('do_qc') == 'true' else False

#     return {'adata': preprocess(adata, scale_max, size_factor, do_QC)}
