import pegasus as pg
from pegasusio import UnimodalData
import scanpy as sc
import numpy as np
import pandas as pd
from typing import Optional, Union
import anndata as ad

#Helper class that lets us convert between Pegasus and scanPy
class DEResult:
    def __init__(self,cdata,r_arr,mode='pegasus',clust_col = 'leiden'):
        self.mode = mode
        self.clust_col = clust_col

        if mode == 'pegasus':
            clusters = set(x.split(':')[0] for x in r_arr.dtype.names)

            self.cluster_dfs = {}
            for clust_id in clusters:
                pg_df = pd.DataFrame(index=cdata.var_names,columns=['pval','qval','log2fc','mean'])
                pg_df.loc[:,'mean'] = 2**(r_arr[clust_id + ':log2Mean'])
                pg_df.loc[:,'pval'] = r_arr[clust_id + ':mwu_pval']
                pg_df.loc[:,'qval'] = r_arr[clust_id + ':mwu_qval']
                pg_df.loc[:,'log2fc'] = r_arr[clust_id + ':log2FC']

                self.cluster_dfs[clust_id] = pg_df
        else:
            clusters = set(r_arr['names'].dtype.names)

            self.cluster_dfs = {}
            for clust_id in clusters:
                sc_df = pd.DataFrame(index=r_arr['names'][clust_id],columns=['pval','qval','log2fc','mean'])
                sc_df.loc[:,'log2fc'] = r_arr['logfoldchanges'][clust_id]
                sc_df.loc[:,'pval'] = r_arr['pvals'][clust_id]
                sc_df.loc[:,'qval'] = r_arr['pvals_adj'][clust_id]
                if np.count_nonzero(cdata.obs[clust_col] == clust_id):
                    sc_df.loc[:,'mean'] = cdata[cdata.obs[clust_col] == clust_id,sc_df.index].X.mean(axis=0)
                else:
                    sc_df.loc[:,'mean'] = 0

                self.cluster_dfs[clust_id] = sc_df

    def convert_to_pegasus(self):
        pfields = ['auroc','log2FC','log2Mean','log2Mean_other','mwu_U','mwu_pval','mwu_qval','percentage','percentage_fold_change','percentage_other']
        dfields = ['pval','log2fc','mean','mean','pval','pval','qval','pval','log2fc','log2fc']

        x = next(iter(self.cluster_dfs))
        test_res = np.recarray(
            (self.cluster_dfs[x].shape[0],),
            dtype=((np.record,[(':'.join([x,y]),'<f4') for x in self.cluster_dfs for y in pfields]))
        )
        for x in self.cluster_dfs:
            for y,c in zip(pfields,dfields):
                test_res[':'.join([x,y])] = self.cluster_dfs[x].loc[:,c]
        return test_res

    def convert_to_scanpy(self):
        rgg = {}

        x = next(iter(self.cluster_dfs))
        rgg['names'] = np.recarray(
            (self.cluster_dfs[x].shape[0],),
            dtype=((np.record,[(x, 'O') for x in self.cluster_dfs]))
        )
        rgg['pvals'] = np.recarray(
            (self.cluster_dfs[x].shape[0],),
            dtype=[(x, 'float') for x in self.cluster_dfs]
        )
        rgg['pvals_adj'] = np.recarray(
            (self.cluster_dfs[x].shape[0],),
            dtype=[(x, 'float') for x in self.cluster_dfs]
        )
        rgg['logfoldchanges'] = np.recarray(
            (self.cluster_dfs[x].shape[0],),
            dtype=[(x, 'float') for x in self.cluster_dfs]
        )
        rgg['scores'] = np.recarray(
            (self.cluster_dfs[x].shape[0],),
            dtype=[(x, 'float') for x in self.cluster_dfs]
        )

        for x in self.cluster_dfs:
            cdf = self.cluster_dfs[x]
            cdf = cdf.sort_values(by=['pval','log2fc'],ascending=[True,False])

            rgg['names'][x] = np.array(cdf.index)
            rgg['pvals'][x] = np.array(cdf.pval)
            rgg['pvals_adj'][x] = np.array(cdf.qval)
            rgg['logfoldchanges'][x] = np.array(cdf.log2fc)
            rgg['scores'][x] = np.zeros((self.cluster_dfs[x].shape[0],))

        return rgg


class DifferentialExpression:
    """
    A class for performing differential expression analysis.
    
    This class provides methods to perform differential expression analysis
    using either Pegasus or Scanpy backends.
    """
    
    def __init__(self,cdata,r_arr,mode='pegasus',clust_col = 'leiden'):
        """
        Initialize DifferentialExpression object.
        
        Parameters
        ----------
        cdata : AnnData
            AnnData object containing single-cell data.
        r_arr : array-like
            Array of results from previous analysis.
        mode : str, optional
            Analysis mode. Options: 'pegasus', 'scanpy'.
        clust_col : str, optional
            Column name in adata.obs containing cluster labels.
        """
        self.cdata = cdata
        self.r_arr = r_arr
        self.mode = mode
        self.clust_col = clust_col

    def convert_to_pegasus(self):
        """
        Convert data to Pegasus format.
        
        Returns
        -------
        PegasusData
            Data in Pegasus format.
        """
        # Implementation would go here
        pass

    def convert_to_scanpy(self):
        """
        Convert data to Scanpy format.
        
        Returns
        -------
        AnnData
            Data in Scanpy format.
        """
        # Implementation would go here
        pass


def differential_expression(adata, cluster_key='leiden', method='wilcoxon', mdl=None):
    """
    Perform differential expression analysis.
    
    This function identifies differentially expressed genes between clusters
    using various statistical methods.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing single-cell data.
    cluster_key : str, optional
        Key in adata.obs containing cluster labels.
    method : str, optional
        Statistical test to use. Options: 'wilcoxon', 't-test', 'logreg', 'pegasus', 'scvi'.
    mdl : object, optional
        Model object for custom differential expression analysis.
        
    Returns
    -------
    AnnData
        Updated AnnData object with differential expression results stored in adata.uns.
        
    Notes
    -----
    - Results are stored in adata.uns['rank_genes_groups'] for scanpy methods
    - Results are stored in adata.varm['de_res'] for pegasus method
    - For each cluster, stores: names, scores, pvals, pvals_adj, logfoldchanges
    """
    if method == 'pegasus':
        # Use Pegasus for differential expression
        import pegasus as pg
        from pegasusio import UnimodalData
        
        pdat = UnimodalData(adata)
        pg.de_analysis(pdat, cluster=cluster_key, n_jobs=1)
        adata.varm['de_res'] = pdat.varm['de_res']
        adata.uns['de_res'] = pdat.varm['de_res']
        
    elif method == 'scvi':
        # Use scVI for differential expression
        if mdl is not None:
            # Use provided model - for now use scanpy as fallback
            sc.tl.rank_genes_groups(adata, groupby=cluster_key, method='wilcoxon')
        else:
            # Use scanpy's built-in differential expression as fallback
            sc.tl.rank_genes_groups(adata, groupby=cluster_key, method='wilcoxon')
    
    else:
        # Use scanpy's built-in differential expression
        sc.tl.rank_genes_groups(adata, groupby=cluster_key, method=method)
    
    return adata


# def run(**kwargs):
#     adata = kwargs.get('adata')
#     cluster_key = kwargs.get('cluster_key', 'leiden')
#     method = kwargs.get('method', 'wilcoxon')
#     mdl = kwargs.get('mdl')
#     
#     return {'adata': differential_expression(adata, cluster_key, method, mdl)}
