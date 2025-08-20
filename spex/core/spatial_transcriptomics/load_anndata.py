import scanpy as sc


def load_anndata(path=None, files=None):
    """
    Load AnnData objects from file(s).
    
    This function loads AnnData objects from single or multiple files.
    When loading multiple files, they are combined into a single AnnData object.
    
    Parameters
    ----------
    path : str, optional
        Path to a single AnnData file (.h5ad).
    files : list, optional
        List of file paths to load and combine.
        
    Returns
    -------
    dict
        Dictionary containing the loaded AnnData object(s):
        - 'adata': AnnData object (single file) or combined AnnData object (multiple files)
        
    Notes
    -----
    - If both path and files are provided, files takes precedence
    - When combining multiple files, a 'filename' column is added to adata.obs
    - Files are combined using scanpy.concat with outer join
    """
    if files:
        adatas = []
        for file in files:
            adata = sc.read_h5ad(file)
            adata.obs['filename'] = file.split('\\')[-1]
            adatas.append(adata)
        combined_adata = sc.concat(adatas, axis=0, join='outer', label='filename', index_unique='-')
        print('combined')
        return {'adata': combined_adata}
    else:
        adata = sc.read_h5ad(path)
        return {'adata': adata}