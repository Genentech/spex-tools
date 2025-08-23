# 📦 Batch Processing Example

## Overview

This example demonstrates how to process multiple images in batch using SPEX. We'll create a pipeline that can handle multiple samples, apply consistent processing parameters, and combine results for downstream analysis.

## Prerequisites

```python
import spex as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from pathlib import Path
import os
from tqdm import tqdm

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
sc.settings.verbosity = 3
```

## Step 1: Organize Data Structure

### Create Sample Directory Structure

```python
# Define data directory structure
data_dir = Path("batch_data")
sample_dirs = {
    "sample_001": data_dir / "sample_001",
    "sample_002": data_dir / "sample_002", 
    "sample_003": data_dir / "sample_003",
    "sample_004": data_dir / "sample_004"
}

# Create directories if they don't exist
for sample_dir in sample_dirs.values():
    sample_dir.mkdir(parents=True, exist_ok=True)

print("Sample directories created:")
for sample_name, sample_path in sample_dirs.items():
    print(f"  {sample_name}: {sample_path}")
```

### Sample Information

```python
# Define sample metadata
sample_info = pd.DataFrame({
    'sample_id': ['sample_001', 'sample_002', 'sample_003', 'sample_004'],
    'condition': ['control', 'control', 'treatment', 'treatment'],
    'replicate': [1, 2, 1, 2],
    'image_file': ['image_001.tiff', 'image_002.tiff', 'image_003.tiff', 'image_004.tiff'],
    'expected_cells': [1000, 1200, 800, 900]
})

print("Sample information:")
print(sample_info)
```

## Step 2: Define Processing Functions

### Image Processing Function

```python
def process_single_image(image_path, sample_id, processing_params):
    """
    Process a single image with consistent parameters.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    sample_id : str
        Sample identifier
    processing_params : dict
        Processing parameters
        
    Returns:
    --------
    dict
        Processing results
    """
    
    print(f"Processing {sample_id}...")
    
    # Load image
    Image, channel = sp.load_image(image_path)
    
    # Preprocess
    Image_bg = sp.background_subtract(Image, list(range(len(channel))), 
                                     radius=processing_params['bg_radius'])
    Image_denoised = sp.nlm_denoise(Image_bg, list(range(len(channel))), 
                                   h=processing_params['nlm_h'])
    
    # Segment
    labels = sp.cellpose_cellseg(Image_denoised, [0], 
                                diameter=processing_params['cell_diameter'])
    
    # Clean segmentation
    labels_clean = sp.remove_small_objects(labels, 
                                          min_size=processing_params['min_size'])
    labels_clean = sp.remove_large_objects(labels_clean, 
                                          max_size=processing_params['max_size'])
    
    # Extract features
    adata = sp.feature_extraction_adata(Image_denoised, labels_clean, channel)
    
    # Add metadata
    adata.obs['sample_id'] = sample_id
    adata.obs['area'] = sp.get_cell_areas(labels_clean)
    adata.obsm['spatial'] = sp.get_cell_coordinates(labels_clean)
    
    # Add cell properties
    properties = sp.get_cell_properties(Image_denoised, labels_clean, 
                                       list(range(len(channel))))
    adata.obs['shape_factor'] = properties['shape_factors']
    adata.obs['eccentricity'] = properties['eccentricities']
    
    return {
        'adata': adata,
        'labels': labels_clean,
        'image_shape': Image.shape,
        'n_cells': labels_clean.max(),
        'channels': channel
    }
```

### Batch Processing Function

```python
def process_batch(sample_info, processing_params, output_dir):
    """
    Process multiple samples in batch.
    
    Parameters:
    -----------
    sample_info : pd.DataFrame
        Sample information dataframe
    processing_params : dict
        Processing parameters
    output_dir : str
        Output directory for results
        
    Returns:
    --------
    dict
        Batch processing results
    """
    
    results = {}
    failed_samples = []
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each sample
    for _, sample in tqdm(sample_info.iterrows(), total=len(sample_info), 
                         desc="Processing samples"):
        
        sample_id = sample['sample_id']
        image_file = sample['image_file']
        
        try:
            # Process image
            result = process_single_image(image_file, sample_id, processing_params)
            
            # Save individual results
            sample_output_dir = output_path / sample_id
            sample_output_dir.mkdir(exist_ok=True)
            
            # Save AnnData
            result['adata'].write(sample_output_dir / f"{sample_id}.h5ad")
            
            # Save segmentation labels
            import tifffile
            tifffile.imwrite(sample_output_dir / f"{sample_id}_labels.tiff", 
                           result['labels'])
            
            # Save metadata
            metadata = {
                'sample_id': sample_id,
                'condition': sample['condition'],
                'replicate': sample['replicate'],
                'n_cells': result['n_cells'],
                'image_shape': result['image_shape'],
                'channels': result['channels']
            }
            
            pd.DataFrame([metadata]).to_csv(sample_output_dir / f"{sample_id}_metadata.csv", 
                                           index=False)
            
            results[sample_id] = result
            print(f"✓ {sample_id}: {result['n_cells']} cells processed")
            
        except Exception as e:
            print(f"✗ {sample_id}: Failed - {str(e)}")
            failed_samples.append(sample_id)
    
    return results, failed_samples
```

## Step 3: Define Processing Parameters

### Consistent Parameters

```python
# Define processing parameters for all samples
processing_params = {
    'bg_radius': 50,        # Background subtraction radius
    'nlm_h': 10,           # NLM denoising strength
    'cell_diameter': 30,   # Expected cell diameter
    'min_size': 50,        # Minimum cell size
    'max_size': 1000,      # Maximum cell size
    'flow_threshold': 0.4, # Cellpose flow threshold
    'cellprob_threshold': 0 # Cellpose probability threshold
}

print("Processing parameters:")
for param, value in processing_params.items():
    print(f"  {param}: {value}")
```

## Step 4: Run Batch Processing

### Execute Processing

```python
# Run batch processing
output_dir = "batch_results"
results, failed_samples = process_batch(sample_info, processing_params, output_dir)

print(f"\nBatch processing completed:")
print(f"  Successful: {len(results)} samples")
print(f"  Failed: {len(failed_samples)} samples")

if failed_samples:
    print(f"  Failed samples: {failed_samples}")
```

### Processing Summary

```python
# Create processing summary
summary_data = []
for sample_id, result in results.items():
    sample_row = sample_info[sample_info['sample_id'] == sample_id].iloc[0]
    
    summary_data.append({
        'sample_id': sample_id,
        'condition': sample_row['condition'],
        'replicate': sample_row['replicate'],
        'n_cells_detected': result['n_cells'],
        'n_cells_expected': sample_row['expected_cells'],
        'image_shape': str(result['image_shape']),
        'n_channels': len(result['channels'])
    })

summary_df = pd.DataFrame(summary_data)
print("\nProcessing Summary:")
print(summary_df)

# Save summary
summary_df.to_csv(f"{output_dir}/processing_summary.csv", index=False)
```

## Step 5: Quality Control

### Batch QC Metrics

```python
def batch_quality_control(results, sample_info):
    """
    Perform quality control across all samples.
    """
    
    qc_metrics = {}
    
    for sample_id, result in results.items():
        adata = result['adata']
        
        # Basic metrics
        qc_metrics[sample_id] = {
            'n_cells': adata.n_obs,
            'n_features': adata.n_vars,
            'mean_area': adata.obs['area'].mean(),
            'std_area': adata.obs['area'].std(),
            'min_area': adata.obs['area'].min(),
            'max_area': adata.obs['area'].max(),
            'mean_shape_factor': adata.obs['shape_factor'].mean(),
            'missing_values': np.isnan(adata.X).sum(),
            'zero_values': (adata.X == 0).sum()
        }
    
    return pd.DataFrame(qc_metrics).T

# Run QC
qc_df = batch_quality_control(results, sample_info)
print("\nQuality Control Metrics:")
print(qc_df)

# Save QC results
qc_df.to_csv(f"{output_dir}/quality_control.csv")
```

### QC Visualization

```python
# Create QC plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Cell count by sample
sample_ids = list(results.keys())
cell_counts = [results[sid]['n_cells'] for sid in sample_ids]
axes[0, 0].bar(sample_ids, cell_counts, color='skyblue', alpha=0.7)
axes[0, 0].set_title('Cell Count by Sample')
axes[0, 0].tick_params(axis='x', rotation=45)

# Cell area distribution
all_areas = []
area_labels = []
for sample_id, result in results.items():
    areas = result['adata'].obs['area']
    all_areas.extend(areas)
    area_labels.extend([sample_id] * len(areas))

area_df = pd.DataFrame({'sample_id': area_labels, 'area': all_areas})
sns.boxplot(data=area_df, x='sample_id', y='area', ax=axes[0, 1])
axes[0, 1].set_title('Cell Area Distribution')
axes[0, 1].tick_params(axis='x', rotation=45)

# Shape factor distribution
all_shapes = []
shape_labels = []
for sample_id, result in results.items():
    shapes = result['adata'].obs['shape_factor']
    all_shapes.extend(shapes)
    shape_labels.extend([sample_id] * len(shapes))

shape_df = pd.DataFrame({'sample_id': shape_labels, 'shape_factor': all_shapes})
sns.boxplot(data=shape_df, x='sample_id', y='shape_factor', ax=axes[0, 2])
axes[0, 2].set_title('Shape Factor Distribution')
axes[0, 2].tick_params(axis='x', rotation=45)

# Expression distribution
all_expression = []
for sample_id, result in results.items():
    expression = result['adata'].X.flatten()
    all_expression.extend(expression)

axes[1, 0].hist(all_expression, bins=50, alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Expression Level')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Overall Expression Distribution')

# Missing values
missing_counts = [qc_df.loc[sid, 'missing_values'] for sid in sample_ids]
axes[1, 1].bar(sample_ids, missing_counts, color='lightcoral', alpha=0.7)
axes[1, 1].set_title('Missing Values by Sample')
axes[1, 1].tick_params(axis='x', rotation=45)

# Zero values
zero_counts = [qc_df.loc[sid, 'zero_values'] for sid in sample_ids]
axes[1, 2].bar(sample_ids, zero_counts, color='lightgreen', alpha=0.7)
axes[1, 2].set_title('Zero Values by Sample')
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{output_dir}/quality_control_plots.png", dpi=300, bbox_inches='tight')
plt.show()
```

## Step 6: Combine Results

### Merge AnnData Objects

```python
def combine_batch_results(results, sample_info):
    """
    Combine all processed samples into a single AnnData object.
    """
    
    adata_list = []
    
    for sample_id, result in results.items():
        adata = result['adata'].copy()
        
        # Add sample metadata
        sample_row = sample_info[sample_info['sample_id'] == sample_id].iloc[0]
        adata.obs['condition'] = sample_row['condition']
        adata.obs['replicate'] = sample_row['replicate']
        
        # Add unique cell IDs
        adata.obs_names = [f"{sample_id}_{i}" for i in range(adata.n_obs)]
        
        adata_list.append(adata)
    
    # Combine all samples
    combined_adata = sc.concat(adata_list, join='outer', index_unique=None)
    
    return combined_adata

# Combine results
combined_adata = combine_batch_results(results, sample_info)

print(f"Combined dataset:")
print(f"  Total cells: {combined_adata.n_obs}")
print(f"  Total features: {combined_adata.n_vars}")
print(f"  Samples: {combined_adata.obs['sample_id'].nunique()}")
print(f"  Conditions: {combined_adata.obs['condition'].unique()}")

# Save combined dataset
combined_adata.write(f"{output_dir}/combined_dataset.h5ad")
```

### Batch Statistics

```python
# Calculate batch statistics
batch_stats = combined_adata.obs.groupby(['sample_id', 'condition']).agg({
    'area': ['count', 'mean', 'std'],
    'shape_factor': ['mean', 'std']
}).round(2)

print("\nBatch Statistics:")
print(batch_stats)

# Save batch statistics
batch_stats.to_csv(f"{output_dir}/batch_statistics.csv")
```

## Step 7: Batch Analysis

### Normalize Combined Data

```python
# Normalize combined data
sc.pp.normalize_total(combined_adata, target_sum=1e4)
sc.pp.log1p(combined_adata)

# Scale data
sc.pp.scale(combined_adata, max_value=10)

# Find highly variable features
sc.pp.highly_variable_features(combined_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
print(f"Highly variable features: {sum(combined_adata.var.highly_variable)}")
```

### Dimensionality Reduction

```python
# PCA
sc.tl.pca(combined_adata, use_highly_variable=True)

# UMAP
sc.pp.neighbors(combined_adata, n_neighbors=10, n_pcs=20)
sc.tl.umap(combined_adata)

# Plot UMAP colored by sample
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sc.pl.umap(combined_adata, color='sample_id', ax=axes[0], show=False)
axes[0].set_title('UMAP by Sample')

sc.pl.umap(combined_adata, color='condition', ax=axes[1], show=False)
axes[1].set_title('UMAP by Condition')

sc.pl.umap(combined_adata, color='area', ax=axes[2], show=False)
axes[2].set_title('UMAP by Cell Area')

plt.tight_layout()
plt.savefig(f"{output_dir}/batch_umap.png", dpi=300, bbox_inches='tight')
plt.show()
```

### Batch Effect Analysis

```python
# Check for batch effects
from scipy import stats

def test_batch_effects(adata, feature='area'):
    """
    Test for significant differences between batches.
    """
    
    conditions = adata.obs['condition'].unique()
    if len(conditions) < 2:
        return None
    
    # Perform t-test
    control_data = adata[adata.obs['condition'] == 'control'].obs[feature]
    treatment_data = adata[adata.obs['condition'] == 'treatment'].obs[feature]
    
    t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
    
    return {
        'feature': feature,
        't_statistic': t_stat,
        'p_value': p_value,
        'control_mean': control_data.mean(),
        'treatment_mean': treatment_data.mean(),
        'control_std': control_data.std(),
        'treatment_std': treatment_data.std()
    }

# Test for batch effects
batch_effects = {}
for feature in ['area', 'shape_factor']:
    result = test_batch_effects(combined_adata, feature)
    if result:
        batch_effects[feature] = result

batch_effects_df = pd.DataFrame(batch_effects).T
print("\nBatch Effect Analysis:")
print(batch_effects_df)

# Save batch effects
batch_effects_df.to_csv(f"{output_dir}/batch_effects.csv")
```

## Step 8: Export Results

### Save All Results

```python
# Create results summary
results_summary = {
    'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_samples_processed': len(results),
    'n_samples_failed': len(failed_samples),
    'total_cells': combined_adata.n_obs,
    'total_features': combined_adata.n_vars,
    'processing_parameters': processing_params,
    'output_files': [
        'combined_dataset.h5ad',
        'processing_summary.csv',
        'quality_control.csv',
        'batch_statistics.csv',
        'batch_effects.csv',
        'quality_control_plots.png',
        'batch_umap.png'
    ]
}

# Save results summary
import json
with open(f"{output_dir}/results_summary.json", 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

print("\nBatch processing completed successfully!")
print(f"Results saved to: {output_dir}")
print(f"Files created:")
for file in results_summary['output_files']:
    print(f"  - {file}")
```

## Summary

This batch processing pipeline provides:

1. **Consistent processing** across multiple samples
2. **Quality control** metrics for each sample
3. **Batch effect analysis** to identify systematic differences
4. **Combined dataset** for downstream analysis
5. **Comprehensive reporting** of processing results

The pipeline is scalable and can be adapted for different experimental designs and sample sizes.

## Next Steps

- 🎯 [Clustering Guide](../tutorials/clustering-guide.md) - Analyze combined dataset
- 🧬 [Spatial Analysis](../tutorials/spatial-analysis.md) - Spatial analysis across samples
- 🎨 [Visualization](visualization.md) - Advanced batch visualization
- 🔧 [API Reference](../api-reference/) - Detailed function documentation
