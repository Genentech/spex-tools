# 🔍 Quality Control Examples

## Overview

Quality control is crucial for reliable spatial omics analysis (proteomics and transcriptomics). This guide provides comprehensive examples for assessing data quality at each step of the pipeline, from image preprocessing to final results validation.

## Prerequisites

```python
import spex as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy import stats
from skimage import measure
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')
```

## 1. Image Quality Assessment

### Signal-to-Noise Ratio Analysis

```python
def calculate_snr(image, channel_idx=0):
    """
    Calculate Signal-to-Noise Ratio for image channel
    
    Parameters:
    -----------
    image : np.ndarray
        Multi-channel image
    channel_idx : int
        Channel index to analyze
        
    Returns:
    --------
    float : SNR value
    """
    channel = image[:, :, channel_idx]
    
    # Estimate noise from background
    background = channel[channel < np.percentile(channel, 10)]
    noise_std = np.std(background)
    
    # Estimate signal from foreground
    foreground = channel[channel > np.percentile(channel, 90)]
    signal_mean = np.mean(foreground)
    
    snr = signal_mean / noise_std if noise_std > 0 else 0
    return snr

# Example usage
Image, channel = sp.load_image('sample_data.tiff')

print("Signal-to-Noise Ratios:")
for i, ch in enumerate(channel):
    snr = calculate_snr(Image, i)
    print(f"Channel {ch}: SNR = {snr:.2f}")
    
    # Quality assessment
    if snr > 10:
        quality = "Excellent"
    elif snr > 5:
        quality = "Good"
    elif snr > 2:
        quality = "Acceptable"
    else:
        quality = "Poor - needs preprocessing"
    
    print(f"  Quality: {quality}")
```

### Channel Correlation Analysis

```python
def analyze_channel_correlation(image, channel_names):
    """
    Analyze correlation between image channels
    
    Parameters:
    -----------
    image : np.ndarray
        Multi-channel image
    channel_names : list
        List of channel names
        
    Returns:
    --------
    pd.DataFrame : Correlation matrix
    """
    n_channels = image.shape[2]
    correlations = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                correlations[i, j] = 1.0
            else:
                # Flatten channels for correlation
                ch1 = image[:, :, i].flatten()
                ch2 = image[:, :, j].flatten()
                correlations[i, j] = np.corrcoef(ch1, ch2)[0, 1]
    
    return pd.DataFrame(correlations, 
                       index=channel_names, 
                       columns=channel_names)

# Example usage
corr_matrix = analyze_channel_correlation(Image, channel)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.2f')
plt.title('Channel Correlation Matrix')
plt.tight_layout()
plt.show()

# Identify highly correlated channels
high_corr_pairs = []
for i in range(len(channel)):
    for j in range(i+1, len(channel)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((channel[i], channel[j], corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print("Highly correlated channel pairs (>0.8):")
    for ch1, ch2, corr in high_corr_pairs:
        print(f"  {ch1} - {ch2}: {corr:.3f}")
        print(f"    Consider removing one channel to reduce redundancy")
```

## 2. Segmentation Quality Control

### Segmentation Metrics

```python
def assess_segmentation_quality(image, labels, channel_idx=0):
    """
    Assess quality of cell segmentation
    
    Parameters:
    -----------
    image : np.ndarray
        Original image
    labels : np.ndarray
        Segmentation labels
    channel_idx : int
        Channel to analyze
        
    Returns:
    --------
    dict : Quality metrics
    """
    channel = image[:, :, channel_idx]
    
    # Basic statistics
    n_cells = labels.max()
    cell_sizes = []
    cell_intensities = []
    
    for i in range(1, n_cells + 1):
        mask = labels == i
        cell_sizes.append(np.sum(mask))
        cell_intensities.append(np.mean(channel[mask]))
    
    cell_sizes = np.array(cell_sizes)
    cell_intensities = np.array(cell_intensities)
    
    # Quality metrics
    metrics = {
        'n_cells': n_cells,
        'mean_cell_size': np.mean(cell_sizes),
        'std_cell_size': np.std(cell_sizes),
        'size_cv': np.std(cell_sizes) / np.mean(cell_sizes),  # Coefficient of variation
        'min_cell_size': np.min(cell_sizes),
        'max_cell_size': np.max(cell_sizes),
        'mean_intensity': np.mean(cell_intensities),
        'intensity_cv': np.std(cell_intensities) / np.mean(cell_intensities)
    }
    
    return metrics, cell_sizes, cell_intensities

# Example usage
labels = sp.cellpose_cellseg(Image, [0], diameter=30)
metrics, sizes, intensities = assess_segmentation_quality(Image, labels)

print("Segmentation Quality Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value:.2f}")

# Visualize cell size distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Cell size distribution
axes[0].hist(sizes, bins=30, alpha=0.7, edgecolor='black')
axes[0].axvline(metrics['mean_cell_size'], color='red', linestyle='--', 
                label=f"Mean: {metrics['mean_cell_size']:.1f}")
axes[0].set_xlabel('Cell Size (pixels)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Cell Size Distribution')
axes[0].legend()

# Cell intensity distribution
axes[1].hist(intensities, bins=30, alpha=0.7, edgecolor='black')
axes[1].axvline(metrics['mean_intensity'], color='red', linestyle='--',
                label=f"Mean: {metrics['mean_intensity']:.1f}")
axes[1].set_xlabel('Mean Cell Intensity')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Cell Intensity Distribution')
axes[1].legend()

plt.tight_layout()
plt.show()

# Quality assessment
print("\nSegmentation Quality Assessment:")
if metrics['size_cv'] < 0.5:
    print("  ✓ Cell size consistency: Good")
else:
    print("  ⚠ Cell size consistency: High variation - consider parameter tuning")

if metrics['intensity_cv'] < 0.3:
    print("  ✓ Cell intensity consistency: Good")
else:
    print("  ⚠ Cell intensity consistency: High variation - check image quality")
```

### Segmentation Validation

```python
def validate_segmentation_boundaries(image, labels, channel_idx=0):
    """
    Validate segmentation boundaries using gradient information
    
    Parameters:
    -----------
    image : np.ndarray
        Original image
    labels : np.ndarray
        Segmentation labels
    channel_idx : int
        Channel to analyze
        
    Returns:
    --------
    float : Boundary quality score
    """
    from scipy import ndimage
    
    channel = image[:, :, channel_idx]
    
    # Calculate image gradients
    grad_x = ndimage.sobel(channel, axis=0)
    grad_y = ndimage.sobel(channel, axis=1)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Find cell boundaries
    boundaries = labels == 0
    boundary_gradients = gradient_magnitude[boundaries]
    
    # Calculate boundary quality score
    if len(boundary_gradients) > 0:
        quality_score = np.mean(boundary_gradients) / np.max(gradient_magnitude)
    else:
        quality_score = 0
    
    return quality_score

# Example usage
boundary_quality = validate_segmentation_boundaries(Image, labels)
print(f"Boundary Quality Score: {boundary_quality:.3f}")

if boundary_quality > 0.3:
    print("  ✓ Segmentation boundaries align well with image gradients")
else:
    print("  ⚠ Segmentation boundaries may not follow cell boundaries well")
```

## 3. Feature Extraction Quality Control

### Feature Quality Assessment

```python
def assess_feature_quality(features_df):
    """
    Assess quality of extracted features
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with extracted features
        
    Returns:
    --------
    dict : Feature quality metrics
    """
    # Remove non-numeric columns
    numeric_features = features_df.select_dtypes(include=[np.number])
    
    # Calculate feature statistics
    feature_stats = {
        'n_features': len(numeric_features.columns),
        'n_cells': len(numeric_features),
        'missing_values': numeric_features.isnull().sum().sum(),
        'zero_variance_features': (numeric_features.var() == 0).sum(),
        'high_correlation_pairs': 0
    }
    
    # Check for highly correlated features
    corr_matrix = numeric_features.corr()
    high_corr_count = 0
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.95:
                high_corr_count += 1
    
    feature_stats['high_correlation_pairs'] = high_corr_count
    
    return feature_stats, numeric_features

# Example usage
# Extract features
features = sp.feature_extraction(Image, labels, list(range(len(channel))))

# Assess feature quality
stats, numeric_features = assess_feature_quality(features)

print("Feature Quality Metrics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# Feature distribution analysis
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Feature variance
variances = numeric_features.var().sort_values(ascending=False)
axes[0, 0].plot(range(len(variances)), variances)
axes[0, 0].set_xlabel('Feature Index')
axes[0, 0].set_ylabel('Variance')
axes[0, 0].set_title('Feature Variance Distribution')
axes[0, 0].set_yscale('log')

# Missing values
missing_counts = numeric_features.isnull().sum()
axes[0, 1].bar(range(len(missing_counts)), missing_counts)
axes[0, 1].set_xlabel('Feature Index')
axes[0, 1].set_ylabel('Missing Values')
axes[0, 1].set_title('Missing Values per Feature')

# Feature correlation heatmap (top features)
top_features = variances.head(10).index
corr_subset = numeric_features[top_features].corr()
sns.heatmap(corr_subset, ax=axes[1, 0], cmap='RdBu_r', center=0,
            square=True, annot=False)
axes[1, 0].set_title('Top 10 Features Correlation')

# Feature distribution example
if len(numeric_features.columns) > 0:
    example_feature = numeric_features.columns[0]
    axes[1, 1].hist(numeric_features[example_feature].dropna(), bins=30, alpha=0.7)
    axes[1, 1].set_xlabel(example_feature)
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Distribution: {example_feature}')

plt.tight_layout()
plt.show()
```

## 4. Clustering Quality Control

### Clustering Validation Metrics

```python
def validate_clustering(features_df, cluster_labels):
    """
    Validate clustering results using multiple metrics
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        Feature matrix
    cluster_labels : np.ndarray
        Cluster assignments
        
    Returns:
    --------
    dict : Validation metrics
    """
    # Remove non-numeric columns and handle missing values
    numeric_features = features_df.select_dtypes(include=[np.number])
    numeric_features = numeric_features.fillna(numeric_features.mean())
    
    # Calculate validation metrics
    metrics = {}
    
    # Silhouette score
    try:
        metrics['silhouette'] = silhouette_score(numeric_features, cluster_labels)
    except:
        metrics['silhouette'] = np.nan
    
    # Calinski-Harabasz score
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(numeric_features, cluster_labels)
    except:
        metrics['calinski_harabasz'] = np.nan
    
    # Davies-Bouldin score
    try:
        from sklearn.metrics import davies_bouldin_score
        metrics['davies_bouldin'] = davies_bouldin_score(numeric_features, cluster_labels)
    except:
        metrics['davies_bouldin'] = np.nan
    
    # Cluster size distribution
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    metrics['n_clusters'] = len(unique_clusters)
    metrics['min_cluster_size'] = np.min(cluster_counts)
    metrics['max_cluster_size'] = np.max(cluster_counts)
    metrics['cluster_size_cv'] = np.std(cluster_counts) / np.mean(cluster_counts)
    
    return metrics

# Example usage
# Perform clustering
adata = sp.feature_extraction_adata(Image, labels, list(range(len(channel))))
clusters = sp.cluster(adata, method='leiden', resolution=0.5)

# Validate clustering
validation_metrics = validate_clustering(features, clusters)

print("Clustering Validation Metrics:")
for key, value in validation_metrics.items():
    print(f"  {key}: {value:.3f}")

# Visualize clustering quality
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Cluster size distribution
unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
axes[0].bar(range(len(cluster_counts)), cluster_counts)
axes[0].set_xlabel('Cluster ID')
axes[0].set_ylabel('Number of Cells')
axes[0].set_title('Cluster Size Distribution')

# Silhouette analysis
if not np.isnan(validation_metrics['silhouette']):
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(features.select_dtypes(include=[np.number]), clusters)
    
    axes[1].hist(silhouette_vals, bins=30, alpha=0.7)
    axes[1].axvline(validation_metrics['silhouette'], color='red', linestyle='--',
                    label=f"Mean: {validation_metrics['silhouette']:.3f}")
    axes[1].set_xlabel('Silhouette Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Silhouette Score Distribution')
    axes[1].legend()

plt.tight_layout()
plt.show()

# Quality assessment
print("\nClustering Quality Assessment:")
if validation_metrics['silhouette'] > 0.3:
    print("  ✓ Clustering quality: Good (silhouette > 0.3)")
elif validation_metrics['silhouette'] > 0.1:
    print("  ⚠ Clustering quality: Acceptable (silhouette > 0.1)")
else:
    print("  ❌ Clustering quality: Poor - consider different parameters")

if validation_metrics['cluster_size_cv'] < 1.0:
    print("  ✓ Cluster size balance: Good")
else:
    print("  ⚠ Cluster size balance: Uneven - consider adjusting resolution")
```

## 5. Spatial Analysis Quality Control

### Spatial Distribution Analysis

```python
def assess_spatial_distribution(labels, cluster_labels):
    """
    Assess spatial distribution of clusters
    
    Parameters:
    -----------
    labels : np.ndarray
        Cell segmentation labels
    cluster_labels : np.ndarray
        Cluster assignments
        
    Returns:
    --------
    dict : Spatial distribution metrics
    """
    # Calculate cell centroids
    cell_props = measure.regionprops_table(labels, properties=['centroid', 'area'])
    centroids = np.column_stack([cell_props['centroid-0'], cell_props['centroid-1']])
    
    # Calculate spatial metrics for each cluster
    spatial_metrics = {}
    
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_centroids = centroids[cluster_mask]
        
        if len(cluster_centroids) > 1:
            # Calculate spatial spread
            centroid_mean = np.mean(cluster_centroids, axis=0)
            distances = np.linalg.norm(cluster_centroids - centroid_mean, axis=1)
            
            spatial_metrics[cluster_id] = {
                'n_cells': len(cluster_centroids),
                'spatial_spread': np.std(distances),
                'density': len(cluster_centroids) / (labels.shape[0] * labels.shape[1])
            }
    
    return spatial_metrics

# Example usage
spatial_metrics = assess_spatial_distribution(labels, clusters)

print("Spatial Distribution Metrics:")
for cluster_id, metrics in spatial_metrics.items():
    print(f"\nCluster {cluster_id}:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")

# Visualize spatial distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Spatial distribution of clusters
unique_clusters = np.unique(clusters)
colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))

for i, cluster_id in enumerate(unique_clusters):
    cluster_mask = clusters == cluster_id
    cluster_cells = labels.copy()
    cluster_cells[~cluster_mask] = 0
    
    # Find cluster boundaries
    cluster_boundaries = cluster_cells > 0
    
    axes[0].contour(cluster_boundaries, levels=[0.5], colors=[colors[i]], 
                    linewidths=2, label=f'Cluster {cluster_id}')

axes[0].set_title('Spatial Distribution of Clusters')
axes[0].legend()
axes[0].axis('off')

# Cluster density comparison
cluster_ids = list(spatial_metrics.keys())
densities = [spatial_metrics[cid]['density'] for cid in cluster_ids]
spreads = [spatial_metrics[cid]['spatial_spread'] for cid in cluster_ids]

scatter = axes[1].scatter(densities, spreads, c=cluster_ids, cmap='Set3', s=100)
axes[1].set_xlabel('Cluster Density')
axes[1].set_ylabel('Spatial Spread')
axes[1].set_title('Cluster Spatial Characteristics')
plt.colorbar(scatter, ax=axes[1], label='Cluster ID')

plt.tight_layout()
plt.show()
```

## 6. Comprehensive Quality Report

```python
def generate_quality_report(image, labels, features, clusters, channel_names):
    """
    Generate comprehensive quality control report
    
    Parameters:
    -----------
    image : np.ndarray
        Original image
    labels : np.ndarray
        Segmentation labels
    features : pd.DataFrame
        Extracted features
    clusters : np.ndarray
        Cluster assignments
    channel_names : list
        Channel names
        
    Returns:
    --------
    dict : Comprehensive quality report
    """
    report = {}
    
    # Image quality
    report['image_quality'] = {}
    for i, ch in enumerate(channel_names):
        snr = calculate_snr(image, i)
        report['image_quality'][ch] = {
            'snr': snr,
            'quality': 'Excellent' if snr > 10 else 'Good' if snr > 5 else 'Acceptable' if snr > 2 else 'Poor'
        }
    
    # Segmentation quality
    seg_metrics, _, _ = assess_segmentation_quality(image, labels)
    report['segmentation_quality'] = seg_metrics
    
    # Feature quality
    feature_stats, _ = assess_feature_quality(features)
    report['feature_quality'] = feature_stats
    
    # Clustering quality
    cluster_metrics = validate_clustering(features, clusters)
    report['clustering_quality'] = cluster_metrics
    
    # Spatial quality
    spatial_metrics = assess_spatial_distribution(labels, clusters)
    report['spatial_quality'] = spatial_metrics
    
    return report

# Example usage
quality_report = generate_quality_report(Image, labels, features, clusters, channel)

print("=" * 60)
print("COMPREHENSIVE QUALITY CONTROL REPORT")
print("=" * 60)

# Image Quality
print("\n📷 IMAGE QUALITY:")
for ch, metrics in quality_report['image_quality'].items():
    print(f"  Channel {ch}: SNR = {metrics['snr']:.2f} ({metrics['quality']})")

# Segmentation Quality
print("\n🔍 SEGMENTATION QUALITY:")
seg_metrics = quality_report['segmentation_quality']
print(f"  Cells detected: {seg_metrics['n_cells']}")
print(f"  Size consistency: {'Good' if seg_metrics['size_cv'] < 0.5 else 'Needs attention'}")

# Feature Quality
print("\n📊 FEATURE QUALITY:")
feat_metrics = quality_report['feature_quality']
print(f"  Features extracted: {feat_metrics['n_features']}")
print(f"  Missing values: {feat_metrics['missing_values']}")
print(f"  Zero-variance features: {feat_metrics['zero_variance_features']}")

# Clustering Quality
print("\n🎯 CLUSTERING QUALITY:")
clust_metrics = quality_report['clustering_quality']
print(f"  Clusters: {clust_metrics['n_clusters']}")
print(f"  Silhouette score: {clust_metrics['silhouette']:.3f}")
print(f"  Size balance: {'Good' if clust_metrics['cluster_size_cv'] < 1.0 else 'Uneven'}")

# Overall Assessment
print("\n📋 OVERALL ASSESSMENT:")
overall_score = 0
total_checks = 0

# Check image quality
for ch, metrics in quality_report['image_quality'].items():
    if metrics['quality'] in ['Excellent', 'Good']:
        overall_score += 1
    total_checks += 1

# Check segmentation
if seg_metrics['size_cv'] < 0.5:
    overall_score += 1
total_checks += 1

# Check clustering
if clust_metrics['silhouette'] > 0.1:
    overall_score += 1
total_checks += 1

overall_percentage = (overall_score / total_checks) * 100
print(f"  Quality Score: {overall_percentage:.1f}%")

if overall_percentage >= 80:
    print("  Status: ✅ EXCELLENT - Ready for analysis")
elif overall_percentage >= 60:
    print("  Status: ⚠️ ACCEPTABLE - Minor issues detected")
else:
    print("  Status: ❌ NEEDS ATTENTION - Significant quality issues")

print("=" * 60)
```

## Summary

This quality control guide provides comprehensive tools for assessing data quality at each step of the spatial omics pipeline (proteomics and transcriptomics):

1. **Image Quality**: SNR analysis and channel correlation
2. **Segmentation Quality**: Cell size/intensity distribution and boundary validation
3. **Feature Quality**: Variance analysis and correlation assessment
4. **Clustering Quality**: Multiple validation metrics and cluster balance
5. **Spatial Quality**: Distribution analysis and spatial characteristics
6. **Comprehensive Reporting**: Overall quality assessment with actionable recommendations

Use these tools to ensure reliable and reproducible results in your spatial omics analysis.
