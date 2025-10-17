# ⚡ Performance Optimization Examples

## Overview

This guide provides comprehensive examples for optimizing performance in spatial omics analysis (proteomics and transcriptomics) using SPEX. Learn how to handle large datasets efficiently, parallelize computations, and optimize memory usage.

## Prerequisites

```python
import spex as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import time
import psutil
import multiprocessing as mp
from functools import partial
import dask.array as da
import dask.dataframe as dd
from memory_profiler import profile

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')
```

## 1. Memory Management

### Memory Usage Monitoring

```python
def get_memory_usage():
    """
    Get current memory usage in MB
    
    Returns:
    --------
    float : Memory usage in MB
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

def monitor_memory(func):
    """
    Decorator to monitor memory usage of functions
    
    Parameters:
    -----------
    func : function
        Function to monitor
        
    Returns:
    --------
    function : Wrapped function with memory monitoring
    """
    def wrapper(*args, **kwargs):
        memory_before = get_memory_usage()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        memory_after = get_memory_usage()
        
        print(f"Function: {func.__name__}")
        print(f"  Execution time: {end_time - start_time:.2f} seconds")
        print(f"  Memory usage: {memory_after - memory_before:.2f} MB")
        
        return result
    return wrapper

# Example usage
@monitor_memory
def load_large_image(file_path):
    """Load large image with memory monitoring"""
    return sp.load_image(file_path)

# Test memory monitoring
Image, channel = load_large_image('large_sample_data.tiff')
print(f"Image shape: {Image.shape}")
print(f"Image memory usage: {Image.nbytes / 1024 / 1024:.2f} MB")
```

### Efficient Image Loading

```python
def load_image_chunked(file_path, chunk_size=1000):
    """
    Load large image in chunks to reduce memory usage
    
    Parameters:
    -----------
    file_path : str
        Path to image file
    chunk_size : int
        Size of chunks to load
        
    Returns:
    --------
    tuple : (image, channel_names)
    """
    import tifffile
    
    with tifffile.TiffFile(file_path) as tif:
        # Get image info
        shape = tif.series[0].shape
        dtype = tif.series[0].dtype
        
        # Calculate chunk dimensions
        if len(shape) == 3:
            height, width, channels = shape
        else:
            height, width = shape
            channels = 1
        
        # Load image in chunks
        image = np.zeros((height, width, channels), dtype=dtype)
        
        for y in range(0, height, chunk_size):
            y_end = min(y + chunk_size, height)
            for x in range(0, width, chunk_size):
                x_end = min(x + chunk_size, width)
                
                # Load chunk
                chunk = tif.series[0].asarray(key=slice(y, y_end))
                if len(chunk.shape) == 2:
                    chunk = chunk[:, :, np.newaxis]
                
                image[y:y_end, x:x_end, :] = chunk
        
        # Get channel names
        channel_names = [f"Channel_{i}" for i in range(channels)]
        
        return image, channel_names

# Example usage
print("Loading large image with chunked approach...")
Image_chunked, channel_names = load_image_chunked('large_sample_data.tiff', chunk_size=500)
print(f"Loaded image shape: {Image_chunked.shape}")
```

### Memory-Efficient Segmentation

```python
def segment_large_image_chunked(image, chunk_size=512, overlap=50):
    """
    Segment large image in chunks with overlap
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    chunk_size : int
        Size of processing chunks
    overlap : int
        Overlap between chunks
        
    Returns:
    --------
    np.ndarray : Segmentation labels
    """
    height, width = image.shape[:2]
    labels = np.zeros((height, width), dtype=np.int32)
    
    # Process image in chunks
    for y in range(0, height, chunk_size - overlap):
        y_end = min(y + chunk_size, height)
        for x in range(0, width, chunk_size - overlap):
            x_end = min(x + chunk_size, width)
            
            # Extract chunk with overlap
            y_start = max(0, y - overlap)
            x_start = max(0, x - overlap)
            y_end_ext = min(height, y_end + overlap)
            x_end_ext = min(width, x_end + overlap)
            
            chunk = image[y_start:y_end_ext, x_start:x_end_ext]
            
            # Segment chunk
            chunk_labels = sp.cellpose_cellseg(chunk, [0], diameter=30)
            
            # Remove overlap and assign labels
            if y > 0:
                chunk_labels = chunk_labels[overlap:, :]
            if x > 0:
                chunk_labels = chunk_labels[:, overlap:]
            if y_end < height:
                chunk_labels = chunk_labels[:-(y_end_ext - y_end), :]
            if x_end < width:
                chunk_labels = chunk_labels[:, :-(x_end_ext - x_end)]
            
            # Adjust labels to avoid conflicts
            max_label = labels.max()
            chunk_labels[chunk_labels > 0] += max_label
            
            # Assign to main labels array
            labels[y:y_end, x:x_end] = chunk_labels
    
    return labels

# Example usage
print("Segmenting large image with chunked approach...")
labels_chunked = segment_large_image_chunked(Image_chunked, chunk_size=512, overlap=50)
print(f"Segmentation complete: {labels_chunked.max()} cells detected")
```

## 2. Parallel Processing

### Parallel Feature Extraction

```python
def extract_features_parallel(image, labels, channel_indices, n_jobs=-1):
    """
    Extract features in parallel for multiple channels
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    labels : np.ndarray
        Segmentation labels
    channel_indices : list
        List of channel indices to process
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
        
    Returns:
    --------
    pd.DataFrame : Combined features
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    def extract_single_channel(ch_idx):
        """Extract features for single channel"""
        return sp.feature_extraction(image, labels, [ch_idx])
    
    # Process channels in parallel
    with mp.Pool(n_jobs) as pool:
        results = pool.map(extract_single_channel, channel_indices)
    
    # Combine results
    combined_features = pd.concat(results, axis=1)
    return combined_features

# Example usage
print("Extracting features in parallel...")
features_parallel = extract_features_parallel(Image_chunked, labels_chunked, 
                                            list(range(len(channel_names))), n_jobs=4)
print(f"Extracted {features_parallel.shape[1]} features for {features_parallel.shape[0]} cells")
```

### Parallel Clustering

```python
def cluster_multiple_resolutions_parallel(features, resolutions, method='leiden', n_jobs=-1):
    """
    Perform clustering with multiple resolutions in parallel
    
    Parameters:
    -----------
    features : pd.DataFrame
        Feature matrix
    resolutions : list
        List of resolution parameters
    method : str
        Clustering method
    n_jobs : int
        Number of parallel jobs
        
    Returns:
    --------
    dict : Clustering results for each resolution
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    def cluster_single_resolution(res):
        """Perform clustering with single resolution"""
        adata = sp.feature_extraction_adata(Image_chunked, labels_chunked, 
                                          list(range(len(channel_names))))
        clusters = sp.cluster(adata, method=method, resolution=res)
        return res, clusters
    
    # Process resolutions in parallel
    with mp.Pool(n_jobs) as pool:
        results = pool.map(cluster_single_resolution, resolutions)
    
    # Organize results
    clustering_results = {res: clusters for res, clusters in results}
    return clustering_results

# Example usage
resolutions = [0.1, 0.3, 0.5, 0.7, 1.0]
print("Performing clustering with multiple resolutions in parallel...")
clustering_results = cluster_multiple_resolutions_parallel(features_parallel, resolutions, n_jobs=4)

for res, clusters in clustering_results.items():
    n_clusters = len(np.unique(clusters))
    print(f"Resolution {res}: {n_clusters} clusters")
```

## 3. Dask Integration for Large Datasets

### Dask-based Feature Extraction

```python
def extract_features_dask(image, labels, channel_indices, chunk_size=1000):
    """
    Extract features using Dask for large datasets
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    labels : np.ndarray
        Segmentation labels
    channel_indices : list
        List of channel indices
    chunk_size : int
        Size of Dask chunks
        
    Returns:
    --------
    dd.DataFrame : Dask DataFrame with features
    """
    # Convert to Dask arrays
    image_dask = da.from_array(image, chunks=(chunk_size, chunk_size, -1))
    labels_dask = da.from_array(labels, chunks=(chunk_size, chunk_size))
    
    # Extract features for each channel
    feature_dfs = []
    
    for ch_idx in channel_indices:
        # Extract channel
        channel = image_dask[:, :, ch_idx]
        
        # Calculate features using Dask
        features_ch = da.map_blocks(
            lambda x, y: sp.feature_extraction(x, y, [0]),
            channel, labels_dask,
            dtype=object
        )
        
        # Convert to Dask DataFrame
        features_df = dd.from_dask_array(features_ch, columns=[f'ch_{ch_idx}_features'])
        feature_dfs.append(features_df)
    
    # Combine all features
    combined_features = dd.concat(feature_dfs, axis=1)
    return combined_features

# Example usage
print("Extracting features using Dask...")
features_dask = extract_features_dask(Image_chunked, labels_chunked, 
                                    list(range(len(channel_names))), chunk_size=500)

# Compute results
features_computed = features_dask.compute()
print(f"Dask features shape: {features_computed.shape}")
```

### Dask-based Spatial Analysis

```python
def spatial_analysis_dask(labels, cluster_labels, chunk_size=1000):
    """
    Perform spatial analysis using Dask
    
    Parameters:
    -----------
    labels : np.ndarray
        Segmentation labels
    cluster_labels : np.ndarray
        Cluster assignments
    chunk_size : int
        Size of Dask chunks
        
    Returns:
    --------
    dict : Spatial analysis results
    """
    # Convert to Dask arrays
    labels_dask = da.from_array(labels, chunks=(chunk_size, chunk_size))
    cluster_labels_dask = da.from_array(cluster_labels, chunks=chunk_size)
    
    # Calculate spatial metrics using Dask
    def calculate_spatial_metrics(labels_chunk, clusters_chunk):
        """Calculate spatial metrics for chunk"""
        # This is a simplified version - in practice, you'd implement
        # the full spatial analysis logic here
        return np.mean(labels_chunk), np.std(clusters_chunk)
    
    # Apply spatial analysis
    spatial_results = da.map_blocks(
        calculate_spatial_metrics,
        labels_dask, cluster_labels_dask,
        dtype=float
    )
    
    # Compute results
    results = spatial_results.compute()
    
    return {
        'mean_labels': results[0],
        'std_clusters': results[1]
    }

# Example usage
print("Performing spatial analysis with Dask...")
spatial_results_dask = spatial_analysis_dask(labels_chunked, clustering_results[0.5])
print("Spatial analysis complete")
```

## 4. Performance Profiling

### Function Performance Profiling

```python
@profile
def profile_feature_extraction(image, labels, channel_indices):
    """
    Profile feature extraction performance
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    labels : np.ndarray
        Segmentation labels
    channel_indices : list
        Channel indices
    """
    features = sp.feature_extraction(image, labels, channel_indices)
    return features

# Example usage
print("Profiling feature extraction...")
features_profiled = profile_feature_extraction(Image_chunked, labels_chunked, 
                                             list(range(len(channel_names))))
```

### Performance Comparison

```python
def compare_performance_methods(image, labels, channel_indices):
    """
    Compare performance of different methods
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    labels : np.ndarray
        Segmentation labels
    channel_indices : list
        Channel indices
    """
    methods = {
        'Sequential': lambda: sp.feature_extraction(image, labels, channel_indices),
        'Parallel': lambda: extract_features_parallel(image, labels, channel_indices, n_jobs=4),
        'Dask': lambda: extract_features_dask(image, labels, channel_indices).compute()
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"Testing {method_name} method...")
        
        # Measure time
        start_time = time.time()
        start_memory = get_memory_usage()
        
        result = method_func()
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        results[method_name] = {
            'time': end_time - start_time,
            'memory': end_memory - start_memory,
            'result_shape': result.shape
        }
    
    return results

# Example usage
print("Comparing performance methods...")
performance_comparison = compare_performance_methods(Image_chunked, labels_chunked, 
                                                   list(range(len(channel_names))))

# Display results
print("\nPerformance Comparison:")
for method, metrics in performance_comparison.items():
    print(f"\n{method}:")
    print(f"  Time: {metrics['time']:.2f} seconds")
    print(f"  Memory: {metrics['memory']:.2f} MB")
    print(f"  Result shape: {metrics['result_shape']}")

# Create performance visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Time comparison
methods = list(performance_comparison.keys())
times = [performance_comparison[m]['time'] for m in methods]
memory = [performance_comparison[m]['memory'] for m in methods]

axes[0].bar(methods, times, color=['blue', 'green', 'red'])
axes[0].set_ylabel('Time (seconds)')
axes[0].set_title('Execution Time Comparison')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(methods, memory, color=['blue', 'green', 'red'])
axes[1].set_ylabel('Memory Usage (MB)')
axes[1].set_title('Memory Usage Comparison')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

## 5. Optimization Strategies

### Batch Processing for Large Datasets

```python
def process_large_dataset_batch(file_list, batch_size=5):
    """
    Process large dataset in batches
    
    Parameters:
    -----------
    file_list : list
        List of file paths
    batch_size : int
        Number of files to process in each batch
        
    Returns:
    --------
    list : Results for all files
    """
    all_results = []
    
    for i in range(0, len(file_list), batch_size):
        batch_files = file_list[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(file_list) + batch_size - 1)//batch_size}")
        
        batch_results = []
        for file_path in batch_files:
            try:
                # Load and process file
                image, channels = sp.load_image(file_path)
                labels = sp.cellpose_cellseg(image, [0], diameter=30)
                features = sp.feature_extraction(image, labels, list(range(len(channels))))
                
                batch_results.append({
                    'file': file_path,
                    'features': features,
                    'n_cells': labels.max()
                })
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                batch_results.append({
                    'file': file_path,
                    'error': str(e)
                })
        
        all_results.extend(batch_results)
        
        # Clear memory after each batch
        import gc
        gc.collect()
    
    return all_results

# Example usage
file_list = ['sample_1.tiff', 'sample_2.tiff', 'sample_3.tiff', 'sample_4.tiff', 'sample_5.tiff']
print("Processing large dataset in batches...")
batch_results = process_large_dataset_batch(file_list, batch_size=2)

for result in batch_results:
    if 'error' not in result:
        print(f"{result['file']}: {result['n_cells']} cells")
    else:
        print(f"{result['file']}: ERROR - {result['error']}")
```

### Memory-Efficient Data Structures

```python
def optimize_data_types(features_df):
    """
    Optimize data types to reduce memory usage
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        Input features DataFrame
        
    Returns:
    --------
    pd.DataFrame : Optimized DataFrame
    """
    # Get memory usage before optimization
    memory_before = features_df.memory_usage(deep=True).sum() / 1024 / 1024
    
    # Optimize numeric columns
    for col in features_df.select_dtypes(include=[np.number]).columns:
        col_min = features_df[col].min()
        col_max = features_df[col].max()
        
        # Optimize integer columns
        if features_df[col].dtype == 'int64':
            if col_min >= 0:
                if col_max < 255:
                    features_df[col] = features_df[col].astype(np.uint8)
                elif col_max < 65535:
                    features_df[col] = features_df[col].astype(np.uint16)
                else:
                    features_df[col] = features_df[col].astype(np.uint32)
            else:
                if col_min > -128 and col_max < 127:
                    features_df[col] = features_df[col].astype(np.int8)
                elif col_min > -32768 and col_max < 32767:
                    features_df[col] = features_df[col].astype(np.int16)
                else:
                    features_df[col] = features_df[col].astype(np.int32)
        
        # Optimize float columns
        elif features_df[col].dtype == 'float64':
            features_df[col] = features_df[col].astype(np.float32)
    
    # Get memory usage after optimization
    memory_after = features_df.memory_usage(deep=True).sum() / 1024 / 1024
    
    print(f"Memory optimization:")
    print(f"  Before: {memory_before:.2f} MB")
    print(f"  After: {memory_after:.2f} MB")
    print(f"  Reduction: {((memory_before - memory_after) / memory_before * 100):.1f}%")
    
    return features_df

# Example usage
print("Optimizing data types...")
features_optimized = optimize_data_types(features_parallel)
```

## 6. Performance Monitoring Dashboard

```python
def create_performance_dashboard(performance_data):
    """
    Create performance monitoring dashboard
    
    Parameters:
    -----------
    performance_data : dict
        Performance metrics data
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Method comparison
    methods = list(performance_data.keys())
    times = [performance_data[m]['time'] for m in methods]
    memory = [performance_data[m]['memory'] for m in methods]
    
    # Time comparison
    axes[0, 0].bar(methods, times, color=['blue', 'green', 'red', 'orange'])
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].set_title('Execution Time by Method')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Memory comparison
    axes[0, 1].bar(methods, memory, color=['blue', 'green', 'red', 'orange'])
    axes[0, 1].set_ylabel('Memory Usage (MB)')
    axes[0, 1].set_title('Memory Usage by Method')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Efficiency plot (time vs memory)
    axes[1, 0].scatter(times, memory, c=range(len(methods)), cmap='viridis', s=100)
    for i, method in enumerate(methods):
        axes[1, 0].annotate(method, (times[i], memory[i]), 
                           xytext=(5, 5), textcoords='offset points')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Memory Usage (MB)')
    axes[1, 0].set_title('Efficiency: Time vs Memory')
    
    # Performance summary table
    axes[1, 1].axis('off')
    table_data = []
    for method in methods:
        table_data.append([
            method,
            f"{performance_data[method]['time']:.2f}s",
            f"{performance_data[method]['memory']:.2f}MB",
            f"{performance_data[method]['result_shape']}"
        ])
    
    table = axes[1, 1].table(cellText=table_data,
                            colLabels=['Method', 'Time', 'Memory', 'Shape'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.tight_layout()
    plt.show()

# Example usage
print("Creating performance dashboard...")
create_performance_dashboard(performance_comparison)
```

## Summary

This performance optimization guide provides comprehensive tools for handling large spatial omics datasets (proteomics and transcriptomics) efficiently:

1. **Memory Management**: Monitoring and optimizing memory usage
2. **Parallel Processing**: Utilizing multiple CPU cores for faster computation
3. **Dask Integration**: Handling datasets that don't fit in memory
4. **Performance Profiling**: Identifying bottlenecks and optimization opportunities
5. **Batch Processing**: Processing large datasets in manageable chunks
6. **Data Type Optimization**: Reducing memory footprint through efficient data types

Use these techniques to scale your spatial omics analysis to handle large datasets while maintaining reasonable performance and memory usage.
