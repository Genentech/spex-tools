# Documentation Changelog

## [2024-12-19] - Documentation Improvements

### Added
- **Detailed documentation for `load_image`**: Added comprehensive descriptions of parameters, return values, usage examples, and notes
- **New "Image Segmentation" page**: Created a dedicated page with complete documentation of all segmentation functions
- **Updated main page**: Transferred information from README.md with improved structure and navigation
- **Usage examples**: Added practical examples for all main functions

### Changed
- **Enhanced `load_image` function documentation**:
  - Added detailed parameter and return value descriptions
  - Included usage examples for OME-TIFF and OME-ZARR formats
  - Added notes about supported formats and implementation details
- **Updated navigation**: Added link to segmentation page in main menu

### Documentation Structure
```
docs/
├── index.md              # Main page with library overview
├── segmentation.md       # Detailed segmentation documentation
├── api.md               # Complete API documentation
└── CHANGELOG.md         # Change history
```

### Supported Image Formats
- **OME-TIFF** (.ome.tiff, .ome.tif) - Primary format for multi-channel images
- **OME-ZARR** (.zarr) - Modern format for large datasets
- **TIFF** - Standard image format (with limited metadata support)

### Segmentation Functions
- `load_image` - Image loading with automatic dimension handling
- `cellpose_cellseg` - Cell segmentation using deep learning
- `stardist_cellseg` - Star-convex object-based segmentation
- `watershed_classic` - Classical watershed segmentation
- `median_denoise`, `nlm_denoise` - Noise filtering
- `background_subtract` - Background subtraction
- Postprocessing functions: `rescue_cells`, `remove_small_objects`, `remove_large_objects`
- `feature_extraction_adata` - Feature extraction to AnnData format
