# Add Affine to Crop

**Date:** 2026-02-04
**Session Title:** Add Affine to Crop

## Overview
Implemented the `use_affine` parameter across `crop*` functions in `easyidp.geotiff` and `easyidp.roi`, enabling users to obtain affine-rotated GeoTIFF crops for rectangular ROIs. This ensures that crops of rotated rectangles are minimally sized and aligned with the rectangle's orientation, rather than the axis-aligned bounding box.

## Changes

### `src/easyidp/geotiff.py`
- **Updated Functions:** `crop_shapely_polygon`, `crop_polygon`, `crop_rectangle`, `crop_rois`.
- **New Parameter:** `use_affine` (bool).
- **Behavior:**
    - If `use_affine=True`, `return_geotiff` is forced to `True`.
    - The resulting cropped GeoTIFF is automatically converted to affine storage (rotated) via `convert_to_affine()`.
    - A fallback mechanism serves a warning if `convert_to_affine` fails (e.g., if the polygon is not a rectangle).
- **Improvements:**
    - Enhanced `_prepare_affine_storage` to handle floating-point precision issues when calculating output dimensions, preventing off-by-one errors for perfectly aligned crops.

### `src/easyidp/roi.py`
- **Updated Function:** `crop`.
- **Changes:** Added `**kwargs` support.
- **Purpose:** Allows passing `use_affine`, `return_geotiff`, and other parameters directly to the underlying `GeoTiff.crop_rois` method.

## Verification

### Automated Tests
Added `TestAffineCrop` class to `tests/test_geotiff.py` with the following test cases:

1.  **`test_crop_polygon_use_affine`**: Validates rotated crops from manually defined rotated rectangle polygons. Checks for correct affine transformation and output dimensions.
2.  **`test_crop_rectangle_use_affine_returns_affine`**: Ensures that even axis-aligned `crop_rectangle` calls return an affine-enabled GeoTIFF when `use_affine=True` is requested.
3.  **`test_roi_crop_lotus`**: Verifies `roi.crop(dom, use_affine=True)` using the local `idp.TestData` (avoiding large dataset downloads).
    - Checks that output is a dictionary of `GeoTiff` objects.
    - Confirms proper affine rotation by comparing crop area with standard axis-aligned crop (rotated crop should be smaller).
    - Validates pixel value consistency between original and cropped images (within interpolation tolerance).

### Regression Testing
Ran full `tests/test_geotiff.py`, confirming all tests (including new affine ones) pass successfully.

## Usage Examples

### Python API

```python
import easyidp as idp

# 1. Load Data
dom = idp.GeoTiff("path/to/dom.tif")
roi = idp.ROI("path/to/roi.shp")

# 2. Standard Crop (Legacy behavior)
# Returns dictionary of numpy arrays (axis-aligned bounding box)
crops_np = roi.crop(dom)
# crops_np['roi_1'] is np.ndarray

# 3. Affine Crop (New feature)
# Returns dictionary of GeoTiff objects (rotated/affine aligned)
crops_affine = roi.crop(dom, use_affine=True)
# crops_affine['roi_1'] is easyidp.GeoTiff object

# 4. Save affine crops
for name, gtiff in crops_affine.items():
    # The saved TIFF will have rotation metadata
    gtiff.save(f"outputs/{name}.tif")
```
