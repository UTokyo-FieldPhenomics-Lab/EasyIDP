# Summary of Changes: Implementing `get_z_from_pcd` (By Gemini 3 Pro High)

**Overview**
This implementation adds the `get_z_from_pcd` method to `easyidp.roi.ROI`, allowing users to extract Z (elevation) values from point clouds (LAS/LAZ/PLY) within ROI polygons, similar to `get_z_from_dsm`.

**Key Changes**

1.  **Dependencies**
    *   Added `scipy` (for `cKDTree`) and `trimesh` (for potential future mesh ops) to `pyproject.toml`.

2.  **`src/easyidp/pointcloud.py`**
    *   **CRS Support**: Added `crs` property (supports `pyproj.CRS`) and `change_crs` method.
    *   **Optimization**: Added `tree` property that lazily creates a 2D `scipy.spatial.cKDTree` for efficient spatial querying.
    *   **Input Handling**: Updated `read_point_cloud` to attempt loading CRS from metadata or sidecar `.crs` files.

3.  **`src/easyidp/roi.py`**
    *   **Implemented `get_z_from_pcd`**:
        *   Efficient implementation using a combination of `cKDTree.query_ball_point` (broad phase, finding points in ROI bounding box) and `matplotlib.path.Path.contains_points` (narrow phase, precise point-in-polygon).
        *   Supports `mode="face"` (aggregated Z value per polygon) and `mode="point"` (per vertex).
        *   Checks for CRS consistency between ROI and PointCloud, issuing warnings on mismatch.
    *   **Refactored `calculate_kernel_stats`**:
        *   Extracted kernel calculation logic into a reusable `calculate_kernel_stats` function.
        *   Updated to support multi-dimensional arrays (handling multi-band GeoTIFF data logic) to align with `geotiff.py` requirements.

4.  **`src/easyidp/geotiff.py`**
    *   **Refactor**: Updated `polygon_math` to import and reuse `easyidp.roi.calculate_kernel_stats`, removing duplicated implementation of statistical kernels (`mean`, `min`, `max`, `pmin5`, etc.).

5.  **Tests**
    *   **`tests/test_roi.py`**: Added comprehensive tests for `get_z_from_pcd` covering various modes (`face`, `point`), buffers, and kernel calculations. Verified correct handling of CRS warnings.
    *   **`tests/test_pointcloud.py`**: Added tests for `crs` property, `change_crs` functionality, and basic point operations.

This implementation ensures consistent behavior between DSM and PCD data extraction while optimizing for performance on large point cloud datasets.

# CRS Handling Refactoring & PointCloud crop_polygon Method (by Claude Opus 4.5)

**Date:** 2026-01-30  
**Files Modified:**
- `src/easyidp/roi.py`
- `src/easyidp/pointcloud.py`

---

## 1. CRS Handling Refactoring

### Problem
The original CRS handling code in `get_z_from_dsm()` and `get_z_from_pcd()` would crash with `AttributeError` when trying to call `.equals()` on `None` CRS values.

### Solution
Refactored CRS handling logic to properly handle all combinations of `None`/has CRS for both ROI and target data.

### CRS Handling Matrix

| Case | `self.crs` | `target.crs` | Behavior |
|------|------------|--------------|----------|
| **1A** | `None` | `None` | ⚠️ Warning, no conversion |
| **1B** | `None` | has CRS | ⚠️ Warning, no conversion |
| **2A** | has CRS | `None` | ⚠️ Warning, no conversion |
| **2B-same** | has CRS | has CRS (same) | ✅ No conversion needed |
| **2B-diff, `keep_crs=False`** | has CRS | has CRS (different) | 🔄 Modify ROI CRS in-place |
| **2B-diff, `keep_crs=True`** | has CRS | has CRS (different) | 📋 Convert temp copy, keep original |

### Code Changes

#### `get_z_from_dsm()` (Lines 773-819)
```python
# Handle CRS conversion based on self.crs and dsm.header["crs"] combinations
dsm_crs = dsm.header["crs"]
if self.crs is None or dsm_crs is None:
    # Case 1A, 1B, 2A: at least one CRS is None
    if self.crs is None and dsm_crs is None:
        logger.warning("Both ROI and DSM have no CRS defined. ...")
    elif self.crs is None:
        logger.warning(f"ROI has no CRS but DSM has CRS [{dsm_crs.name}]. ...")
    else:
        logger.warning(f"ROI has CRS [{self.crs.name}] but DSM has no CRS. ...")
    poly_dict = self.id_item.copy()
elif self.crs.equals(dsm_crs):
    logger.debug("ROI CRS is same as DSM CRS, no conversion needed")
    poly_dict = self.id_item.copy()
else:
    if not keep_crs:
        self.change_crs(dsm_crs)
        poly_dict = self.id_item.copy()
    else:
        poly_dict = idp.geotools.convert_proj(self.id_item, self.crs, dsm_crs)
```

#### `get_z_from_pcd()` (Lines 885-931)
Same logic applied for `pcd.crs` instead of `dsm.header["crs"]`.

---

## 2. New `crop_polygon()` Method for PointCloud

### Problem
The `_get_z_in_poly()` helper function was defined locally inside `get_z_from_pcd()`, making it not reusable elsewhere.

### Solution
Moved the logic to a new public method `PointCloud.crop_polygon()` that returns all XYZ points inside a given 2D polygon.

### New Import in `pointcloud.py`
```python
from matplotlib.path import Path as mplPath
```

### New Method: `PointCloud.crop_polygon()`

```python
def crop_polygon(self, polygon_xy):
    """Get all points inside a 2D polygon.

    Uses KDTree with bounding box pre-filtering for fast spatial query,
    then applies exact polygon containment test.

    Parameters
    ----------
    polygon_xy : np.ndarray
        A 2D polygon coordinates with shape (n, 2)

    Returns
    -------
    np.ndarray
        The xyz coordinates of points inside the polygon, shape (m, 3).
        Returns empty array with shape (0, 3) if no points found.

    Examples
    --------
    >>> xyz_inside = pcd.crop_polygon(polygon)
    >>> z_values = xyz_inside[:, 2]
    """
```

### Usage Change in `roi.py`

**Before:**
```python
# Local helper function
def _get_z_in_poly(poly_pts_xy, pcd_tree, pcd_points):
    ...
    return final_z

# Pre-fetch for performance
pcd_tree = pcd.tree
pcd_pts_all = pcd.points

# Usage
z_vals = _get_z_in_poly(poly_cal, pcd_tree, pcd_pts_all)
```

**After:**
```python
# Direct method call
xyz_vals = pcd.crop_polygon(poly_cal)
z_vals = xyz_vals[:, 2] if len(xyz_vals) > 0 else np.array([])
```

---

## 3. Test Results

All tests passed after modifications:
```
tests/test_roi.py ..................... [47%]
tests/test_pointcloud.py ......................... [100%]
45 passed in 8.02s
```

---

## Benefits

1. **Fixed potential crash**: The old code would crash with `AttributeError` when CRS is `None`
2. **Cleaner code**: Guard clauses check for `None` cases first, simplifying the rest of the logic
3. **Better logging**: More descriptive messages indicating what's happening
4. **Reusable method**: `pcd.crop_polygon()` can now be used anywhere, not just in `get_z_from_pcd()`
5. **Reduced code duplication**: Removed 33 lines of local helper function code from `roi.py`
