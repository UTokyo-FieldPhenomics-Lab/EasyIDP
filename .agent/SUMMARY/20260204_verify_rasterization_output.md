# Verify Rasterization output
**Date**: 2026-02-04
**Session**: Verify Rasterization output

## Summary of Changes
1.  **Resolved Circular Import**:
    -   Modified `src/easyidp/geotiff.py` to use a string forward reference (`"idp.ROI | str | Path"`) for the `roi` parameter in `create_binary_mask`.
    -   This prevents the import cycle between `geotiff`, `roi`, and `__init__`.

2.  **Implemented `create_binary_mask`**:
    -   Added functionality to rasterize `easyidp.ROI` objects or shapefiles into a binary mask matching a target GeoTIFF's grid.
    -   Handled batch processing optimizations (shallow copy of ROI object).
    -   Fixed a bug where `item_label` was not copied during ROI batch processing, leading to empty masks.
    -   Added `all_touched` parameter to give users control over rasterization precision (default `False`, but often useful as `True` for small features).
    -   Ensured the function returns a 3D array `(H, W, 1)` to be consistent with `easyidp.GeoTiff` internal 3D storage convention.

3.  **Enhanced `GeoTiff.save` Robustness**:
    -   Updated `src/easyidp/geotiff.py`'s `save` method to correctly handle 2D `(H, W)` arrays by refraining from using `np.moveaxis` on them.
    -   This fixes `ValueError: Source shape ... is inconsistent` when saving 1-band images that were stored as 2D arrays.

4.  **Verification**:
    -   Created `tests/test_rasterize.py` to verify the feature.
    -   Debugged data path issues (using `test_data.tiff.mask_rice_geotiff_with_polygon` and corresponding shapefile).
    -   Confirmed correct rasterization (pixel sum > 0) and output file generation with `all_touched=True`.
    -   Fixed `fixture` naming (`shared_data`) in the test.

## Key Files Modified
-   `src/easyidp/geotiff.py`
-   `tests/test_rasterize.py`
-   `src/easyidp/data.py` (Test data paths updated)

## Additional Updates (Refactoring)
1.  **Renamed Function**:
    -   Renamed `create_binary_mask` to `create_binary_mask_for_geotiff` in `src/easyidp/geotiff.py` to be more explicit.
2.  **Refactored Tests**:
    -   Moved tests from `tests/test_rasterize.py` to `tests/test_geotiff.py` as a new class `TestCreateBinaryMaskForGeoTiff`.
    -   Added a new test case `test_create_binary_mask_empty_polygon` using `test_data.tiff.mask_rice_geotiff_empty_polygon` to verify empty mask generation when no polygons intersect.
    -   Cleaned up temporary test file `tests/test_rasterize.py`.
3.  **Verification**:
    -   Ran full suite `tests/test_geotiff.py` (47 tests passed).
