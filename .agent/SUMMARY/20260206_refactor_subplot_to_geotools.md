# Refactor: Merge Subplot to Geotools & ROI Save Logic

## Task Description
- Merge `src/easyidp/subplot.py` and `tests/test_subplot.py` into `src/easyidp/geotools.py` and `tests/test_geotools.py`.
- Refactor `ROI.save_shp` logic: move implementation to `src/easyidp/shp.py` as `write_shp` function, and wrap it in `ROI.save()` (and `ROI.save_shp()`).

## Changes

### 1. Code Migration
- **Moved** `generate_subplots` and related helper functions from `src/easyidp/subplot.py` to `src/easyidp/geotools.py`.
- **Deleted** `src/easyidp/subplot.py`.
- **Updated** `src/easyidp/__init__.py` to remove `subplot` sub-module imports. Subplot functionality is now accessed via `easyidp.geotools.generate_subplots` (API change: existing code using `easyidp.generate_subplots` needs update or re-export).
    - *Note*: Removed `from .subplot import generate_subplots` from top-level `__init__.py`.

### 2. ROI Save Refactoring
- **Added** `write_shp` function in `src/easyidp/shp.py`. This function handles shapefile writing logic independently of the ROI class, although it can accept an ROI object or a dictionary of polygons.
- **Updated** `ROI.save_shp` in `src/easyidp/roi.py` to delegate to `easyidp.shp.write_shp`.
- **Added** `ROI.save` method in `src/easyidp/roi.py` as a generic entry point for saving ROI data (currently supports `.shp` via delegation).

### 3. Tests
- **Created** `tests/test_geotools.py` by migrating tests from `tests/test_subplot.py`.
- **Updated** tests to use `idp.geotools.generate_subplots`.
- **Verified** all tests passed correctly, including subplot generation and shapefile saving.
- **Deleted** `tests/test_subplot.py`.

## Files Modified
- `src/easyidp/geotools.py`
- `src/easyidp/shp.py`
- `src/easyidp/roi.py`
- `src/easyidp/__init__.py`
- `tests/test_geotools.py` (Created)
- `src/easyidp/subplot.py` (Deleted)
- `tests/test_subplot.py` (Deleted)
