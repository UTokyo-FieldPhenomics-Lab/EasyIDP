# Logic Optimization for `use_affine` in `GeoTiff`

**Date:** 2026-02-04
**Session Title:** Optimization of Affine GeoTiff Logic

## Overview
Analyzed and optimized the `use_affine` parameter logic across `GeoTiff` class methods (`save`, `convert_to_affine`) and utility functions (`back2raw2geotiff`). The goal was to eliminate redundant computations, ensure consistency between memory objects and saved files, and clarify parameter ambiguity.

## Changes

### 1. `GeoTiff.save` Optimization (Optimization A)
- **Modified:** `src/easyidp/geotiff.py`
- **Change:** Added a check `if self.use_affine and use_affine:`.
- **Effect:** If the `GeoTiff` object is already in affine mode (memory), calling `save(use_affine=True)` now skips the redundant re-conversion/resampling step. It directly saves the existing affine data.
- **Fixes:** Previously, it would re-calculate the affine transformation on already transformed data, leading to potential degradation and performance loss.

### 2. `back2raw2geotiff` Consistency (Optimization B)
- **Modified:** `src/easyidp/geotiff.py` (specifically `_process_single_image_task`)
- **Change:** When `use_affine=True` is requested:
    1. The worker now explicitly calls `gtiff.convert_to_affine()` on the in-memory object.
    2. Then calls `gtiff.save()` (without `use_affine=True`, relying on the object state).
- **Effect:** The returned dictionary now contains `GeoTiff` objects that are *already* in affine mode, matching the files saved to disk.
- **Fixes:** Previously, the function saved an affine file but returned a standard (non-rotated) `GeoTiff` object, causing inconsistency for downstream users.

### 3. `get_header` Deepcopy (Optimization C)
- **Modified:** `src/easyidp/geotiff.py`
- **Change:** Changed `header['profile'] = src.profile.copy()` to `copy.deepcopy(src.profile)`.
- **Effect:** Ensures complete independence of the profile dictionary.
- **Reason:** Shallow copies of the profile (which contains mutable objects like `Affine` transform) could lead to unintended side effects when modifying the header in one place affecting others.

### 4. Bug Fixes
- **`convert_to_affine` Fix:** Fixed a potential crash where `self._imarray` was accessed directly (which might be `None` due to lazy loading). Changed to use `self.imarray` property.
- **`save` Fix:** Similar fix to ensure `self.imarray` is loaded before copying.

## Verification
- Created temporary test file `tests/test_geotiff_opt.py`.
- Verified Optimization A: Saving an already-affine object with `use_affine=True` works correctly without error.
- Verified Optimization B: `back2raw2geotiff` returns objects with `use_affine=True` and correct rotation transform.
- Regression testing: Ran full `tests/test_geotiff.py` to ensure no existing functionality was broken. All 49 tests passed (with expected warnings).

## Conclusion
The `use_affine` logic is now robust and consistent. Users can expect that if they ask for affine processing, they get it in both the saved output and the returned Python objects.
