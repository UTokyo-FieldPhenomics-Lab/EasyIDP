# Affine Rectangle Vertex Order Fix Summary

## Background

This session investigated and fixed a GeoTIFF affine conversion bug in `idp.geotiff.back2raw2geotiff(..., use_affine=True)`.

The issue was observed while processing:

- Notebook: `/home/crest/e/hwang_Pro/jupyter/24_naro_rice/01_slice_dom.ipynb`
- DOM crop output: `/home/crest/e/hwang_Pro/data/202604_naro_rice/04_slices/R5_batch/2308070820_ikei_15_100_200/dom/0-1.tif`
- Raw output: `/home/crest/e/hwang_Pro/data/202604_naro_rice/04_slices/R5_batch/2308070820_ikei_15_100_200/raw/0-1.tif`

Symptoms:

- `roi.crop(dom, ...)` produced a correct DOM slice.
- `roi.back2raw(ms)` projected ROI vertices correctly onto raw images.
- `gtif_dict['0-1']['DJI_0358'].imarray` became all zeros when `use_affine=True`.
- The saved raw GeoTIFF appeared below the correct DOM position in QGIS.

## Root Cause

The original affine conversion assumed that rectangle vertices were ordered as:

```text
top-left -> top-right -> bottom-right -> bottom-left
```

The notebook generated ROIs as:

```python
roi[label] = np.asarray([
    (x_min, y_min),
    (x_max, y_min),
    (x_max, y_max),
    (x_min, y_max),
    (x_min, y_min),
])
```

This is bottom-left first. The previous implementation treated the first vertex as the affine origin and sampled rows using a fixed y direction. For bottom-left-first rectangles, sampling moved outside the source crop, so `map_coordinates(..., cval=0)` filled the output with zeros.

## Implementation

Changed file:

- `src/easyidp/geotiff.py`

Main changes:

- Added order-independent rectangle geometry parsing via `_get_rectangle_affine_info()`.
- Added centroid-based point sorting helper `_sort_rectangle_points()`.
- Added rectangle validation helper `_is_ordered_rectangle()`.
- Changed `_prepare_affine_storage()` to use explicit vectors:

```text
geo = origin + col * pixel_size_x * col_vec + row * pixel_size_y * row_vec
```

- Built the affine transform from the same `col_vec` and `row_vec` used for sampling.
- Preserved the old `_is_valid_rectangle()` return interface for compatibility.
- Fixed affine pixel size calculation with `_transform_pixel_sizes()` so 90-degree or tall rectangles do not produce `scale=[0, 0]`.
- Updated `convert_to_affine()` and `save(use_affine=True)` to use the new rectangle geometry flow.

## Tests Added

Changed file:

- `tests/test_geotiff.py`

New test coverage:

- Bottom-left-first axis-aligned rectangles preserve data and bounds.
- Tall rectangles preserve non-zero pixel scale and can be converted back from affine.
- Rotated rectangles are invariant to:
  - different starting corners,
  - reversed winding,
  - non-cyclic vertex permutations.

New helper assertions:

- `_affine_pixel_corners()` maps affine pixel corners into geo coordinates.
- `_assert_same_corner_set()` compares rectangle corners independent of order.

## Review Feedback Addressed

A code-review subagent identified a high-severity regression risk:

- Tall axis-aligned rectangles could select a vertical edge as `col_vec`, creating an affine transform where `transform.a == 0` and `transform.e == 0`.
- The previous `scale=[abs(transform.a), abs(transform.e)]` logic then produced zero pixel sizes and broke `convert_from_affine()`.

Fix:

- Pixel sizes are now computed from affine column and row vector norms:

```text
pixel_size_x = hypot(transform.a, transform.d)
pixel_size_y = hypot(transform.b, transform.e)
```

Additional tests were added for this case.

## Verification

Commands run:

```bash
uv run pytest
```

Final result:

```text
230 passed, 1 skipped, 37 warnings
```

Additional checks:

```bash
uv run python -m compileall src/easyidp/geotiff.py tests/test_geotiff.py
```

Result: passed.

Ruff check:

```bash
uv run python -m ruff check src/easyidp/geotiff.py tests/test_geotiff.py
```

Result: failed because `ruff` is not installed in the current `.venv`.

Mypy check:

```bash
uv run python -m mypy src/easyidp/geotiff.py
```

Result: failed with existing project-level typing issues, mainly missing third-party stubs, decorator typing, and existing Optional handling. The new `_get_rectangle_affine_info()` call-site type issue was resolved.

## Notes

The working tree had unrelated modified files at the end of the session:

- `pyproject.toml`
- `src/easyidp/__init__.py`
- `uv.lock`

They were not intentionally changed as part of this affine rectangle fix.
