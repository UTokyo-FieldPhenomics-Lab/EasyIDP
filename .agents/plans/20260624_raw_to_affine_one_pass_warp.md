# Raw To Affine One-Pass Warp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the automatic `back2raw2geotiff(..., use_affine=True)` two-step resampling path with a direct raw-to-affine one-pass warp that preserves raw projected GSD.

**Architecture:** Keep `one_raw_roi2geotiff()` and all `use_affine=False` behavior unchanged. Add internal raw-to-affine helpers in `src/easyidp/geotiff.py`, then route only the `back2raw2geotiff(..., use_affine=True)` worker branch through the new one-pass path. Existing DOM/DSM affine workflows (`convert_to_affine()`, `save(use_affine=True)`, `roi.crop(use_affine=True)`) remain unchanged.

**Tech Stack:** Python, NumPy, rasterio `Affine`, scikit-image `ProjectiveTransform` and `warp`, pytest, EasyIDP `TestData` Lotus Pix4D fixtures.

---

## Context

The bug is documented in `.agents/references/20260622_affine_crop_gsd_issue.md`. The current automatic affine back2raw flow is:

```text
back2raw2geotiff(..., use_affine=True)
  -> _process_single_image_task()
  -> one_raw_roi2geotiff()        # raw RGB -> axis-aligned GeoTiff, resampling #1
  -> GeoTiff.convert_to_affine()  # axis-aligned GeoTiff -> affine GeoTiff, resampling #2
```

This is wrong for the target use case because:

- The intermediate standard GeoTiff estimates `scale_x` and `scale_y` from axis-aligned bounding boxes.
- `convert_to_affine()` interprets those scales along the rotated ROI local axes.
- The RGB data is interpolated twice.
- The final output can have anisotropic and incorrect GSD, e.g. `0.00335 x 0.00644 m/px` instead of raw projected `~0.00404 m/px`.

The new flow must be:

```text
back2raw2geotiff(..., use_affine=True)
  -> _process_single_image_task()
  -> _one_raw_roi2affine_geotiff()  # raw RGB -> final affine GeoTiff, resampling once
```

Do not add `24_naro` to package test data. Use synthetic geometry for exact GSD assertions and existing Lotus `TestData` for integration coverage. Keep `24_naro` as an optional local manual validation dataset only.

---

## File Structure

**Modify:** `src/easyidp/geotiff.py`

Responsibilities:

- Keep `one_raw_roi2geotiff()` unchanged for the standard path.
- Add small module-level helpers for raw-to-affine geometry and one-pass warp.
- Change `_process_single_image_task()` so only `use_affine=True` valid rectangles use the one-pass helper.
- Keep fallback to standard `one_raw_roi2geotiff()` for non-rectangular ROI or affine helper failure.

**Modify:** `tests/test_geotiff.py`

Responsibilities:

- Replace the current failing two-step GSD regression test with a direct helper test.
- Add Lotus integration tests using existing `shared_data` fixture.
- Add a guard test proving `back2raw2geotiff` affine processing no longer calls `convert_to_affine()`.
- Keep existing `use_affine=False`, `convert_to_affine()`, `save(use_affine=True)`, and `roi.crop(use_affine=True)` tests unchanged.

**Do not modify:** `src/easyidp/data.py`

Reason:

- Existing `TestData` already includes Lotus raw photos, Pix4D params, ROI shapefile, DSM, and back2raw fixture generation through `tests/__init__.py`.
- Adding `24_naro` would increase data size and reduce CI reproducibility without improving root-cause coverage.

---

## Task 1: Replace The Synthetic Regression Test

**Files:**

- Modify: `tests/test_geotiff.py:1643-1701`

- [ ] **Step 1: Replace the current two-step GSD test with a one-pass helper test**

Replace `test_one_raw_roi2geotiff_affine_preserves_rotated_rectangle_gsd()` with:

```python
    def test_one_raw_roi2affine_geotiff_preserves_rotated_rectangle_gsd(self):
        """Direct raw-to-affine conversion preserves source edge GSD."""
        gsd = 0.004
        geo_width, geo_height = 0.9, 0.6
        raw_width, raw_height = geo_width / gsd, geo_height / gsd
        geo_angle = math.radians(30.0)
        raw_angle = math.radians(88.0)
        geo_rot = np.array([
            [math.cos(geo_angle), -math.sin(geo_angle)],
            [math.sin(geo_angle), math.cos(geo_angle)],
        ])
        raw_rot = np.array([
            [math.cos(raw_angle), -math.sin(raw_angle)],
            [math.sin(raw_angle), math.cos(raw_angle)],
        ])

        geo_local = np.array([
            [-geo_width / 2, geo_height / 2],
            [geo_width / 2, geo_height / 2],
            [geo_width / 2, -geo_height / 2],
            [-geo_width / 2, -geo_height / 2],
        ])
        raw_local = np.array([
            [-raw_width / 2, raw_height / 2],
            [raw_width / 2, raw_height / 2],
            [raw_width / 2, -raw_height / 2],
            [-raw_width / 2, -raw_height / 2],
        ])

        roi_geo = geo_local @ geo_rot.T + np.array([1000.0, 2000.0])
        roi_raw = raw_local @ raw_rot.T + np.array([180.0, 180.0])
        roi_geo = np.vstack([roi_geo, roi_geo[0]])
        roi_raw = np.vstack([roi_raw, roi_raw[0]])

        rows, cols = np.mgrid[0:360, 0:360]
        raw_img = np.dstack([
            cols.astype(np.uint8),
            rows.astype(np.uint8),
            ((rows + cols) % 256).astype(np.uint8),
        ])

        affine_gtiff = idp.geotiff._one_raw_roi2affine_geotiff(
            roi_crs=pyproj.CRS.from_epsg(32654),
            roi_geo_coords=roi_geo,
            raw_img=raw_img,
            roi_raw_px_coords=roi_raw,
            nodata=0,
            has_alpha=False,
        )

        transform = affine_gtiff.header["profile"]["transform"]
        col_gsd = float(np.hypot(transform.a, transform.d))
        row_gsd = float(np.hypot(transform.b, transform.e))

        assert affine_gtiff.use_affine is True
        assert col_gsd == pytest.approx(gsd, rel=0.05)
        assert row_gsd == pytest.approx(gsd, rel=0.05)
        assert affine_gtiff.width == pytest.approx(raw_width, abs=2)
        assert affine_gtiff.height == pytest.approx(raw_height, abs=2)
        assert affine_gtiff.imarray.sum() > 0
```

- [ ] **Step 2: Run the new test before implementation**

Run:

```bash
uv run pytest tests/test_geotiff.py::TestUseAffineParamOptimization::test_one_raw_roi2affine_geotiff_preserves_rotated_rectangle_gsd -q
```

Expected result:

```text
FAILED ... AttributeError: module 'easyidp.geotiff' has no attribute '_one_raw_roi2affine_geotiff'
```

---

## Task 2: Add Raw-To-Affine Geometry Helpers

**Files:**

- Modify: `src/easyidp/geotiff.py`, insert helpers after `pixel2geo()` and before `one_raw_roi2geotiff()`.

- [ ] **Step 1: Add `_prepare_raw_roi_inputs()`**

Add:

```python
def _prepare_raw_roi_inputs(
    raw_img: str | Path | np.ndarray,
    roi_geo_coords: np.ndarray,
    roi_raw_px_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw image and remove duplicated ROI closure points.

    Parameters
    ----------
    raw_img : str, pathlib.Path, or numpy.ndarray
        Raw image path or pre-loaded raw image array.
    roi_geo_coords : numpy.ndarray
        ROI coordinates in projected CRS, shape (n, 2) or (n, 3).
    roi_raw_px_coords : numpy.ndarray
        Matching ROI coordinates in raw image pixels, shape (n, 2).

    Returns
    -------
    tuple of numpy.ndarray
        Raw image array, 2D geo coordinates, and raw pixel coordinates.

    Examples
    --------
    >>> raw_img, roi_geo, roi_px = _prepare_raw_roi_inputs(path, geo, px)
    """
    if isinstance(raw_img, (str, Path)):
        raw_img_path = Path(raw_img)
        if not raw_img_path.exists():
            raise FileNotFoundError(f"Raw image not found: {raw_img_path}")
        raw_img = imread(raw_img_path)

    roi_geo_2d = np.asarray(roi_geo_coords[:, :2], dtype=float).copy()
    roi_raw_px = np.asarray(roi_raw_px_coords, dtype=float).copy()
    if np.allclose(roi_geo_2d[0], roi_geo_2d[-1]):
        roi_geo_2d = roi_geo_2d[:-1]
        roi_raw_px = roi_raw_px[:-1]

    return raw_img, roi_geo_2d, roi_raw_px
```

- [ ] **Step 2: Add `_estimate_raw_edge_gsd()`**

Add:

```python
def _estimate_raw_edge_gsd(roi_geo_2d: np.ndarray, roi_raw_px: np.ndarray) -> float:
    """Estimate projected raw GSD from corresponding polygon edges.

    Parameters
    ----------
    roi_geo_2d : numpy.ndarray
        ROI coordinates in projected CRS, shape (n, 2).
    roi_raw_px : numpy.ndarray
        Matching raw image pixel coordinates, shape (n, 2).

    Returns
    -------
    float
        Median projected ground sampling distance in CRS units per pixel.

    Examples
    --------
    >>> _estimate_raw_edge_gsd(roi_geo_2d, roi_raw_px)
    0.004
    """
    if len(roi_geo_2d) != len(roi_raw_px) or len(roi_geo_2d) < 3:
        raise ValueError("Geo and raw ROI coordinates must have matching polygon vertices.")

    geo_edges = np.roll(roi_geo_2d, -1, axis=0) - roi_geo_2d
    raw_edges = np.roll(roi_raw_px, -1, axis=0) - roi_raw_px
    geo_lengths = np.linalg.norm(geo_edges, axis=1)
    raw_lengths = np.linalg.norm(raw_edges, axis=1)
    valid = raw_lengths > 1e-10
    if not np.any(valid):
        raise ValueError("Raw ROI polygon has no non-zero edges.")

    return float(np.median(geo_lengths[valid] / raw_lengths[valid]))
```

- [ ] **Step 3: Add `_crop_raw_image_by_roi()`**

Add:

```python
def _crop_raw_image_by_roi(
    raw_img: np.ndarray,
    roi_raw_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop raw image around ROI with the existing 10 percent buffer.

    Parameters
    ----------
    raw_img : numpy.ndarray
        Raw image array in HWC or HW layout.
    roi_raw_px : numpy.ndarray
        ROI pixel coordinates on the raw image, shape (n, 2).

    Returns
    -------
    tuple of numpy.ndarray
        Cropped image and local ROI pixel coordinates.

    Examples
    --------
    >>> cropped, local_px = _crop_raw_image_by_roi(raw_img, roi_raw_px)
    """
    img_height, img_width = raw_img.shape[:2]
    roi_min = roi_raw_px.min(axis=0)
    roi_max = roi_raw_px.max(axis=0)
    buffer_size = (roi_max - roi_min) * 0.1
    buffered_min = np.maximum(roi_min - buffer_size, [0, 0]).astype(np.int32)
    buffered_max = np.minimum(roi_max + buffer_size, [img_width, img_height]).astype(
        np.int32
    )

    cropped_img = raw_img[
        buffered_min[1] : buffered_max[1], buffered_min[0] : buffered_max[0]
    ]
    return cropped_img, roi_raw_px - buffered_min
```

- [ ] **Step 4: Add `_estimate_projective_transform()`**

Add:

```python
def _estimate_projective_transform(src: np.ndarray, dst: np.ndarray) -> ProjectiveTransform:
    """Estimate a projective transform with skimage version compatibility.

    Parameters
    ----------
    src : numpy.ndarray
        Source points, shape (n, 2).
    dst : numpy.ndarray
        Destination points, shape (n, 2).

    Returns
    -------
    skimage.transform.ProjectiveTransform
        Estimated transform from source to destination coordinates.

    Examples
    --------
    >>> pt = _estimate_projective_transform(raw_px, dst_px)
    """
    pt = ProjectiveTransform()
    try:
        return ProjectiveTransform.from_estimate(src=src, dst=dst)
    except AttributeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            pt.estimate(src=src, dst=dst)
        return pt
```

- [ ] **Step 5: Run import smoke test**

Run:

```bash
uv run python - <<'PY'
import easyidp as idp
print(idp.geotiff._estimate_raw_edge_gsd)
PY
```

Expected result:

```text
<function _estimate_raw_edge_gsd at ...>
```

---

## Task 3: Implement `_one_raw_roi2affine_geotiff()`

**Files:**

- Modify: `src/easyidp/geotiff.py`, insert after helpers from Task 2 and before `one_raw_roi2geotiff()`.

- [ ] **Step 1: Add `_build_raw_affine_header()`**

Add:

```python
def _build_raw_affine_header(
    roi_crs: pyproj.CRS,
    imarray: np.ndarray,
    transform,
    origin: np.ndarray,
    target_gsd: float,
    nodata: float | int,
    has_alpha: bool,
) -> dict:
    """Build GeoTiff header for direct raw-to-affine output.

    Parameters
    ----------
    roi_crs : pyproj.CRS
        Output coordinate reference system.
    imarray : numpy.ndarray
        Output image array.
    transform : affine.Affine
        Output affine transform.
    origin : numpy.ndarray
        Affine top-left origin in geo coordinates.
    target_gsd : float
        Output projected ground sampling distance.
    nodata : float or int
        Nodata value.
    has_alpha : bool
        Whether the output should omit nodata for alpha-capable images.

    Returns
    -------
    dict
        EasyIDP GeoTiff header dictionary.

    Examples
    --------
    >>> header = _build_raw_affine_header(crs, img, transform, origin, 0.004, 0, True)
    """
    n_bands = imarray.shape[2] if imarray.ndim == 3 else 1
    return {
        "height": imarray.shape[0],
        "width": imarray.shape[1],
        "dim": n_bands,
        "dtype": imarray.dtype,
        "nodata": nodata if not has_alpha else None,
        "scale": [target_gsd, target_gsd],
        "tie_point": [origin[0], origin[1]],
        "crs": roi_crs,
        "has_alpha": False,
        "transform": transform,
        "profile": {
            "driver": "GTiff",
            "height": imarray.shape[0],
            "width": imarray.shape[1],
            "count": n_bands,
            "dtype": str(imarray.dtype),
            "crs": roi_crs,
            "transform": transform,
        },
    }
```

- [ ] **Step 2: Add the one-pass affine helper**

Add:

```python
def _one_raw_roi2affine_geotiff(
    roi_crs: pyproj.CRS,
    roi_geo_coords: np.ndarray,
    raw_img: str | Path | np.ndarray,
    roi_raw_px_coords: np.ndarray,
    nodata: float | int = 0,
    has_alpha: bool = True,
) -> GeoTiff:
    """Transform a rectangular raw ROI directly into affine GeoTiff storage.

    Parameters
    ----------
    roi_crs : pyproj.CRS
        The coordinate reference system of the ROI.
    roi_geo_coords : numpy.ndarray
        GIS coordinates of ROI polygon, shape (n, 2) or (n, 3).
    raw_img : str, pathlib.Path, or numpy.ndarray
        Raw image path or pre-loaded raw image array.
    roi_raw_px_coords : numpy.ndarray
        ROI pixel coordinates on the raw image, shape (n, 2).
    nodata : float or int, optional
        Value for pixels outside the ROI, by default 0.
    has_alpha : bool, optional
        If True, keep nodata unset for RGB-like outputs, by default True.

    Returns
    -------
    GeoTiff
        A GeoTiff object already in affine mode.

    Examples
    --------
    >>> gtiff = _one_raw_roi2affine_geotiff(crs, roi_geo, raw_img, roi_px)
    >>> gtiff.use_affine
    True
    """
    from rasterio.transform import Affine

    raw_img, roi_geo_2d, roi_raw_px = _prepare_raw_roi_inputs(
        raw_img, roi_geo_coords, roi_raw_px_coords
    )
    rect_info = GeoTiff()._get_rectangle_affine_info(roi_geo_2d)
    if rect_info is None:
        raise ValueError("Polygon is not a valid rectangle for affine raw conversion.")

    target_gsd = _estimate_raw_edge_gsd(roi_geo_2d, roi_raw_px)
    if not np.isfinite(target_gsd) or target_gsd <= 0:
        raise ValueError("Estimated raw edge GSD must be a positive finite value.")

    origin = rect_info["origin"]
    col_vec = rect_info["col_vec"]
    row_vec = rect_info["row_vec"]
    out_width = int(np.ceil(rect_info["width"] / target_gsd))
    out_height = int(np.ceil(rect_info["height"] / target_gsd))

    local_geo = roi_geo_2d - origin
    dst_px = np.column_stack([
        local_geo @ col_vec / target_gsd,
        local_geo @ row_vec / target_gsd,
    ])
    cropped_img, roi_local_px = _crop_raw_image_by_roi(raw_img, roi_raw_px)
    pt = _estimate_projective_transform(roi_local_px, dst_px)

    output_shape = (out_height, out_width)
    if cropped_img.ndim == 3:
        output_shape = (out_height, out_width, cropped_img.shape[2])
    warped_img = warp(
        cropped_img,
        pt.inverse,
        output_shape=output_shape,
        order=1,
        preserve_range=True,
        cval=nodata,
    ).astype(cropped_img.dtype)

    transform = Affine(
        target_gsd * col_vec[0],
        target_gsd * row_vec[0],
        origin[0],
        target_gsd * col_vec[1],
        target_gsd * row_vec[1],
        origin[1],
    )
    header = _build_raw_affine_header(
        roi_crs, warped_img, transform, origin, target_gsd, nodata, has_alpha
    )

    mask = np.ones((out_height, out_width), dtype=bool)
    gtiff = GeoTiff(imarray=warped_img, header=header, mask=mask)
    gtiff.set_mask_polygon(np.vstack([roi_geo_2d, roi_geo_2d[0]]), is_geo=True)
    gtiff._use_affine = True
    return gtiff
```

- [ ] **Step 3: Run the synthetic helper test**

Run:

```bash
uv run pytest tests/test_geotiff.py::TestUseAffineParamOptimization::test_one_raw_roi2affine_geotiff_preserves_rotated_rectangle_gsd -q
```

Expected result:

```text
1 passed
```

---

## Task 4: Route `back2raw2geotiff(use_affine=True)` Through One-Pass Helper

**Files:**

- Modify: `src/easyidp/geotiff.py:3043-3065`

- [ ] **Step 1: Replace the worker branch**

Replace the current unconditional `one_raw_roi2geotiff()` followed by `convert_to_affine()` block with:

```python
            if use_affine:
                try:
                    gtiff = _one_raw_roi2affine_geotiff(
                        roi_crs=roi_crs,
                        roi_geo_coords=roi_geo_coords,
                        raw_img=full_image,
                        roi_raw_px_coords=px_coords,
                        nodata=nodata,
                        has_alpha=has_alpha,
                    )
                except ValueError as e:
                    logger.warning(
                        f"Could not convert ROI {roi_id} on image {task['img_id']} "
                        f"to one-pass affine mode: {e}. Falling back to standard storage."
                    )
                    gtiff = one_raw_roi2geotiff(
                        roi_crs=roi_crs,
                        roi_geo_coords=roi_geo_coords,
                        raw_img=full_image,
                        roi_raw_px_coords=px_coords,
                        nodata=nodata,
                        has_alpha=has_alpha,
                    )
            else:
                gtiff = one_raw_roi2geotiff(
                    roi_crs=roi_crs,
                    roi_geo_coords=roi_geo_coords,
                    raw_img=full_image,
                    roi_raw_px_coords=px_coords,
                    nodata=nodata,
                    has_alpha=has_alpha,
                )
```

- [ ] **Step 2: Run existing standard path tests**

Run:

```bash
uv run pytest tests/test_geotiff.py::test_one_raw_roi2geotiff tests/test_geotiff.py::test_one_raw_roi2geotiff_options tests/test_geotiff.py::test_back2raw2geotiff tests/test_geotiff.py::test_back2raw2geotiff_no_save -q
```

Expected result:

```text
4 passed
```

- [ ] **Step 3: Run existing affine optimization test**

Run:

```bash
uv run pytest tests/test_geotiff.py::TestUseAffineParamOptimization::test_back2raw2geotiff_optimization_B -q
```

Expected result:

```text
1 passed
```

If the Lotus first ROI is axis-aligned enough that `transform.b` and `transform.d` are both close to zero, change only the rotation assertion to compare pixel vector lengths and `use_affine`, because axis-aligned affine storage is valid:

```python
        assert gtiff.use_affine is True
        t = gtiff.header["transform"]
        assert float(np.hypot(t.a, t.d)) > 0
        assert float(np.hypot(t.b, t.e)) > 0
```

---

## Task 5: Add Lotus Integration Tests

**Files:**

- Modify: `tests/test_geotiff.py`, add methods under `class TestUseAffineParamOptimization` after `test_back2raw2geotiff_optimization_B`.

- [ ] **Step 1: Add a small GSD utility in the test file**

Add near existing test helpers at the top of `tests/test_geotiff.py`:

```python
def _edge_median_gsd(roi_geo, roi_raw_px):
    """Return median projected GSD from matching geo/raw edges."""
    roi_geo = np.asarray(roi_geo[:, :2], dtype=float)
    roi_raw_px = np.asarray(roi_raw_px, dtype=float)
    if np.allclose(roi_geo[0], roi_geo[-1]):
        roi_geo = roi_geo[:-1]
        roi_raw_px = roi_raw_px[:-1]

    geo_edges = np.roll(roi_geo, -1, axis=0) - roi_geo
    raw_edges = np.roll(roi_raw_px, -1, axis=0) - roi_raw_px
    raw_lengths = np.linalg.norm(raw_edges, axis=1)
    return float(np.median(np.linalg.norm(geo_edges, axis=1) / raw_lengths))
```

- [ ] **Step 2: Add Lotus GSD integration test**

Add:

```python
    def test_back2raw2geotiff_lotus_affine_preserves_raw_edge_gsd(self, shared_data, tmp_path):
        """Lotus back2raw affine output keeps raw edge-derived GSD."""
        p4d = shared_data["p4d"]
        roi = shared_data["roi"]
        out_all = shared_data["out_all"]
        out_folder = tmp_path / "lotus_affine_gsd"

        results = idp.geotiff.back2raw2geotiff(
            recons=p4d,
            back2raw_result=out_all,
            roi=roi,
            output_folder=out_folder,
            use_affine=True,
            num_workers=1,
        )

        roi_id = next(iter(results))
        img_id = next(iter(results[roi_id]))
        gtiff = results[roi_id][img_id]
        expected_gsd = _edge_median_gsd(roi[roi_id], out_all[roi_id][img_id])
        transform = gtiff.header["profile"]["transform"]

        assert gtiff.use_affine is True
        assert float(np.hypot(transform.a, transform.d)) == pytest.approx(
            expected_gsd, rel=0.05
        )
        assert float(np.hypot(transform.b, transform.e)) == pytest.approx(
            expected_gsd, rel=0.05
        )
        assert gtiff.imarray.sum() > 0

        saved_file = out_folder / str(roi_id) / f"{img_id}.tif"
        reloaded = idp.GeoTiff(saved_file)
        assert reloaded.use_affine is True
```

- [ ] **Step 3: Add guard test that prevents returning to the two-step path**

Add:

```python
    def test_process_single_image_task_use_affine_does_not_call_convert_to_affine(
        self, shared_data, monkeypatch
    ):
        """The back2raw affine worker uses direct one-pass raw conversion."""
        p4d = shared_data["p4d"]
        roi = shared_data["roi"]
        out_all = shared_data["out_all"]
        roi_id = next(iter(out_all))
        img_id = next(iter(out_all[roi_id]))
        img_path = p4d.photos[img_id].path

        def fail_convert_to_affine(self):
            raise AssertionError("convert_to_affine should not be called")

        monkeypatch.setattr(idp.GeoTiff, "convert_to_affine", fail_convert_to_affine)

        task = {
            "img_id": img_id,
            "img_path": img_path,
            "rois": {roi_id: out_all[roi_id][img_id]},
        }
        common_args = {
            "roi_static_data": {roi_id: roi[roi_id][:, :2]},
            "roi_crs": roi.crs,
            "nodata": 0,
            "has_alpha": True,
            "output_folder": None,
            "use_affine": True,
        }

        results = idp.geotiff._process_single_image_task(task, common_args)
        assert results[roi_id].use_affine is True
```

- [ ] **Step 4: Run the new Lotus tests**

Run:

```bash
uv run pytest tests/test_geotiff.py::TestUseAffineParamOptimization::test_back2raw2geotiff_lotus_affine_preserves_raw_edge_gsd tests/test_geotiff.py::TestUseAffineParamOptimization::test_process_single_image_task_use_affine_does_not_call_convert_to_affine -q
```

Expected result:

```text
2 passed
```

---

## Task 6: Verify Existing Affine Workflows Are Unchanged

**Files:**

- No file edits if tests pass.

- [ ] **Step 1: Run DOM/DSM affine conversion tests**

Run:

```bash
uv run pytest tests/test_geotiff.py::TestAffineConversion tests/test_geotiff.py::TestAffineCrop -q
```

Expected result:

```text
all selected tests passed
```

- [ ] **Step 2: Run save affine optimization test**

Run:

```bash
uv run pytest tests/test_geotiff.py::TestUseAffineParamOptimization::test_save_optimization_A -q
```

Expected result:

```text
1 passed
```

- [ ] **Step 3: Run geotiff test module**

Run:

```bash
uv run pytest tests/test_geotiff.py -q
```

Expected result:

```text
all tests passed
```

---

## Task 7: Quality Gates

**Files:**

- No file edits unless tools identify issues.

- [ ] **Step 1: Run ruff on changed files**

Run:

```bash
uv run ruff check src/easyidp/geotiff.py tests/test_geotiff.py
```

Expected result:

```text
All checks passed!
```

- [ ] **Step 2: Run mypy if configured for this repository**

Run:

```bash
uv run mypy src/easyidp/geotiff.py
```

Expected result:

```text
Success: no issues found in 1 source file
```

If the repository has existing third-party stub issues unrelated to this change, record the exact output in the final implementation report and do not broaden this fix.

- [ ] **Step 3: Optional local 24_naro validation**

Run the downstream notebook or script that generated the original file, then inspect the fixed output with:

```bash
uv run python - <<'PY'
from pathlib import Path
import rasterio

p = Path('/home/crest/e/hwang_Pro/data/202604_naro_rice/04_slices/R6_batch.grid_projected/2408140835_ikei_15m_200_Auto/raw/grid_003395.tif')
with rasterio.open(p) as ds:
    t = ds.transform
    print('width,height:', ds.width, ds.height)
    print('pixel vector lengths:', (t.a*t.a + t.d*t.d) ** 0.5, (t.b*t.b + t.e*t.e) ** 0.5)
PY
```

Expected approximate result for the original example:

```text
width,height: around 223 149
pixel vector lengths: around 0.00404 0.00404
```

This is a manual validation step only. Do not add `24_naro` files to `TestData` unless synthetic and Lotus tests pass but the real case still fails.

---

## Review Checklist

- [ ] `back2raw2geotiff(..., use_affine=False)` still calls `one_raw_roi2geotiff()` and keeps the old axis-aligned output behavior.
- [ ] `one_raw_roi2geotiff()` body is unchanged except possible refactoring shared helpers that preserve output byte-for-byte or test-equivalent behavior.
- [ ] `back2raw2geotiff(..., use_affine=True)` no longer calls `convert_to_affine()` in the worker path.
- [ ] New affine output uses one explicit `warp(..., order=1)` resampling step.
- [ ] Final affine transform column and row vector lengths match raw edge-derived GSD.
- [ ] `GeoTiff.convert_to_affine()` still supports DOM/DSM crop workflows.
- [ ] `GeoTiff.save(use_affine=True)` still works for standard GeoTiff objects.
- [ ] `roi.crop(dom, use_affine=True)` still works and remains unrelated to raw-to-affine conversion.
- [ ] No `24_naro` files are added to `TestData` or repository fixtures.

---

## Commit Guidance

Do not commit unless the user explicitly requests it. If a commit is requested after implementation and verification, use:

```bash
git add src/easyidp/geotiff.py tests/test_geotiff.py .agents/plans/20260624_raw_to_affine_one_pass_warp.md .agents/references/20260622_affine_crop_gsd_issue.md
git commit -m "fix(geotiff): warp back2raw affine crops in one pass" \
  -m "- add direct raw-to-affine GeoTiff conversion for rectangular back2raw ROIs" \
  -m "- preserve raw edge-derived GSD for affine outputs" \
  -m "- keep non-affine back2raw and DOM crop affine paths unchanged"
```
