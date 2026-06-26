# PointCloud Refactor Architecture Plan

> **Scope:** This is the approved architecture-level plan for the v2.1 point-cloud refactor. It intentionally avoids full function-body implementation details. Detailed task steps and tests should be written before implementation.

**Goal:** Make `idp.PointCloud` faster and semantically clearer while preserving existing point-cloud workflows as much as possible.

**Architecture:** Keep `PointCloud` as the only public point-cloud class for v2.1. Internally split the current monolithic module into `core.py`, `geometry.py`, `compat.py`, and `io/` backends. Use `select_by_index()` plus KDTree-backed crop index queries as the core selection model, and keep old crop APIs as compatibility wrappers with their existing inputs and outputs.

**Tech Stack:** NumPy, SciPy `cKDTree`, Shapely, pyproj, laspy, plyfile, tqdm, EasyIDP ROI.

---

## Historical Context

Git history shows the current API grew in layers:

| API | Introduced By | Meaning |
|---|---|---|
| `crop_point_cloud()` | `6c65a3e Support point cloud crop ROI` | Original single-polygon crop path returning a `PointCloud`. |
| `crop_rois()` | `5a41bd1 support *crop function` | Batch ROI wrapper around `crop_point_cloud()`. |
| `cKDTree` point queries | `17d32d9 feat(pcd): get z from pcd` | Performance path for PCD height extraction. |
| `crop_polygon()` | `5660fbe fix(pcd): modify by claude` | Reusable KDTree-backed polygon query returning raw XYZ arrays. |

The result is two overlapping crop engines: the old object-returning crop path and the newer KDTree query path. The refactor should merge the internals without breaking the old public behavior.

## Design Principles

- Prefer one clean public class: `easyidp.PointCloud`.
- Do not expose public `PointCloudData`, public cropper classes, public spatial index classes, or reader registries in v2.1.
- Keep Open3D out of core dependencies; mimic useful API ideas such as `select_by_index()`, not the dependency stack.
- Use KDTree-backed index queries for point-cloud spatial selection.
- Use `select_by_index()` as the only internal place that copies points, colors, normals, CRS, offset, and metadata into a cropped point cloud.
- Preserve `offset` as a first-class internal storage concept for projected CRS precision.
- Preserve existing legacy crop inputs and outputs through `compat.py` until there is a deliberate deprecation cycle.
- Do not optimize for streaming large LAS/LAZ in this refactor, but avoid designs that prevent chunked IO later.

## Proposed File Layout

```text
src/easyidp/pointcloud/
  __init__.py
  core.py
  geometry.py
  compat.py
  io/
    __init__.py
    las.py
    ply.py
```

## File Responsibilities

| File | Responsibility |
|---|---|
| `pointcloud/__init__.py` | Export public `PointCloud`, new `read_point_cloud`, new `write_point_cloud`, and legacy standalone compatibility names. |
| `pointcloud/core.py` | Define `PointCloud`, point data properties, CRS/offset behavior, `select_by_index()`, `crop()`, `to_crs()`, `change_crs()`, `save()`, and conversion helpers. |
| `pointcloud/geometry.py` | Normalize supported crop geometries and query selected point indices using the internal XY KDTree plus exact polygon containment. It remains importable as an Advanced API, but is not part of the default public API. |
| `pointcloud/compat.py` | Hold old API wrappers such as `crop_point_cloud()`, `crop_polygon()`, `crop_rois()`, `read_ply()`, `read_las()`, `read_laz()`, `write_ply()`, `write_las()`, and `write_laz()` while preserving their current inputs and outputs. Every function in this module emits `FutureWarning` and recommends the new API. |
| `pointcloud/io/__init__.py` | Route new read/write calls by explicit `format` or path suffix. |
| `pointcloud/io/las.py` | Read and write LAS/LAZ using laspy, including CRS and offset metadata where practical. |
| `pointcloud/io/ply.py` | Read and write PLY using plyfile. |

## Public API Shape

### Main Class

```python
idp.PointCloud(path=None, offset=None)
```

`PointCloud` remains the only public point-cloud object. It owns point coordinates, optional colors, optional normals, CRS, offset, file metadata, and the internal spatial-index cache.

### Core Properties

| Property | Type | Notes |
|---|---|---|
| `points` | `np.ndarray` or `None` | Absolute XYZ coordinates exposed to users. Internally stored as local coordinates plus offset. |
| `colors` | `np.ndarray` or `None` | Optional RGB values. |
| `normals` | `np.ndarray` or `None` | Optional normal vectors. |
| `crs` | `pyproj.CRS` or `None` | `None` is valid for local or indoor reconstructions. |
| `offset` | `np.ndarray` | Three-element XYZ offset used to preserve precision for large coordinates. |
| `tree` | `scipy.spatial.cKDTree` or `None` | Internal XY index exposed only for current compatibility. New code should use geometry query helpers. |

### Core Methods

| Method | Input | Output | Purpose |
|---|---|---|---|
| `select_by_index(indices, invert=False)` | integer sequence | `PointCloud` | Copy selected points and all semantically valid metadata into a new `PointCloud`. |
| `crop(target, invert=False)` | Shapely geometry or EasyIDP ROI | `PointCloud` or `dict[str, PointCloud]` | New crop entry point. |
| `to_crs(target_crs)` | CRS-like input | `PointCloud` | Return a converted copy without mutating the source. |
| `change_crs(target_crs)` | CRS-like input | `None` | Mutate the current object in place. |
| `save(path)` | path-like | `None` | Save point cloud by extension. |
| `to_numpy(copy=True)` | bool | `np.ndarray` | Return XYZ points only. |
| `has_points()` | none | bool | Existing predicate. |
| `has_colors()` | none | bool | Existing predicate. |
| `has_normals()` | none | bool | Existing predicate. |
| `is_empty()` | none | bool | Convenience predicate. |

## Crop API Rules

### New API Inputs

| Input | Output | Notes |
|---|---|---|
| `shapely.Polygon` | `PointCloud` | Crop by a single polygon along the Z axis. |
| `shapely.MultiPolygon` | `PointCloud` | Treat all parts as one logical geometry and return one point cloud. |
| `shapely.box(minx, miny, maxx, maxy)` | `PointCloud` | BBox crop represented as an ordinary Shapely polygon. |
| `easyidp.ROI` | `dict[str, PointCloud]` | Crop each ROI feature and preserve ROI labels. This remains a supported long-term API. |

### Legacy API Inputs

Legacy APIs keep their current inputs and outputs for compatibility:

| Old API | Current Input | Current Output | Refactored Behavior |
|---|---|---|---|
| `crop_point_cloud(polygon_xy)` | `np.ndarray` with shape `(n, 2)` | `PointCloud` or `None` for empty crop | Validate the old input shape, query indices through `geometry.py`, return `select_by_index(indices)` or `None` to preserve current behavior. |
| `crop_polygon(polygon_xy)` | array-like polygon with at least 2 columns | `np.ndarray` with shape `(m, 3)` | Query the same indices and return selected absolute XYZ points. |
| `crop_rois(roi, save_folder=None)` | `easyidp.ROI` or `dict` | `dict[str, PointCloud]` | Loop over ROI items, call the shared query plus `select_by_index()`, and preserve save behavior. |

Compatibility wrappers live in `compat.py`. They should not contain independent crop algorithms. Every wrapper emits `FutureWarning` because compatibility APIs may be deprecated after the v2.1 migration window.

## Crop Dispatch Model

New `PointCloud.crop()` should stay small and dispatch by input type:

```text
crop(target)
  -> Shapely Polygon or MultiPolygon -> query indices -> select_by_index() -> PointCloud
  -> EasyIDP ROI -> crop each feature -> dict[str, PointCloud]
  -> unsupported input -> TypeError with migration guidance
```

Legacy methods use the same query/select internals but preserve their historical input validation and return types.

`PointCloud.crop(ROI)` and `ROI.crop(pcd)` should both remain available long-term. `ROI.crop(pcd)` should be a thin wrapper over the point-cloud crop core rather than an independent implementation.

## Geometry Query Model

The shared query flow is:

```text
polygon geometry
  -> bounding box
  -> cKDTree.query_ball_point(..., p=np.inf)
  -> exact polygon containment on candidate XY points
  -> selected point indices
```

Performance requirements:

- Build the XY KDTree lazily and invalidate it whenever `_points` or `_offset` changes.
- Avoid repeated `self.points` calls in hot paths because `points` allocates `_points + _offset`.
- For exact containment, compute absolute XY only for candidate points, not for the full point cloud.
- `ROI.get_z_from_pcd()` should be able to use the shared index query without constructing temporary point clouds.

## Internal Storage Model

Keep the current offset-based storage model, but make it more disciplined.

| Internal Field | Meaning |
|---|---|
| `_points` | Local XYZ coordinates after subtracting `_offset`. |
| `_offset` | Absolute XYZ offset. |
| `points` | Public absolute XYZ array computed from `_points + _offset`. |
| `_tree` | Internal XY KDTree cache. |

Offset remains necessary because many point clouds use projected CRS coordinates such as EPSG:32654, where X/Y values are large. Internal operations should not accidentally convert offset-aware local storage into repeated full absolute-coordinate copies.

## CRS Conversion Design

The whole EasyIDP refactor should use this naming convention:

| Method | Semantics |
|---|---|
| `to_crs(target_crs)` | Return a new object of the same public type without modifying the source. |
| `change_crs(target_crs)` | Mutate the current object in place. |

For `PointCloud` specifically:

- `change_crs()` keeps the existing in-place API.
- `to_crs()` creates a copy and calls the same conversion logic on that copy.
- Conversion must use `pyproj.Transformer.from_crs(self.crs, target_crs, always_xy=True)`.
- Conversion must transform absolute coordinates, then automatically recompute offset for the target coordinates.
- Recomputing offset is the default because keeping a projected CRS offset after converting to another CRS can produce unstable or confusing internal coordinates.
- After conversion, clear `_tree` and refresh display/cache state.
- Preserve `colors` and file-independent metadata.
- Preserve `normals` for compatibility, but document that normal vectors are not rotated by CRS conversion.
- Raise a clear error when `self.crs is None`; no silent conversion is allowed.

## IO Design

### Public IO

| Function | Input | Output | Notes |
|---|---|---|---|
| `read_point_cloud(path, format=None, offset=None)` | path-like, optional format | `PointCloud` | Prefer explicit `format`; otherwise detect by source file suffix. |
| `write_point_cloud(target, pcd, format=None)` | path-like target, `PointCloud`, optional format | `None` | Prefer explicit `format`; otherwise detect by target file suffix. |

The new IO API should make format selection explicit and deterministic:

- `format` accepts canonical values such as `"ply"`, `"las"`, and `"laz"`, case-insensitively.
- If `format` is omitted, the reader uses the source filename suffix and the writer uses the target filename suffix.
- If both `format` and suffix are provided for writing and they mismatch, emit a mismatch warning and append the canonical format suffix to the target path. For example, `target="xxx.las", format="ply"` writes `xxx.las.ply`.
- If neither `format` nor a usable suffix is available, raise a clear error instead of falling back to stale object state such as `self.file_ext`.

### Legacy IO Names

These names remain temporarily for v2.1 compatibility, live in `compat.py`, route through the new IO layer, and emit `FutureWarning`:

- `read_ply()`
- `read_las()`
- `read_laz()`
- `write_ply()`
- `write_las()`
- `write_laz()`

Legacy standalone IO functions should emit `FutureWarning` and point users to the new API:

```python
read_point_cloud(path, format="ply")
write_point_cloud(target, pcd, format="laz")
```

Class-level `PointCloud.read_point_cloud()` and `PointCloud.write_point_cloud()` may remain compatibility methods in `compat.py`; they also emit `FutureWarning`. The preferred user path is constructor/new reader for loading and `save()`/new writer for writing.

IO cleanup should also cover known edge cases:

- Do not silently infer an output format from stale `file_ext` when the user provides an ambiguous output path.
- Raise an error immediately when a write target has no suffix and no explicit `format`, because there is no reliable target format.
- Emit a mismatch warning and write to an adjusted path when a write target suffix conflicts with explicit `format`, for example `xxx.las` plus `format="ply"` becomes `xxx.las.ply`.
- Read LAS CRS VLR where laspy supports it.
- Write LAS CRS metadata where practical.
- Support files without colors or normals.

## Compatibility Strategy

Use `compat.py` for old API names and old ndarray/list/dict-style workflows that should not drive new core design.

Compatibility methods should:

- Preserve current accepted input types where tests and docs rely on them.
- Preserve current return types, including `crop_point_cloud()` returning `None` on empty crop.
- Delegate to `geometry.py` and `PointCloud.select_by_index()` instead of keeping duplicate algorithms.
- Emit `FutureWarning` for every function in `compat.py`, including legacy crop methods and legacy standalone IO functions.
- Make the warning text direct: the compatibility API may be deprecated in a future release, and users should migrate to the named new API.

## Error Guidance For New API

New `crop()` should fail fast on unsupported inputs and tell users the replacement.

Examples of desired error guidance:

- `np.ndarray` input: use `shapely.Polygon(coords)` or the compatibility method `crop_point_cloud(coords)`.
- `list` input: use `shapely.Polygon(coords)` or the compatibility method.
- bbox tuple input: use `shapely.box(minx, miny, maxx, maxy)`.
- dict input: use `easyidp.ROI` or call `crop()` per geometry.

## Covered Current EasyIDP Needs

| Current Need | Coverage |
|---|---|
| `idp.PointCloud(path)` loads PLY/LAS/LAZ | Kept. |
| Manual Pix4D offset handling | Kept through `offset`. |
| Large projected CRS coordinates | Kept through offset-based storage. |
| `points`, `colors`, `normals` access | Kept. |
| CRS assignment and no-CRS point clouds | Kept; `crs=None` remains valid. |
| In-place CRS conversion | Kept through `change_crs()`, now with automatic offset recomputation. |
| Return-a-copy CRS conversion | Added through `to_crs()`. |
| Save point cloud to PLY/LAS/LAZ | Kept through `save()` and IO layer. |
| Legacy standalone point-cloud IO functions | Kept in `compat.py` with `FutureWarning`, routed through the new IO API. |
| Crop one polygon with old ndarray API | Kept through `crop_point_cloud()` and `crop_polygon()` compatibility wrappers. |
| Crop one polygon with new geometry API | Added through `crop(Polygon(...))`. |
| Crop ROI plots | Kept through `crop_rois()` and added through `crop(ROI)`. |
| ROI height extraction from PCD | Uses shared geometry-index query without temporary crops. |
| Reconstruction classes storing `pcd` | Continue using `idp.PointCloud`. |

## Issue Coverage

| Issue Or Known Problem | Coverage |
|---|---|
| #55 cropped point cloud loses CRS | Solved by making `select_by_index()` copy CRS for all crop outputs. |
| #101 cropped preview loses color info | Solved by making `select_by_index()` copy colors/normals and refresh display state. |
| #102 Open3D conversion | Deferred. Optional `to_open3d()` can be added after the core refactor stabilizes. |
| #93 indoor/no-GPS reconstruction | Preserved because `crs=None` remains valid, and only CRS conversion requires CRS. |
| #78 OpenDroneMap support | Not directly solved, but keeping `PointCloud` stable helps future reconstruction adapters. |
| #121 DEM/CHM/PCD higher-level processing | Not directly solved, but clearer crop and IO boundaries make later processing modules easier. |
| Duplicate point-cloud crop algorithms | Solved by making old and new crop APIs share the same KDTree query. |
| `PointCloud.points` hot-path allocation | Mitigated by avoiding full `points` access during geometry queries. |
| `PointCloud.write_point_cloud("out")` format ambiguity | Should be fixed during IO separation. |
| LAS/PLY files without color data | Should be fixed during IO separation with explicit tests. |
| LAS CRS metadata | Should be improved in `io/las.py`. |

## Point-Cloud Issue Fix Scope

This refactor should treat the following point-cloud issues and known defects as first-class acceptance criteria, not incidental cleanup.

| Area | Required Fix | Implementation Hook | Test Coverage |
|---|---|---|---|
| Cropped point cloud loses CRS (#55) | Every crop and selection result preserves `crs`. | `PointCloud.select_by_index()` copies `crs`. | Crop by polygon, ROI crop, and inverted selection all assert CRS equality. |
| Cropped preview loses color info (#101) | Crop and selection results preserve `colors`, `normals`, and refreshed display state. | `PointCloud.select_by_index()` copies optional arrays and updates print/cache state once. | Crop colored and no-color fixtures; assert colors are preserved only when present. |
| Indoor/no-GPS reconstruction (#93) | `crs=None` remains valid for load, crop, save, and selection workflows. | Only `to_crs()` and `change_crs()` require a defined source CRS. | Construct no-CRS point clouds and verify crop/save paths do not require CRS. |
| Open3D conversion request (#102) | Do not add Open3D as a core dependency. | Defer optional `to_open3d()` until core API is stable. | No test in this refactor beyond verifying core imports do not require Open3D. |
| OpenDroneMap support (#78) | Keep reconstruction code depending on stable `PointCloud`, not IO internals. | `Pix4D`, `Metashape`, and future ODM adapters use `PointCloud` and public IO only. | Existing reconstruction PCD assignment tests continue to pass. |
| DEM/CHM/PCD higher-level processing (#121) | Do not implement new processing algorithms in this refactor. | Keep crop and IO boundaries clean enough for later modules. | No direct feature tests; avoid adding downsampling/CHM scope. |
| Duplicate crop engines | Remove independent bbox-only crop implementation. | `crop()`, `crop_point_cloud()`, `crop_polygon()`, `crop_rois()`, and `ROI.crop(pcd)` share geometry index query. | Compare old crop outputs against new shared query results. |
| `points` hot-path allocation | Avoid repeated full `_points + _offset` allocation during spatial queries. | `geometry.py` computes absolute XY only for candidate indices. | Add a regression/performance-oriented unit test by monkeypatching or structuring helper calls where practical. |
| Ambiguous write format | No suffix plus no `format` raises immediately. | `write_point_cloud(target, pcd, format=None)` validates target format before writing. | Assert `write_point_cloud("out", pcd)` raises a clear error. |
| Explicit format and suffix mismatch | Warn and write to adjusted target, e.g. `xxx.las` + `format="ply"` -> `xxx.las.ply`. | IO dispatcher normalizes target path before backend write. | Assert warning text and adjusted file path. |
| No-color PLY/LAS/LAZ | Reading and writing should not assume RGB fields are present. | `io/ply.py` and `io/las.py` treat colors as optional. | Fixtures or synthetic files without colors round-trip without crashing. |
| LAS CRS metadata | Read/write CRS metadata where laspy supports it. | `io/las.py` uses laspy CRS VLR helpers where available. | LAS fixture with CRS or synthetic write/read test asserts CRS preservation. |

Out of scope for this refactor:

- Open3D-backed IO or processing.
- Chunked streaming crop for huge LAS/LAZ files.
- Downsampling, denoising, normal estimation, clustering, DEM, CHM, or other point-cloud analysis algorithms.
- Public `PointCloudData`, public spatial-index class, or public IO backend registry.

## Deferred Decisions

- Public `PointCloudData` class.
- Public point-cloud index class.
- Public IO backend registry.
- Direct GeoPandas `GeoDataFrame` input.
- `crop(..., explode=True)` for `MultiPolygon` parts.
- `crop(..., return_type=...)` or `format=...`.
- Open3D as a core dependency.
- Streaming/chunked crop for very large LAS/LAZ files.
- Downsampling, outlier filtering, normal estimation, clustering, CHM derivation, or other point-cloud processing algorithms.

## Documentation Update Scope

Documentation updates are part of the refactor, because the new API keeps legacy methods available while changing what users should prefer.

### User-Facing API Docs

Update these docs to make the preferred workflows clear:

| Document | Required Update |
|---|---|
| `docs/python_api/pointcloud.rst` | Present `PointCloud`, `read_point_cloud()`, `write_point_cloud()`, `crop()`, `to_crs()`, and `save()` as the main API. Move legacy standalone IO names out of the main summary. |
| `docs/python_api/autodoc/easyidp.pointcloud.PointCloud.rst` | Add `crop()`, `select_by_index()`, `to_crs()`, and `is_empty()` to the main method list. Keep legacy methods documented as compatibility APIs with `FutureWarning`. |
| `docs/index.rst` | Keep high-level ROI point-cloud crop examples working, but prefer `roi.crop(pcd)` and `pcd.crop(roi)` where appropriate. |
| ROI docs | Document that `ROI.crop(pcd)` is a thin wrapper around the point-cloud crop core and equivalent to `PointCloud.crop(ROI)`. |

### Advanced API Docs

`easyidp.pointcloud.geometry` remains importable but should be documented as Advanced API only.

Required docs behavior:

- Add an `Advanced API` section to the point-cloud docs if one does not already exist.
- List `easyidp.pointcloud.geometry` helpers under Advanced API without adding them to the default public autosummary.
- If explicit autodoc stubs are created for geometry helpers, mark them with `:orphan:` so they are linkable but not shown in the left sidebar toctree.
- State that geometry helpers are intended for internal integration and advanced users who need index-level point selection.

### Compatibility Documentation

Every `compat.py` function should have docs that mention the new API replacement:

| Compatibility API | Preferred Replacement |
|---|---|
| `crop_point_cloud(polygon_xy)` | `crop(shapely.Polygon(polygon_xy))` |
| `crop_polygon(polygon_xy)` | Use geometry index helpers for internal code, or `crop(...).points` for user code. |
| `crop_rois(roi, save_folder=None)` | `crop(roi)` or `ROI.crop(pcd)`. |
| `read_ply(path)` | `read_point_cloud(path, format="ply")`. |
| `read_las(path)` | `read_point_cloud(path, format="las")`. |
| `read_laz(path)` | `read_point_cloud(path, format="laz")`. |
| `write_ply(path, data)` | `write_point_cloud(path, pcd, format="ply")`. |
| `write_las(path, data)` | `write_point_cloud(path, pcd, format="las")`. |
| `write_laz(path, data)` | `write_point_cloud(path, pcd, format="laz")`. |
| `PointCloud.read_point_cloud(path)` | `idp.PointCloud(path)` or `read_point_cloud(path)`. |
| `PointCloud.write_point_cloud(path)` | `pcd.save(path)` or `write_point_cloud(path, pcd)`. |

Documentation acceptance criteria:

- New examples should use Shapely geometry for new `PointCloud.crop()` polygon examples.
- Old examples can remain only when they are explicitly labeled compatibility examples.
- The docs must explain that all compatibility APIs emit `FutureWarning` in v2.1.
- IO examples must show both suffix-based dispatch and explicit `format` dispatch.
- IO docs must explain the mismatch rule: `target="xxx.las", format="ply"` warns and writes `xxx.las.ply`.
- CRS docs must explain `to_crs()` vs `change_crs()` and automatic PointCloud offset recomputation.

## Proposed Implementation Phases

### Phase 1: Package Split Without Public API Breakage

- Create `src/easyidp/pointcloud/` package layout.
- Move the current public `PointCloud` implementation into `core.py` with the same import path behavior through `pointcloud/__init__.py`.
- Keep `easyidp.PointCloud` and `easyidp.pointcloud.PointCloud` working.
- Add tests that import all old public point-cloud names.

### Phase 2: Selection And Crop Engine

- Implement `PointCloud.select_by_index(indices, invert=False)`.
- Move geometry index helpers into `geometry.py`.
- Make all point selection use KDTree candidate queries plus exact polygon containment.
- Avoid full `points` allocation in crop hot paths.
- Refactor `crop_point_cloud()`, `crop_polygon()`, and `crop_rois()` in `compat.py` while preserving their current inputs and outputs.
- Add `FutureWarning` tests for the legacy crop wrappers and verify the warning names the new `crop()` API.
- Add new `PointCloud.crop()` for Shapely `Polygon`, `MultiPolygon`, and EasyIDP ROI.

### Phase 3: CRS Conversion Semantics

- Implement `PointCloud.to_crs(target_crs)` as return-a-new-object conversion.
- Refactor `PointCloud.change_crs(target_crs)` as in-place conversion that transforms absolute coordinates and recomputes offset automatically.
- Preserve colors and normals, and document normal-vector behavior.
- Add tests for projected-to-geographic and geographic-to-projected conversions, same-CRS no-op behavior, `crs=None` errors, offset recomputation, and tree invalidation.

### Phase 4: IO Separation

- Move LAS/LAZ logic into `io/las.py`.
- Move PLY logic into `io/ply.py`.
- Add `read_point_cloud(path, format=None, offset=None)` and `write_point_cloud(target, pcd, format=None)` dispatchers.
- Dispatch by explicit `format` when provided; otherwise dispatch by source or target filename suffix.
- Move legacy `read_las()`, `read_laz()`, `read_ply()`, `write_las()`, `write_laz()`, and `write_ply()` into `compat.py` and route them through the new modules.
- Add `FutureWarning` tests for the legacy standalone IO functions.
- Fix ambiguous output suffix behavior.
- Add tests for no-color files and CRS metadata where fixtures allow it.

### Phase 5: Cross-Module Integration And Docs

- Update `ROI.get_z_from_pcd()` to use the shared crop-index query directly.
- Keep both `PointCloud.crop(ROI)` and `ROI.crop(pcd)`. Implement `ROI.crop(pcd)` as a thin wrapper around the point-cloud crop core.
- Keep `Pix4D.load_pcd()` and reconstruction `pcd` assignment working through `idp.PointCloud`.
- Update API docs to show `crop()` and `to_crs()` as the preferred APIs while documenting legacy compatibility methods.
- Document `easyidp.pointcloud.geometry` as Advanced API rather than default public API.
- Document that every `compat.py` function emits `FutureWarning` and may be deprecated in a future release.
- Update point-cloud issue documentation and examples to show which v2.1 fixes are covered by the refactor.

### Phase 6: Issue-Fixing Regression Tests

- Add focused tests that prove each point-cloud issue in `Point-Cloud Issue Fix Scope` is fixed or intentionally deferred.
- Add crop metadata tests for #55 and #101: cropped `PointCloud` objects preserve CRS, colors, normals, offset, and refreshed display/cache state.
- Add no-CRS workflow tests for #93: load or construct no-CRS point clouds, crop them, select by index, and save them without requiring CRS.
- Add import tests for #102 scope control: importing EasyIDP and the point-cloud module must not require Open3D.
- Add reconstruction compatibility tests for #78: existing `Pix4D.load_pcd()` and reconstruction `pcd` assignment still work through `idp.PointCloud`.
- Add crop-engine equivalence tests: `crop()`, `crop_point_cloud()`, `crop_polygon()`, `crop_rois()`, and `ROI.crop(pcd)` select the same points for the same geometry where return types allow comparison.
- Add IO edge-case tests: no suffix plus no `format` raises, suffix/format mismatch warns and writes the adjusted path, no-color files do not crash, and LAS CRS metadata round-trips where fixtures support it.
- Mark deferred issues (#102 optional Open3D conversion and #121 higher-level PCD processing) with explicit non-goal tests or documentation checks rather than partial implementations.
- Run the point-cloud and ROI test subsets after these regression tests, then run the relevant full test group before considering the refactor complete.

## Resolved Review Decisions

- `easyidp.pointcloud.geometry` remains importable, but docs list it under Advanced API rather than the default public API.
- Keep both `PointCloud.crop(ROI)` and `ROI.crop(pcd)`. `ROI.crop(pcd)` should be a simple wrapper over the point-cloud crop core.
- Legacy crop methods start emitting `FutureWarning` in v2.1, together with all other `compat.py` functions.
- A write target without suffix and without explicit `format` raises an error immediately.
- A write target whose suffix conflicts with explicit `format` emits a mismatch warning and writes to an adjusted path such as `xxx.las.ply`.
