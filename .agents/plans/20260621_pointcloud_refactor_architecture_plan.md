# PointCloud Refactor Architecture Plan

> **Scope:** This is an architecture-level plan for discussion and review. It intentionally avoids function-body implementation details. Detailed task steps and tests should be written after this plan is approved.

**Goal:** Refactor `idp.PointCloud` into a clean, lightweight, Open3D-inspired API while keeping EasyIDP's GIS-oriented polygon and ROI workflows.

**Architecture:** Keep `PointCloud` as the only public point-cloud class for v2.1. Split IO and crop geometry helpers out of the class, make `select_by_index()` the core selection primitive, and make `crop()` the single new crop entry point. Legacy API names may remain temporarily as wrappers that call the new API and emit future deprecation warnings.

**Tech Stack:** NumPy, SciPy `cKDTree`, Shapely, pyproj, laspy, plyfile, tqdm, EasyIDP ROI.

---

## Design Principles

- Prefer one clean public class: `easyidp.PointCloud`.
- Avoid public `PointCloudData` and public point-cloud index classes until a second storage backend exists.
- Keep Open3D out of core dependencies; mimic its API style, not its implementation.
- Selection and crop operations return new `PointCloud` objects.
- Conversion is explicit and separate from crop, for example `to_numpy()` and optional future `to_open3d()`.
- New `crop()` should not accept bare `list` or `np.ndarray` polygons. Users should pass Shapely geometry such as `Polygon`, `MultiPolygon`, or `box(...)`.
- Legacy APIs can accept old ndarray polygon inputs through a dedicated compatibility layer.
- Use Shapely as the geometry boundary between ROI/vector data and point-cloud crop logic.
- Preserve `offset` as a first-class part of internal storage for precision with large CRS coordinates.
- Do not optimize for streaming large point clouds in the first refactor, but avoid designs that make streaming impossible later.

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
| `pointcloud/__init__.py` | Export `PointCloud`, `read_point_cloud`, `write_point_cloud`, and temporary legacy IO names. |
| `pointcloud/core.py` | Define `PointCloud`, its data properties, CRS/offset behavior, `select_by_index()`, `crop()`, `save()`, and conversion methods. |
| `pointcloud/geometry.py` | Normalize Shapely crop inputs and query point indices for `Polygon` and `MultiPolygon`. |
| `pointcloud/compat.py` | Hold temporary wrappers for old API names and old ndarray polygon inputs, all with future deprecation warnings. |
| `pointcloud/io/__init__.py` | Route `read_point_cloud()` and `write_point_cloud()` by file extension. |
| `pointcloud/io/las.py` | Read and write LAS/LAZ using laspy, including CRS/offset metadata where practical. |
| `pointcloud/io/ply.py` | Read and write PLY using plyfile. |

## Public API Shape

### Main Class

```python
idp.PointCloud(path=None, offset=None)
```

The class remains the user-facing point-cloud object. It owns point coordinates, optional colors, optional normals, CRS, offset, and internal spatial-index cache.

### Core Properties

| Property | Type | Notes |
|---|---|---|
| `points` | `np.ndarray` or `None` | Absolute XYZ coordinates exposed to users. Internally stored as local coordinates plus offset. |
| `colors` | `np.ndarray` or `None` | Optional RGB values. |
| `normals` | `np.ndarray` or `None` | Optional normal vectors. |
| `crs` | `pyproj.CRS` or `None` | `None` is valid for local or indoor reconstructions. |
| `offset` | `np.ndarray` | Three-element XYZ offset used to preserve precision for large CRS coordinates. |

### Core Methods

| Method | Input | Output | Purpose |
|---|---|---|---|
| `select_by_index(indices, invert=False)` | integer sequence | `PointCloud` | Open3D-style primitive for selection and deletion. Copies points, colors, normals, CRS, and offset. |
| `crop(target, invert=False)` | Shapely geometry or EasyIDP ROI | `PointCloud` or `dict[str, PointCloud]` | Main crop entry point. |
| `save(path)` | path-like | `None` | Save point cloud by extension. |
| `to_numpy(copy=True)` | bool | `np.ndarray` | Return XYZ points only. |
| `has_points()` | none | bool | Existing predicate. |
| `has_colors()` | none | bool | Existing predicate. |
| `has_normals()` | none | bool | Existing predicate. |
| `is_empty()` | none | bool | New Open3D-style convenience predicate. |

## Crop API Rules

### Accepted New Inputs

| Input | Output | Notes |
|---|---|---|
| `shapely.Polygon` | `PointCloud` | Crop by a single polygon along the Z axis. |
| `shapely.MultiPolygon` | `PointCloud` | Crop by the union of all parts and return one logical point cloud. |
| `shapely.box(minx, miny, maxx, maxy)` | `PointCloud` | BBox crop represented as an ordinary Shapely polygon. |
| `easyidp.ROI` | `dict[str, PointCloud]` | Crop each ROI feature and preserve ROI labels. |

### Not Accepted By New `crop()`

| Input | Decision | User Fix |
|---|---|---|
| `np.ndarray` polygon | Not accepted in new API. | Use `shapely.Polygon(coords)`. |
| `list` polygon | Not accepted in new API. | Use `shapely.Polygon(coords)`. |
| `dict[str, polygon]` | Not accepted in new API. | Use `easyidp.ROI` or call `crop()` per geometry. |
| GeoPandas `GeoDataFrame` | Deferred. | Use EasyIDP ROI until ROI/GeoPandas architecture lands. |

### MultiPolygon Policy

- A `MultiPolygon` is treated as one logical geometry by default.
- For ROI or shapefile workflows, a feature's geometry identity should be preserved even if that geometry is multipart.
- Splitting multipart geometries into separate outputs is deferred. Users can explicitly iterate `multipolygon.geoms` if needed.

## Crop Dispatch Model

`PointCloud.crop()` should stay small and dispatch by input type.

```text
crop(target)
  ├─ Shapely Polygon or MultiPolygon -> crop one geometry -> PointCloud
  ├─ EasyIDP ROI -> crop each ROI item -> dict[str, PointCloud]
  └─ unsupported input -> TypeError with migration guidance
```

The shared internal flow is:

```text
geometry input
  -> query selected indices
  -> select_by_index(indices)
  -> new PointCloud
```

This keeps crop logic decoupled from data-copy logic. `select_by_index()` is the single place responsible for copying point attributes and metadata.

## Internal Storage Model

Keep the current offset-based storage model, but make it more disciplined.

| Internal Field | Meaning |
|---|---|
| `_points` | Local XYZ coordinates after subtracting `_offset`. |
| `_offset` | Absolute XYZ offset. |
| `points` | Public absolute XYZ view/value computed from `_points + _offset`. |
| `_tree` | Internal XY KDTree cache. Not public API. |

Offset remains necessary because many point clouds use projected CRS coordinates such as EPSG:32654, where X/Y values are large. Storing local coordinates preserves numerical precision and matches LAS-style offset handling.

## IO Design

### Public IO

| Function | Input | Output | Notes |
|---|---|---|---|
| `read_point_cloud(path, offset=None)` | path-like | `PointCloud` | Auto-detect by `.ply`, `.las`, or `.laz`. |
| `write_point_cloud(path, pcd)` | path-like + `PointCloud` | `None` | Auto-detect output format by suffix. |

### Legacy IO Names

These names can remain temporarily for v2.1 compatibility, but should live in or route through the IO layer:

- `read_ply()`
- `read_las()`
- `read_laz()`
- `write_ply()`
- `write_las()`
- `write_laz()`

## Compatibility Strategy

Use `compat.py` for old API names and old ndarray/list polygon inputs.

Temporary compatibility methods may include:

| Old API | Compatibility Behavior |
|---|---|
| `PointCloud.crop_point_cloud(polygon_array)` | Emit future deprecation warning, convert array to `shapely.Polygon`, call `crop()`. |
| `PointCloud.crop_polygon(polygon_array)` | Emit future deprecation warning, convert array to `shapely.Polygon`, call `crop()`, return `.points`. |
| `PointCloud.crop_rois(roi)` | Emit future deprecation warning, call `crop(roi)`. |
| `PointCloud.write_point_cloud(path)` | Emit future deprecation warning, call `save(path)`. |
| `PointCloud.read_point_cloud(path)` | Emit future deprecation warning, replace current object state with `read_point_cloud(path)`. |

Compatibility code should contain no new business logic. Future cleanup should be possible by removing `compat.py` and the legacy method shims.

## Error Guidance For New API

New `crop()` should fail fast on unsupported inputs and tell users the replacement.

Examples of desired error guidance:

- `np.ndarray` input: use `shapely.Polygon(coords)`.
- `list` input: use `shapely.Polygon(coords)`.
- bbox tuple input: use `shapely.box(minx, miny, maxx, maxy)`.
- dict input: use `easyidp.ROI` or loop over the dict and call `crop()` for each geometry.

## Covered Current EasyIDP Needs

| Current Need | Coverage |
|---|---|
| `idp.PointCloud(path)` loads PLY/LAS/LAZ | Kept. |
| Manual Pix4D offset handling | Kept through `offset`. |
| Large projected CRS coordinates | Kept through offset-based storage. |
| `points`, `colors`, `normals` access | Kept. |
| CRS assignment and no-CRS point clouds | Kept; `crs=None` remains valid. |
| Save point cloud to PLY/LAS/LAZ | Kept through `save()` and IO layer. |
| Crop one polygon | New `crop(Polygon(...))`. |
| Crop ROI plots | New `crop(ROI)`. |
| ROI height extraction from PCD | Can use the shared geometry-index query without constructing temporary crops. |
| Reconstruction classes storing `pcd` | Continue using `idp.PointCloud`. |

## Issue Coverage

| Issue | Coverage |
|---|---|
| #55 cropped point cloud loses CRS | `select_by_index()` copies CRS for all crop outputs. |
| #101 cropped preview loses color info | `select_by_index()` copies colors/normals and refreshes preview. |
| #102 Open3D conversion | Deferred to optional conversion method; not a core dependency. |
| #93 indoor/no-GPS reconstruction | `crs=None` remains valid; only CRS conversion requires CRS. |
| #78 OpenDroneMap support | Reconstruction can keep depending on `idp.PointCloud` without knowing IO internals. |
| #121 DEM/CHM/PCD higher-level processing | Deferred; cleaner point-cloud boundaries make it easier later. |

## Deferred Decisions

- Public `PointCloudData` class.
- Public point-cloud index class.
- Direct GeoPandas `GeoDataFrame` input.
- `crop(..., explode=True)` for `MultiPolygon` parts.
- `crop(..., return_type=...)` or `format=...`.
- `to_dict()`.
- Open3D as a core dependency.
- Streaming/chunked crop for very large LAS/LAZ files.
- Downsampling, outlier filtering, normal estimation, clustering, or CHM derivation.

## Proposed Implementation Phases

### Phase 1: API And Behavior Stabilization

- Introduce package layout.
- Implement `PointCloud.select_by_index()`.
- Implement `PointCloud.crop()` for Shapely `Polygon`, `MultiPolygon`, and EasyIDP ROI.
- Preserve offset, CRS, colors, and normals through selection and crop.
- Add clear TypeErrors for unsupported new API inputs.
- Add compatibility wrappers with future deprecation warnings.

### Phase 2: IO Separation

- Move LAS/LAZ logic into `io/las.py`.
- Move PLY logic into `io/ply.py`.
- Add `read_point_cloud()` and `write_point_cloud()` dispatchers.
- Keep legacy `read_las/read_laz/read_ply/write_*` names routed through the new modules.

### Phase 3: Cross-Module Integration

- Update ROI point-cloud height extraction to use shared crop-index logic where useful.
- Keep `Pix4D.load_pcd()` and reconstruction `pcd` assignment working through `idp.PointCloud`.
- Update docs to make Shapely inputs the standard crop examples.

### Phase 4: Cleanup And Future Extensions

- Review whether legacy wrappers should remain or be removed before final v2.1 release.
- Consider optional `to_open3d()` only after the core refactor is stable.
- Consider streaming LAS/LAZ crop only after in-memory API behavior is locked down.

## Review Questions

- Should `geometry.py` be public or renamed `_geometry.py` to signal internal use?
- Should `PointCloud.crop(ROI)` remain in pointcloud, or should `ROI.crop(pcd)` be the only batch-crop API after ROI refactor?
- Should v2.1 emit `FutureWarning` for legacy methods immediately, or only document them as legacy first?
- Should `read_point_cloud()` return `PointCloud` only, or should lower-level IO functions keep returning raw arrays for advanced users?
