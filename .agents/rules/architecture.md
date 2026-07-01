# EasyIDP v2.1 Architecture Rules

Use this file as the concise architecture rule for v2.1 refactoring. Detailed module notes live in `.agents/references/v2.1refactor/` and should be read when working on the related module.

## Direction

- v2.1 prioritizes a cleaner new architecture over strict old API compatibility.
- Break part of the old API when it removes confusing boundaries or long-term maintenance burden.
- Keep compatibility shims only when they protect common user workflows during migration.
- Prefer composition over inheriting third-party library internals.
- Make every public workflow usable without import-time network access, hidden global state, or interactive prompts.

## Functional Domains

- ROI and vector data: boundary loading, subplot/subgrid generation, attributes, CRS, preview, vector export.
- Raster data: GeoTIFF reading, windowed crop, mask handling, affine rotated crop, save/export.
- Point cloud data: lightweight point cloud IO, CRS, spatial index, crop, save/export.
- Reconstruction projects: parser adapters for photogrammetry software, projection engine, result filtering, visualization, raw ROI export.
- Data and tooling: demo dataset download, validation, docs, tests, future MCP and skills interfaces.
- Configuration: JSON-backed user settings for data directory, logging level, startup banner, and future package-wide preferences.
- Legacy collection utilities: avoid carrying `Container` into v2.1 unless a short compatibility shim is required.

## Module Boundaries

- `geometry` / ROI layer owns vector features, attributes, CRS, subplot generation, and preview inputs.
- `raster` layer owns raster profiles, windows, masks, affine transforms, and GeoTIFF writing.
- `pointcloud` layer owns point cloud data, IO backends, spatial indexes, and croppers.
- `reconstruction` layer owns project parsers, adapters, camera models, coordinate transforms, and projection.
- `config` layer owns package-wide settings and persistence. It must use a JSON config file, avoid network access, and persist only through explicit user calls such as `idp.config.set(...)` or `idp.config.reset()`.
- `data` layer owns dataset manifests and downloaders. It must not run downloads at import time, and should read the default dataset root from `idp.config` rather than storing its own long-lived configuration.
- `visualization` layer owns plotting. Core processing modules should not depend on pyplot global state.
- `structures` should not define broad public containers unless they have a clear, typed role after ROI and reconstruction are refactored.

## Third-Party Library Policy

- ROI should use GeoPandas/Shapely by composition. Do not inherit `GeoDataFrame` unless there is a concrete need.
- GeoTIFF should use Rasterio by composition. Do not inherit `DatasetReader` or `DatasetWriter`.
- PointCloud should remain lightweight: compose `laspy`, `plyfile`, `scipy.spatial`, and optional extras. Do not make Open3D a core dependency.
- Reconstruction parsers may use targeted third-party helpers, but all parser outputs must be normalized into EasyIDP data models.
- Dataset downloads should be explicit and non-interactive; do not perform network checks or package installation at import time.

## API Shape

- Prefer explicit data objects over parallel dict/list state.
- Prefer pure operations returning new objects unless an in-place method is clearly named with `_inplace`.
- For CRS conversion APIs, use `to_crs(...)` for return-a-new-object workflows and `change_crs(...)` for in-place mutation on the current object. Keep this distinction consistent across ROI, GeoTiff, PointCloud, and future spatial data classes.
- Use stable JSON-serializable result objects for future MCP/tools.
- Avoid returning large arrays from future tool-facing APIs; return paths, metadata, summaries, and warnings.
- Public configuration should flow through `idp.config`; prefer explicit `get(...)`, `set(...)`, and `reset()` calls over hidden environment-variable behavior or import-time mutation.

## CRS Conversion Policy

- `to_crs(target_crs)` should return a new object of the same public type without modifying the source object.
- `change_crs(target_crs)` should mutate the object in place and keep its existing public API behavior unless an explicit migration plan says otherwise.
- CRS transforms must use `pyproj.Transformer.from_crs(..., always_xy=True)` unless a module documents a different axis-order reason.
- PointCloud CRS conversion must transform absolute coordinates, then recompute offset automatically so internal local coordinates remain numerically stable after projection changes.
- CRS conversion should preserve non-coordinate attributes such as ROI labels, point colors, normals, masks, and metadata when those values remain semantically valid.

## Test Layout Policy

- New or refactored tests should be grouped by module in directories such as `tests/test_config/` and `tests/test_data/`, not accumulated into one large `tests/test_<module>.py` file.
- Inside each module test directory, split tests by behavior or responsibility, for example `test_api.py`, `test_persistence.py`, `test_manifest.py`, `test_repr.py`, and `test_downloader.py`.
- Keep `tests/conftest.py` for shared fixtures only. Avoid module-level heavy fixture construction when a local fixture can keep tests isolated.
- When refactoring an existing large module test file, migrate the touched module first and avoid unrelated churn in other test modules.

## Documentation API Layout Policy

Documentation layout rules are maintained separately in
`.agents/rules/documentation_api_layout.md`. Read that file before creating or
refactoring API docs, autosummary templates, class pages, or Advanced API pages.

## Configuration Policy

- Add `idp.config` as the single public configuration entry point for package-wide preferences.
- Store persistent user configuration as JSON, not TOML/INI/env vars, to avoid new dependencies and keep machine-readable settings simple.
- Configuration reads may happen at import time, but must be local, fast, and side-effect-light.
- `idp.config.get(key)` should read the latest JSON state before returning a value so manual config-file edits are visible in the current session.
- `idp.config.set(...)` and `idp.config.reset()` should persist JSON immediately using an atomic write. Do not require a separate public `save()` call for ordinary users.
- The initial configuration scope includes `data_dir`, `log_level`, and `show_banner`; future settings should be added here rather than as scattered module globals.
- `data_dir` controls the default root for dataset archives, temporary extraction directories, manifests, and final extracted datasets.
- Changing `data_dir` must not automatically move or copy existing data caches. Cache migration, if ever added, must be an explicit data-layer operation with a dry-run path.
- Dataset constructors may accept an explicit `cache_root` to override `idp.config.data_dir` for that object only.
- Do not use environment variables as the primary configuration API for data paths in v2.1.

## Recommended v2.1 Package Shape

```text
easyidp.geometry
  ROIFeature
  ROICollection
  readers/writers
  subplot generation

easyidp.raster
  RasterDataset
  RasterCropper
  RasterMask
  RasterWriter

easyidp.pointcloud
  PointCloud as the only public class for v2.1
  core.py for data, CRS, offset, selection, crop, save
  geometry.py for KDTree-backed crop index queries
  compat.py for legacy point-cloud APIs
  io/las.py and io/ply.py for readers/writers

easyidp.reconstruction
  ReconstructionProject
  ReconstructionBundle
  parsers
  adapters
  ProjectionEngine
  filters

easyidp.config
  EasyIDPConfig
  get/set/reset
  immediate JSON persistence

easyidp.data
  DatasetSpec as json file
  Dataset class with manifest and downloader

legacy/internal
  Container compatibility only if needed during migration
```

## Detailed References

- ROI and subplot architecture: `.agents/references/v2.1refactor/roi_geopandas_architecture.md`
- GeoTIFF and Rasterio architecture: `.agents/references/v2.1refactor/rasterio_geotiff_architecture.md`
- Point cloud lightweight architecture: `.agents/references/v2.1refactor/pointcloud_architecture.md`
- Reconstruction adapter architecture: `.agents/references/v2.1refactor/reconstruction_adapters.md`
- Data and dataset architecture: `.agents/references/v2.1refactor/data_dataset_architecture.md`
- Container and collection architecture: `.agents/references/v2.1refactor/container_collection_architecture.md`
- Docs and test consistentency subagent analyze: `.agents/references/v2.1refactor/subagents/investigate_package_docs_tests_consistency.md`
