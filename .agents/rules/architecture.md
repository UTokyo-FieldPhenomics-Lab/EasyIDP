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
- Legacy collection utilities: avoid carrying `Container` into v2.1 unless a short compatibility shim is required.

## Module Boundaries

- `geometry` / ROI layer owns vector features, attributes, CRS, subplot generation, and preview inputs.
- `raster` layer owns raster profiles, windows, masks, affine transforms, and GeoTIFF writing.
- `pointcloud` layer owns point cloud data, IO backends, spatial indexes, and croppers.
- `reconstruction` layer owns project parsers, adapters, camera models, coordinate transforms, and projection.
- `data` layer owns dataset manifests and downloaders. It must not run downloads at import time.
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
- Use stable JSON-serializable result objects for future MCP/tools.
- Avoid returning large arrays from future tool-facing APIs; return paths, metadata, summaries, and warnings.

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
  PointCloudData
  readers/writers
  SpatialIndex
  PointCloudCropper

easyidp.reconstruction
  ReconstructionProject
  ReconstructionBundle
  parsers
  adapters
  ProjectionEngine
  filters

easyidp.data
  DatasetSpec
  DatasetRegistry
  Downloaders

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
