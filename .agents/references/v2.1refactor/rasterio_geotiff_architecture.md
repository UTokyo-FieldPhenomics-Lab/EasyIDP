# Rasterio And GeoTIFF Architecture Notes

## Current Model

- `GeoTiff` is a custom class holding `file_path`, `header`, `_imarray`, `_mask`, `_mask_polygon`, and affine state.
- It handles file IO, metadata, coordinate conversion, crop, mask generation, affine rotated storage, statistics, and save logic in one large module.
- Rasterio is already the real backend for reading, writing, masks, transforms, and sampling.

## Problem

- Inheriting Rasterio classes is not a good fit because `DatasetReader`/`DatasetWriter` are created by `rasterio.open()` and are file-handle-like objects.
- `header` duplicates rasterio `profile`, `transform`, width, height, CRS, nodata, and other metadata.
- Many operations reopen the same file and lack window-first APIs for large rasters.
- Mask handling and save behavior need a clearer single policy.

## Recommended Direction

- Use Rasterio by composition, not inheritance.
- Treat rasterio `profile`, `transform`, `crs`, `window`, and `mask` as the core backend objects.
- Make `profile` the single metadata truth. Keep old `header` only as a migration shim if needed.
- Add explicit windowed read/crop APIs before optimizing large data workflows.

## Target Module Shape

```text
easyidp.raster
  dataset.py     # RasterDataset path/profile/read/window metadata
  cropper.py     # polygon/window/rotated crop
  mask.py        # mask source and resolution policy
  writer.py      # GeoTIFF writing and profile update
  affine.py      # rotated affine crop utilities
```

## Target Data Flow

```text
RasterDataset(path)
  -> read(window=None)
  -> crop(geometry, mode='window'|'mask'|'rotated')
  -> RasterTile(data, profile, mask, geometry)
  -> writer.save(tile, path, apply_mask=True, use_affine=True)
```

## Mask Policy

- Support mask sources explicitly: polygon geometry, boolean array, alpha band, nodata, GDAL internal mask.
- Resolve masks through one `resolve_mask()` path.
- Keep polygon geometry as high-level source when possible, and rasterize only at read/save time.

## Rotated Affine Crop

- Preserve the current EasyIDP feature: rotated rectangle ROI can be saved with affine transform to avoid external empty pixels.
- Keep this as a first-class raster crop mode, not an ad-hoc save option.
- Prefer Rasterio/GDAL warp utilities where they are clearer; keep scipy-based resampling only where needed.

## Migration Notes

- Preserve common old calls such as `GeoTiff(path)`, `crop_polygon()`, `crop_rois()`, `point_query()`, and `save()` during transition if practical.
- New internal code should depend on `RasterDataset`/`RasterTile` style objects instead of `header` dicts.
- `cvtools` should remain a pixel-space helper for raw image arrays, not a geospatial raster backend.

## References:

Detailed subagent investigation can refer to: `.agents/references/v2.1refactor/subagents/investigate_raster_architecture.md`
