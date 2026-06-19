# ROI And GeoPandas Architecture Notes

## Current Model

- `Container` stores `id_item: dict[int, item]` and `item_label: dict[str, int]` to support int and str lookup.
- `ROI(Container)` stores polygon coordinates as raw `numpy.ndarray` values.
- ROI attributes are tracked by parallel lists/dicts such as `_attrs` and `_field_schema`.
- Subplot metadata is currently an extra structure rather than first-class tabular data.
- `shp.py` uses pyshp and `jsonfile.py` uses geojson/json, so vector IO is split across custom readers.

## Problem

- ROI is effectively a partial GeoDataFrame implemented by hand.
- Attributes and geometries can drift because they are stored separately.
- Multipart polygons, holes, geometry validation, CRS-aware IO, and rich vector formats are better handled by GeoPandas/Shapely.
- `ROI` also mixes IO, geometry storage, CRS conversion, crop dispatch, height sampling, and back projection handoff.

## Recommended Direction

- Use `GeoDataFrame` by composition: `ROICollection` owns a private `_gdf`.
- Do not initially inherit `GeoDataFrame`; pandas subclassing makes API semantics and metadata propagation harder.
- Expose `.gdf` for advanced users and internal modules.
- Keep `roi['plot_id']` and `roi[0]` as temporary compatibility shims when useful, but make `loc`/`iloc` style access the clean direction.

## Target Data Model

```python
@dataclass
class ROIFeature:
    name: str
    geometry: shapely.Geometry
    attrs: dict[str, Any]


class ROICollection:
    _gdf: geopandas.GeoDataFrame

    @property
    def crs(self):
        return self._gdf.crs

    @property
    def gdf(self):
        return self._gdf
```

## Subplot Generation

- Treat subplot/subgrid output as another `ROICollection` / `GeoDataFrame`.
- Store `row`, `col`, `status`, `boundary_id`, `subplot_id`, and rule metadata as columns.
- Use Shapely for rotation, minimum rotated rectangle, clipping, and intersection tests.
- Reuse ideas from `/home/crest/Documents/Github/EasyPlantFieldID/src/utils/subplot_generate` where practical.
- Preview should read from the GeoDataFrame directly, usually by plotting `status` and geometry.

## IO Policy

- Prefer `geopandas.read_file()` and `GeoDataFrame.to_file()` for shp, geojson, gpkg, and future vector formats.
- Keep pyshp only if a specific schema-control case cannot be handled by GeoPandas.
- GeoJSON and shapefile readers should normalize to the same internal model.

## Migration Notes

- v2.1 may break parts of the old `Container` behavior if it clarifies the API.
- Preserve common workflows: load ROI, iterate ROI, crop raster/pointcloud, back-project ROI.
- Deprecate direct access to `_attrs`, `_field_schema`, `id_item`, and `item_label`.
- Replace `save_shp()` with general `to_file()` style behavior.

## References:

Detailed subagent investigation can refer to: `.agents/references/v2.1refactor/subagents/investigate_ROI_architecture.md`
