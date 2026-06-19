# Container And Collection Architecture Notes

## Current Model

- `Container(dict)` stores real data in `id_item: dict[int, Any]` and `item_label: dict[str, int]`, while the inherited dict stays mostly unused.
- It supports int lookup, str label lookup, suffix matching, slicing, pretty printing, and deletion with index renumbering.
- `ROI` inherits `Container`.
- `Recons.sensors` and `Recons.photos` compose `Container` for `Sensor` and `Photo` lookup.
- `ProjectPool` inherits `Container` but is not implemented.

## Problem

- Inheriting `dict` is misleading because normal dict methods do not reflect `id_item` and `item_label` correctly.
- `id_item` and `item_label` are parallel mutable state and can drift.
- `__delitem__` renumbers integer keys and is O(n).
- `copy()` uses deep copy and can be expensive.
- Pretty printing mutates global NumPy print options.
- Suffix matching for photos is an implicit hack rather than explicit label normalization.
- The class is untyped and cannot guarantee stored values are ROI geometries, sensors, or photos.

## Recommended Direction

- Do not carry `Container` forward as a core v2.1 abstraction.
- ROI should move to a GeoDataFrame-backed `ROICollection`.
- Reconstruction should move to explicit dataclasses, lists, and lookup maps in `ReconstructionBundle`.
- `ProjectPool` should be removed unless reintroduced later with a clear project-collection design.

## ROI Replacement

```text
Container old behavior
  roi['plot'] -> ndarray
  roi[0] -> ndarray
  roi.keys() -> labels

v2.1 direction
  roi.gdf.loc['plot']
  roi.gdf.iloc[0]
  roi.gdf.index
```

- Temporary compatibility shims may keep `roi['plot']` or `roi[0]` if they protect common workflows.
- Direct use of `id_item`, `item_label`, `_attrs`, and `_field_schema` should be deprecated or removed.

## Reconstruction Replacement

```python
@dataclass
class ReconstructionBundle:
    sensors: list[Sensor]
    photos: list[Photo]
    sensors_by_id: dict[str, Sensor]
    photos_by_id: dict[str, Photo]
    photos_by_label: dict[str, Photo]

    def get_photo(self, key: str) -> Photo:
        ...
```

- Use explicit `get_photo_by_id`, `get_photo_by_label`, or one normalized `get_photo` method.
- Normalize photo labels and file suffixes during parsing/adaptation rather than inside collection lookup.
- Iteration should be over `list[Photo]`, not over a custom dict subclass.

## Strategy Options

- Full removal: recommended for v2.1 because ROI and reconstruction both have cleaner replacement models.
- Internal-only wrapper: acceptable only as a short migration shim, not a public API.
- New `LabelledCollection[T]`: possible but avoid unless repeated non-GeoDataFrame labeled collections remain after reconstruction refactor.

## Migration Notes

- Remove `idp.Container` from the preferred public API.
- Rewrite tests around `ROICollection` and `ReconstructionBundle` behavior.
- Drop `ProjectPool` dead code or redesign it later as an explicit collection of `ReconstructionProject` objects.
- If a compatibility wrapper is kept, it should not inherit `dict` and should not mutate global print options.

## References:

Detailed subagent investigation can refer to: `.agents/references/v2.1refactor/subagents/investigate_data_dataset.md`
