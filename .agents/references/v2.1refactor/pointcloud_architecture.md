# Point Cloud Architecture Notes

## Current Model

- `PointCloud` owns data, CRS, offset, IO, spatial index, crop logic, and display logic.
- Data is stored as `_points` plus `_offset`, with optional `colors` and `normals`.
- IO is implemented with `plyfile` for PLY and `laspy` for LAS/LAZ.
- Spatial index is `scipy.spatial.cKDTree` over XY coordinates.
- There are duplicate crop paths: one returns points, another returns a `PointCloud`.

## Problem

- The class is too large and mixes storage, IO, algorithms, and presentation.
- Accessing `points` can allocate full arrays because it adds offset every time.
- Open3D would be a natural rich point cloud type, but it is too heavy for EasyIDP core dependencies.
- The current design cannot easily support streaming large LAS/LAZ crops.

## Recommended Direction

- Keep PointCloud lightweight and compositional.
- Do not make Open3D a core dependency.
- Split into data object, IO backends, spatial index, and cropper.
- Make Open3D an optional `contrib` integration later if useful.

## Target Module Shape

```text
easyidp.pointcloud
  data.py       # PointCloudData: xyz, offset, colors, normals, crs
  io.py         # reader/writer registry
  las.py        # laspy/laz backend
  ply.py        # plyfile backend
  index.py      # SpatialIndex protocol, KDTree implementation
  cropper.py    # polygon crop and ROI crop
```

## Target Data Model

```python
@dataclass
class PointCloudData:
    points: np.ndarray
    offset: np.ndarray | None = None
    colors: np.ndarray | None = None
    normals: np.ndarray | None = None
    crs: pyproj.CRS | None = None
```

## IO Policy

- Use `laspy` for LAS/LAZ and support CRS VLR read/write where possible.
- Use `plyfile` for PLY.
- Use `laspy.open(...).chunk_iterator()` for future large point cloud streaming.
- Remove or make optional any dependency that is not used by the implementation.

## Crop Policy

- Use one crop engine: bbox/KDTree prefilter plus polygon contains test.
- Provide two public result styles if needed: raw ndarray points and `PointCloudData` object.
- Avoid repeated full `points` allocations in hot paths.

## Optional Open3D Integration

- Place Open3D support under optional/contrib modules only.
- Possible uses: PCD IO, visualization, downsampling, normal estimation, 3D KDTree.
- Do not import Open3D from core modules.

## Migration Notes

- v2.1 may rename confusing methods like `crop_polygon()` and `crop_point_cloud()` if it makes the API clearer.
- Keep easy loading and saving workflows: `PointCloud(path)`, `save(path)`, `crop_rois(roi)` where practical.
- Internal reconstruction and ROI modules should depend on a small point cloud protocol instead of the old monolithic class.
