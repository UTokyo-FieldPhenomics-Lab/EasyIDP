# Reconstruction Adapter Architecture Notes

## Current Model

- `Recons` is the shared base for `Pix4D` and `Metashape`.
- `Sensor`, `Photo`, `Calibration`, and `ChunkTransform` store camera and transform data.
- `Pix4D` and `Metashape` both parse project files, normalize data, project ROI points, filter image hits, and expose visualization helpers.
- `Photo.transform` has conflicting meanings: Pix4D uses a 3x4 projection matrix, while Metashape uses a 4x4 camera transform.

## Problem

- Parser logic, project model, coordinate transforms, projection math, filtering, and export are tightly coupled.
- Adding OpenDroneMap, Reality Capture, COLMAP, or DJI Terra would currently copy large chunks of `Pix4D`/`Metashape` logic.
- Camera models and coordinate systems vary significantly across software.
- Current APIs rely on mutable project state such as temporary CRS changes and cached photo positions.

## Recommended Direction

- Split reconstruction into Parser, Adapter, Project Model, and ProjectionEngine.
- Keep software-specific file parsing isolated.
- Normalize parser outputs into stable EasyIDP data models.
- Keep projection math in a reusable engine; use adapters for software-specific coordinate and camera model details.

## Target Module Shape

```text
easyidp.reconstruction
  project.py      # ReconstructionProject facade
  models.py       # Sensor, Photo, CameraModel, ReconstructionBundle
  engine.py       # ProjectionEngine
  filters.py      # distance/coverage/angle filters
  export.py       # json/png/raw-geotiff export
  parsers/
    pix4d.py
    metashape.py
    odm.py
    colmap.py
    reality_capture.py
    dji_terra.py
  adapters/
    pix4d.py
    metashape.py
    odm.py
    colmap.py
    reality_capture.py
    dji_terra.py
```

## Target Data Flow

```text
ReconstructionProject.open(path, software='auto')
  -> detect software
  -> Parser.parse(path) -> RawProjectData
  -> Adapter.normalize(raw) -> ReconstructionBundle
  -> ProjectionEngine(project, adapter)

project.back_project(roi)
  -> ROI CRS -> project local coordinates
  -> project local -> image pixels
  -> boundary/camera filters
  -> BackProjectionResult
```

## Core Models

```python
@dataclass
class Photo:
    id: str
    label: str
    path: Path | None
    sensor_id: str
    camera_to_world: np.ndarray | None = None
    world_to_camera: np.ndarray | None = None
    projection_matrix: np.ndarray | None = None


@dataclass
class ReconstructionBundle:
    software: str
    crs: pyproj.CRS | None
    sensors: list[Sensor]
    photos: list[Photo]
    products: ReconstructionProducts
    metadata: dict[str, Any]
```

## Software Notes

- Metashape: XML/zip parser, chunk transform, geocentric world CRS, Agisoft camera model.
- Pix4D: parameter folder parser, offset handling, pmatrix projection, Pix4D calibration.
- OpenDroneMap: OpenSfM JSON, camera models, georeferencing quality varies with GCP/GPS.
- COLMAP: cameras/images text or binary files, arbitrary local scale unless externally georeferenced.
- Reality Capture: likely needs exported camera CSV/XML, local/georeferenced transform risks.
- DJI Terra: output structure is version-sensitive; parser should be isolated and tolerant.

## Projection And Export Policy

- `ProjectionEngine` owns ROI-to-image projection orchestration.
- Adapters own CRS-to-local transforms, local-to-camera projection, and distortion model details.
- Filters such as distance, coverage, and view angle should live outside software adapters.
- Visualization should live outside core projection.
- Raw ROI GeoTIFF export should use the raster layer and should not be embedded inside parser/adapters.

## Migration Notes

- Keep `idp.Metashape` and `idp.Pix4D` as compatibility facades if useful.
- New development should target `ReconstructionProject` and `ProjectionEngine`.
- v2.1 may break confusing old names such as `Photo.transform` to separate matrix semantics.
- Keep future forward projection in mind, but do not implement it in v2.1 unless needed.

## References:

Detailed subagent investigation can refer to: 	`.agents/references/v2.1refactor/subagents/investigate_reconstruction_adapters.md`
