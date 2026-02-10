# Subplot Generator Feature Implementation

## Overview
Implemented a subplot generator for EasyIDP that creates grid-based subplots within field boundary polygons. This feature supports both grid-count and fixed-size modes, handles non-rectangular boundaries with configurable filtering, and includes visualization tools.

During implementation, the code was refactored to integrate closely with existing modules (`geotools`, `shp`) rather than standing alone.

## Implementation Details

### 1. Subplot Generation (`src/easyidp/geotools.py`)
- **Function**: `generate_subplots(boundary, ...)`
- **Logic**:
    - Calculates Minimum Area Rectangle (MAR) to align grid with field orientation.
    - Generates grid cells based on `row_num`/`col_num` OR `width`/`height`.
    - Classifies subplots as `inside`, `touch`, or `outside` relative to the boundary.
    - Filters results based on `keep` parameter (`"all"`, `"touch"`, `"inside"`).
- **Refactoring**: 
    - Initially created in `subplot.py`, then merged into `geotools.py` to consolidate geometric utilities.
    - Deleted `subplot.py` and removed references from `__init__.py`.

### 2. Visualization (`src/easyidp/visualize.py`)
- **Function**: `show_subplots(boundary_roi, subplot_roi, ...)`
- **Features**: 
    - Visualizes boundary and subplots on a matplotlib axis.
    - Color-codes subplots by status:
        - **Green**: Inside
        - **Orange**: Touch
        - **Red**: Outside
    - Supports saving to file via `save_as`.

### 3. File Saving (`src/easyidp/shp.py` & `src/easyidp/roi.py`)
- **Refactoring**:
    - Extracted Shapefile writing logic from `ROI` class to a standalone function `write_shp` in `src/easyidp/shp.py`.
    - Updated `ROI.save_shp` to delegate to `idp.shp.write_shp`.
    - Added generic `ROI.save()` method.
- **Features**: Does not just save geometry but also subplot metadata (`row`, `col`, `status`) to the Shapefile attributes (`.dbf`).

## API Usage

```python
import easyidp as idp

# 1. Load boundary
boundary = idp.ROI("field_boundary.shp")

# 2. Generate Subplots
# Option A: By Grid (e.g., 4 rows x 6 cols)
subplots = idp.geotools.generate_subplots(
    boundary, 
    row_num=4, 
    col_num=6,
    x_interval=0.5, 
    y_interval=0.5,
    keep="touch"  # Options: "all" | "touch" | "inside"
)

# Option B: By Size (e.g., 2m x 3m plots)
subplots_size = idp.geotools.generate_subplots(
    boundary, 
    width=2.0, 
    height=3.0
)

# 3. Visualize
idp.visualize.show_subplots(
    boundary, 
    subplots, 
    title="Field Subplots",
    save_as="subplots_vis.png"
)

# 4. Save to Shapefile
subplots.save("output_subplots.shp") 
# Or: subplots.save_shp("output_subplots.shp")
```

## Testing

### Unit Tests
- **Geotools Tests**: `tests/test_geotools.py` (Migrated from `test_subplot.py`)
    - Validates grid generation, naming conventions (`R1C1`), size calculations, and filtering logic.
    - 20 tests passed.
- **Visualization Tests**: `tests/test_visualize.py` (`TestShowSubplots` class)
    - Generates sample images to verify rendering of different `keep` modes.
    - 4 tests passed.

### Visual Verification
Generated test outputs in `tests/out/visual_test/`:
- `rect_grid.png`: Standard grid on rectangular boundary.
- `l_shape_keep_all.png`: L-shaped boundary showing all MAR subplots.
- `l_shape_keep_touch.png`: Filtered to exclude fully distinct subplots.
- `l_shape_keep_inside.png`: Filtered to include only fully contained subplots.

## Files Modified
| File | Status | Description |
|------|--------|-------------|
| `src/easyidp/geotools.py` | Modified | Added `generate_subplots` and helpers |
| `src/easyidp/shp.py` | Modified | Added `write_shp` function |
| `src/easyidp/roi.py` | Modified | Refactored `save_shp` and added `save` |
| `src/easyidp/visualize.py` | Modified | Added `show_subplots` |
| `src/easyidp/__init__.py` | Modified | Removed old `subplot` references |
| `tests/test_geotools.py` | Created | New home for subplot logic tests |
| `tests/test_visualize.py` | Modified | Added visualization tests |
| `src/easyidp/subplot.py` | Deleted | Merged into geotools |
| `tests/test_subplot.py` | Deleted | Merged into test_geotools |
