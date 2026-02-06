# Subplot Generator - Implementation Summary

## 20260206 Session: Subplot Generator Feature

### What Was Done

Implemented a subplot generator for EasyIDP that creates grid-based subplots within field boundary polygons.

### New Files

| File | Description |
|------|-------------|
| subplot.py | Core module with `generate_subplots()` function |
| test_subplot.py | 19 test cases for subplot generation |

### Modified Files

| File | Changes |
|------|---------|
| roi.py | Added `save_shp()` method |
| visualize.py | Added `show_subplots()` function |
| \_\_init\_\_.py | Added subplot module imports |

### API Usage

```python
import easyidp as idp

# Load boundary
boundary = idp.ROI("field_boundary.shp")

# Generate by grid (row x col)
subplots = idp.generate_subplots(
    boundary, row_num=4, col_num=6,
    x_interval=0.5, y_interval=0.5,
    keep="touch"  # "all" | "touch" | "inside"
)

# Or generate by size (width x height in meters)
subplots = idp.generate_subplots(
    boundary, width=2.0, height=3.0,
    x_interval=0.3, y_interval=0.3
)

# Visualize
idp.visualize.show_subplots(boundary, subplots)

# Save to shapefile
subplots.save_shp("output_subplots.shp")
```

### Key Features

- **Dual input modes**: Grid (row_num/col_num) or size (width/height)
- **MAR-based orientation**: Automatically aligns to field direction
- **Keep mode filtering**: `"all"`, `"touch"`, `"inside"` for non-rectangular boundaries
- **Status tracking**: Each subplot has `inside`/`touch`/`outside` status
- **Visualization**: Color-coded by status (green=inside, orange=touch, red=outside)

### Test Results

```
tests/test_subplot.py: 19 passed ✓
```
