# Optimization of `back2raw2geotiff` (2026-02-04)

## Overview
This update optimizes the `easyidp.geotiff.back2raw2geotiff` function to address severe I/O bottlenecks when processing large datasets. The original implementation loaded the same raw image repeatedly for every ROI contained within it. The new implementation loads each image exactly once and processes all associated ROIs in memory, achieving significant speedups.

## Key Changes & Principles

### 1. Loop Inversion (I/O Reduction)
*   **Before**: Iterate ROIs -> Find Image -> Load Image -> Crop -> Save.
    *   *Problem*: If 100 ROIs are on `IMG_01.JPG`, the 20MB image is read 100 times (2GB I/O).
*   **After**: Iterate Images -> Load Image (Once) -> Iterate ROIs on this Image -> Crop -> Save.
    *   *Result*: `IMG_01.JPG` is read only once (20MB I/O).

### 2. Multiprocessing
*   Processing is parallelized at the **Image level**.
*   Each process handles one image and all its contained ROIs.
*   Implemented using `concurrent.futures.ProcessPoolExecutor`.

### 3. Memory Safety Strategy
To prevent Out-Of-Memory (OOM) errors when spinning up multiple workers loading large images:
*   **Metadata Estimation**: Instead of pre-loading a file, we use `recons.sensors` metadata (`width`, `height`) to calculate the raw byte size of an image (`W * H * 3 bytes`).
*   **Dynamic Worker Count**:
    *   Get available system RAM via `psutil`.
    *   Calculate `max_safe_workers = (Available RAM * 75%) // (Image Bytes * 1.5 safety factor)`.
    *   Automatically cap workers to avoiding swapping or crashing.

## API Changes

### `easyidp.geotiff.back2raw2geotiff`
Added `num_workers` parameter.

```python
def back2raw2geotiff(
    ...,
    num_workers: int | None = None,
) -> dict
```
*   `num_workers=None` (Default): Auto-calculate safe worker count based on RAM.
*   `num_workers=N`: Force use of N processes.

### `easyidp.geotiff.one_raw_roi2geotiff`
Signature updated to accept pre-loaded image arrays.

```python
def one_raw_roi2geotiff(
    ...,
    raw_img: str | Path | np.ndarray, 
    roi_raw_px_coords: np.ndarray,
    ...
)
```
*   `raw_img`: Can now be a `numpy.ndarray` (image already in memory) or a `Path` (load inside function).
*   `roi_raw_px_coords`: Renamed from `roi_raw_px` for clarity (internal change).

## Usage Examples

### Standard Usage (Auto-Optimization)
The default behavior automatically uses parallel processing with a safe number of workers.

```python
import easyidp as idp

# ... prepare recons and back2raw_out ...

geotiffs = idp.geotiff.back2raw2geotiff(
    recons=ms,
    back2raw_result=back2raw_out,
    roi=roi,
    output_folder='./output'
)
```

### Manual Worker Control
If you want to force specific parallelism (e.g., for benchmarking or specific resource constraints):

```python
geotiffs = idp.geotiff.back2raw2geotiff(
    ...,
    num_workers=4  # Force 4 processes
)
```

## Internal Implementation Details
*   **Data Pivoting**: The input dictionary `{roi_id: {img_id: px}}` is internally pivoted to `{img_id: {roi_id: px}}` to facilitate image-first processing.
*   **Worker Function**: A new module-level function `_process_single_image_task` handles the workload for a single image to ensure picklability for multiprocessing.
