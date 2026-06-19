# back2raw 批量优化实现总结

**日期**: 2026-01-30  
**作者**: Antigravity Assistant

---

## 概述

为 `easyidp.Metashape` 类实现了批量优化的 `back2raw_batch()` 方法，通过使用 NumPy 向量化操作实现 **4.44x-23.88x** 的性能提升。

---

## 性能基准测试结果

| ROI 数量 | 照片数 | 原版本 (`back2raw`) | 批量版 (`back2raw_batch`) | 加速比 |
|---------|--------|---------------------|---------------------------|--------|
| 4 | 151 | 0.0604s | 0.0136s | **4.44x** |
| 16 | 151 | 0.2398s | 0.0190s | **12.62x** |
| 112 | 151 | 1.6909s | 0.0708s | **23.88x** |

---

## 修改文件列表

### 1. `src/easyidp/reconstruct.py`

**修改内容**: 更新 `Calibration._calibrate_metashape_frame()` 方法

- 改进了文档字符串，添加了 NumPy 风格的完整参数说明
- 优化了内部计算逻辑，使变量命名更清晰
- 保持对 1D 和 2D 数组的兼容性（支持批量处理）

```python
def _calibrate_metashape_frame(
    self,
    xh: np.ndarray,
    yh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch distortion correction for Metashape frame cameras."""
    # ... 优化后的实现
```

### 2. `src/easyidp/metashape.py`

**新增方法**:

#### `_prepare_camera_transforms()`
预计算所有启用照片的变换矩阵。

```python
def _prepare_camera_transforms(self) -> tuple[np.ndarray, list, dict, dict]:
    """Precompute combined transform matrices for all enabled photos."""
```

**返回值**:
- `transforms`: `(N, 4, 4)` 相机变换矩阵
- `photo_names`: 与 transforms 对应的照片名列表
- `sensor_groups`: `{sensor_id: [photo_indices]}`
- `sensors_dict`: `{sensor_id: Sensor}`

#### `_batch_project_to_cameras()`
使用 `np.einsum` 批量投影所有点到所有相机。

```python
def _batch_project_to_cameras(
    self,
    points_local: np.ndarray,
    transforms: np.ndarray,
    sensor_groups: dict,
    sensors_dict: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Project points to all cameras using batch matrix operations."""
```

**核心优化**:
```python
# 使用 einsum 进行批量矩阵运算
# (points - t) @ R 的批量计算
xyz_batch = np.einsum('nmi,nij->nmj', diff, r_batch)
```

#### `back2raw_batch()`
公开 API，与 `back2raw()` 兼容的优化版本。

```python
def back2raw_batch(self, roi, save_folder=None, **kwargs) -> dict:
    """Projects ROIs to raw images using batch matrix operations."""
```

**关键特性**:
1. **点去重**: 合并所有 ROI 点，减少冗余计算
2. **批量矩阵运算**: 使用 `np.einsum` 一次计算所有点对所有相机的投影
3. **按 sensor 分组校正**: 按相同 sensor 分组进行批量畸变校正
4. **进度条显示**: 添加了 5 步骤的进度条

### 3. `tests/test_back2raw_performance.py` [新文件]

**测试类**:

- `TestBack2rawBatchConsistency`: 验证批量版与原版结果一致
- `TestBack2rawBatchPerformance`: 性能基准测试
- `TestBack2rawBatchEdgeCases`: 边缘情况测试

---

## 优化原理

### 原始实现的瓶颈

```
O(R × M × N) 复杂度
├── R 个 ROI 循环
│   └── N 张照片循环
│       └── M 个顶点计算
```

### 优化后的处理流程

```
Step 1: 收集并去重所有 ROI 点 → (M_unique, 3)
Step 2: 一次性 CRS 转换 (消除重复转换)
Step 3: 预计算相机变换矩阵 → (N, 4, 4)
Step 4: einsum 批量投影 → (N, M_unique, 2)
Step 5: 按 ROI 重建结果
```

### einsum 矩阵运算

```python
# 原始逐张处理
for photo in photos:
    xyz = (points - t).dot(R)

# 优化后批量处理
diff = points[np.newaxis, :, :] - t[:, np.newaxis, :]  # (N, M, 3)
xyz_batch = np.einsum('nmi,nij->nmj', diff, r_batch)   # (N, M, 3)
```

---

## 使用示例

```python
import easyidp as idp

# 加载项目和 ROI
ms = idp.Metashape(project_path)
roi = idp.ROI(shapefile_path, name_field=0)
roi.get_z_from_dsm(dsm_path)

# 使用优化版本 (API 与 back2raw 完全相同)
result = ms.back2raw_batch(roi)

# 结果格式: {roi_name: {photo_name: pixel_coords (n, 2)}}
```

---

## 测试验证

```bash
# 运行所有测试
uv run pytest tests/test_back2raw_performance.py -v -s

# 结果: 6 passed, 1 skipped
```

---

Nya~♡
