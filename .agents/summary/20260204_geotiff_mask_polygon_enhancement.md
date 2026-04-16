# GeoTiff Mask Polygon Enhancement - 完整实现总结

## 概述

本次更新为 `easyidp.GeoTiff` 类实现了基于 polygon 的 mask 系统和 affine 旋转存储功能，提供了更高精度的 ROI 边界表示和 QGIS 兼容的旋转 GeoTiff 输出。

**测试状态：** ✅ 42 passed (25 原有 + 17 新增)

---

## 核心架构变更

### 1. Polygon-First Mask 架构

**设计理念：**
- Geo polygon 作为源数据（source of truth）
- Binary mask 从 polygon 按需计算
- 保证精度无损，避免栅格化误差

**新增属性：**
```python
self._mask_polygon: np.ndarray | None      # (n, 2) polygon 坐标
self._mask_polygon_is_geo: bool            # True = geo coords, False = pixel coords
self._use_affine: bool                     # True = 使用 affine 旋转存储
```

**新增属性（只读）：**
```python
@property
def use_affine(self) -> bool:
    """是否使用 affine 旋转存储（从 transform 自动检测）"""
    
@property
def mask_polygon(self) -> np.ndarray | None:
    """返回存储的 polygon（原样返回）"""
    
@property
def mask_polygon_geo(self) -> np.ndarray | None:
    """总是返回 geo 坐标的 polygon"""
    
@property
def mask_polygon_pixel(self) -> np.ndarray | None:
    """总是返回 pixel 坐标的 polygon"""
```

**Mask 优先级：**
```python
@property
def mask(self) -> np.ndarray:
    # 优先级：
    # 1. polygon -> binary (如果有 polygon)
    # 2. affine mode -> 全 True (如果 use_affine)
    # 3. legacy nodata/alpha (传统方式)
```

---

## 核心功能实现

### 2. Polygon 操作方法

#### 2.1 设置 Polygon
```python
def set_mask_polygon(self, polygon: np.ndarray, is_geo: bool = True):
    """设置 mask polygon
    
    Parameters
    ----------
    polygon : np.ndarray
        (n, 2) polygon 坐标，自动闭合
    is_geo : bool
        True = geo 坐标，False = pixel 坐标
    """
```

#### 2.2 坐标转换（内部方法）
```python
def _polygon_pixel_to_geo(self, polygon: np.ndarray) -> np.ndarray:
    """Pixel polygon -> Geo polygon"""
    
def _polygon_geo_to_pixel(self, polygon: np.ndarray) -> np.ndarray:
    """Geo polygon -> Pixel polygon"""
```

#### 2.3 Polygon ↔ Binary 转换
```python
def _polygon_to_binary(self, polygon_pixel: np.ndarray) -> np.ndarray:
    """使用 skimage.draw.polygon 将 polygon 转为 binary mask"""
    
def _binary_to_polygon(self, mask: np.ndarray) -> np.ndarray:
    """使用 skimage.measure.find_contours 提取 polygon"""
```

---

### 3. Affine 旋转存储

#### 3.1 矩形验证
```python
def _is_valid_rectangle(self, polygon: np.ndarray) -> tuple[bool, float, tuple]:
    """检查 polygon 是否为有效矩形
    
    Returns
    -------
    is_valid : bool
        是否为 4 个顶点、90° 角的矩形
    angle : float
        旋转角度（度）
    bounds : tuple
        (origin_x, origin_y, width, height)
    
    Notes
    -----
    - 使用 rtol=0, atol=1e-6 避免大坐标值的相对误差
    - 角度容差 ±2°
    """
```

#### 3.2 Affine 存储准备
```python
def _prepare_affine_storage(
    self,
    imarray: np.ndarray,
    profile: dict,
    polygon_geo: np.ndarray,
    angle: float,
    bounds: tuple,
) -> tuple[np.ndarray, dict]:
    """准备 affine 旋转存储
    
    核心算法：
    1. 计算输出图像尺寸（基于矩形 bounds）
    2. 为每个输出像素计算对应的 geo 坐标
       - 沿第一条边方向（angle）
       - 沿垂直方向（angle + 90°）
    3. 将 geo 坐标转换为输入图像的 pixel 坐标
    4. 使用 scipy.ndimage.map_coordinates 双线性插值采样
    5. 构建新的 affine transform（包含旋转）
    
    关键修复：
    - Y 轴反向：geo Y 向上增加，pixel row 向下增加
    - 正确公式：
      geo_x = origin_x + col * dx * cos(a) + row * dy * sin(a)
      geo_y = origin_y + col * dx * sin(a) - row * dy * cos(a)
    """
```

---

### 4. 模式转换方法

#### 4.1 转换到 Affine 模式
```python
def convert_to_affine(self) -> 'GeoTiff':
    """转换为 affine 旋转存储（返回新对象）
    
    要求：
    - 必须先设置 mask_polygon
    - Polygon 必须是有效矩形
    
    返回：
    - 新的 GeoTiff 对象（use_affine=True）
    - 原对象不变（数据保护）
    
    Raises
    ------
    ValueError : 如果 polygon 不是矩形
    """
```

#### 4.2 转换回标准模式
```python
def convert_from_affine(self, target_bounds: tuple | None = None) -> 'GeoTiff':
    """从 affine 转回标准存储（返回新对象）
    
    Parameters
    ----------
    target_bounds : tuple, optional
        (min_x, min_y, max_x, max_y)
        默认使用 polygon 的 bounding box
    
    返回：
    - 新的 GeoTiff 对象（use_affine=False）
    - 原对象不变（数据保护）
    """
```

---

### 5. 保存和读取

#### 5.1 保存增强
```python
def save(
    self, 
    save_path: str | Path, 
    overwrite: bool = False, 
    apply_mask: bool = True,
    use_affine: bool = False,  # 新增参数
) -> bool:
    """保存 GeoTiff
    
    use_affine 参数：
    - True: 如果 polygon 是矩形，保存为 affine 旋转模式
    - False: 标准保存
    
    重要：
    - save() 不修改对象本身
    - 只影响保存的文件
    - Polygon 写入 EASYIDP_MASK_POLYGON metadata tag (WKT 格式)
    """
```

#### 5.2 读取增强
```python
def open(self, tif_path: str | Path):
    """打开 GeoTiff
    
    自动检测：
    1. 读取 EASYIDP_MASK_POLYGON tag -> 恢复 polygon
    2. 检测 transform 的 b/d 系数 -> 设置 use_affine
    """
```

---

### 6. 集成到现有功能

#### 6.1 crop_shapely_polygon
```python
def crop_shapely_polygon(
    self, 
    shapely_polygon: Polygon, 
    save_path: str | Path | None = None, 
    return_geotiff: bool = False
):
    """裁剪 polygon 区域
    
    新增：自动存储 polygon 到结果 GeoTiff
    """
```

#### 6.2 one_raw_roi2geotiff
```python
def one_raw_roi2geotiff(
    roi_crs: pyproj.CRS,
    roi_geo_coords: np.ndarray,
    raw_img_path: str | Path,
    roi_raw_px: np.ndarray,
    nodata: float | int = 0,
    has_alpha: bool = True,
) -> GeoTiff:
    """Raw 图像转 GeoTiff
    
    新增：存储 ROI polygon 到结果 GeoTiff
    """
```

#### 6.3 back2raw2geotiff
```python
def back2raw2geotiff(
    recons: idp.reconstruct.Recons,
    back2raw_result: dict,
    roi,
    output_folder: str | Path | None = None,
    nodata: float | int = 0,
    has_alpha: bool = True,
    use_affine: bool = False,  # 新增参数
) -> dict:
    """批量转换 back2raw 结果
    
    新增：use_affine 参数传递到 save()
    """
```

---

## 使用示例

### 示例 1: 基本 Polygon Mask
```python
import easyidp as idp
import numpy as np

# 加载 GeoTiff
gtiff = idp.GeoTiff('input.tif')

# 设置矩形 mask（geo 坐标）
polygon = np.array([
    [368025.0, 3955479.0],
    [368027.0, 3955479.0],
    [368027.0, 3955477.0],
    [368025.0, 3955477.0],
])
gtiff.set_mask_polygon(polygon, is_geo=True)

# 保存 - polygon 自动写入 metadata
gtiff.save('output.tif', overwrite=True)

# 重新加载 - polygon 自动恢复
gtiff2 = idp.GeoTiff('output.tif')
print(gtiff2.mask_polygon)  # 恢复的 polygon
print(gtiff2.mask)          # 从 polygon 计算的 binary mask
```

### 示例 2: Affine 旋转保存
```python
# 方式 1: 保存时使用 affine（不修改对象）
gtiff.set_mask_polygon(rect_polygon, is_geo=True)
gtiff.save('output_affine.tif', use_affine=True)
# gtiff 本身不变，只有保存的文件使用 affine

# 方式 2: 转换对象到 affine 模式
affine_gtiff = gtiff.convert_to_affine()
print(affine_gtiff.use_affine)  # True
print(affine_gtiff.imarray.shape)  # 更小（裁剪到矩形）
print(gtiff.use_affine)  # False（原对象不变）

# 保存 affine 对象
affine_gtiff.save('affine.tif')

# 在 QGIS 中打开 affine.tif 会正确显示旋转
```

### 示例 3: 模式转换
```python
# 标准 -> Affine
gtiff = idp.GeoTiff('standard.tif')
gtiff.set_mask_polygon(rect_coords, is_geo=True)
affine_gtiff = gtiff.convert_to_affine()

# Affine -> 标准
standard_gtiff = affine_gtiff.convert_from_affine()

# 往返转换保持 polygon
np.testing.assert_allclose(
    standard_gtiff.mask_polygon_geo[:4],
    rect_coords,
    atol=1e-6
)
```

### 示例 4: Crop 自动保存 Polygon
```python
gtiff = idp.GeoTiff('dom.tif')
polygon = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

# Crop - polygon 自动保存
cropped = gtiff.crop_polygon(polygon, is_geo=True, return_geotiff=True)
print(cropped.mask_polygon)  # 自动保存了

# 保存后重新加载
cropped.save('cropped.tif')
reloaded = idp.GeoTiff('cropped.tif')
print(reloaded.mask_polygon)  # polygon 保留
```

### 示例 5: Back2raw 使用 Affine
```python
import easyidp as idp

# 准备数据
roi = idp.ROI('plots.shp')
roi.get_z_from_dsm('dsm.tif')
ms = idp.Metashape('project.psx')
back2raw_out = roi.back2raw(ms)

# 转换为 GeoTiff（使用 affine 旋转）
geotiffs = idp.geotiff.back2raw2geotiff(
    recons=ms,
    back2raw_result=back2raw_out,
    roi=roi,
    output_folder='./output',
    use_affine=True,  # 矩形 ROI 会使用 affine 存储
)

# 结果文件更小，QGIS 中正确显示旋转
```

---

## 技术细节

### 坐标系统
- **Geo 坐标：** 地理坐标系（如 UTM），Y 轴向上
- **Pixel 坐标：** 图像坐标系，row 向下增加
- **Transform：** Affine 变换矩阵，处理坐标转换

### Affine Transform 结构
```python
# 标准（无旋转）
transform = Affine.translation(x0, y0) * Affine.scale(dx, -dy)
# [dx,  0, x0]
# [ 0, dy, y0]

# 带旋转
transform = Affine.translation(x0, y0) * Affine.rotation(angle) * Affine.scale(dx, -dy)
# [dx*cos, -dy*sin, x0]
# [dx*sin,  dy*cos, y0]
```

### Polygon 闭合规则
- 输入可以是开放或闭合的
- `set_mask_polygon` 自动添加闭合点（如果需要）
- 内部存储总是闭合的
- 使用 `rtol=0, atol=1e-6` 检测闭合（避免大坐标值误判）

### 插值方法
- 使用 `scipy.ndimage.map_coordinates` 双线性插值（order=1）
- 边界外使用 `cval=0`（黑色）
- 保持原始数据类型（dtype）

---

## 测试覆盖

### TestMaskPolygon (11 tests)
1. `test_set_mask_polygon_geo` - 设置 geo polygon
2. `test_set_mask_polygon_pixel` - 设置 pixel polygon
3. `test_mask_binary_from_polygon` - Polygon -> Binary 转换
4. `test_polygon_metadata_storage` - Metadata 存储和读取
5. `test_affine_rectangle_detection` - 矩形检测
6. `test_affine_non_rectangle_warning` - 非矩形警告
7. `test_backward_compatibility_no_polygon` - 向后兼容
8. `test_affine_save_and_reload` - Affine 保存和重载
9. `test_affine_coordinate_conversion` - 坐标转换
10. `test_crop_polygon_stores_mask` - Crop 保存 polygon
11. `test_one_raw_roi2geotiff_stores_polygon` - ROI 转换保存 polygon

### TestAffineConversion (6 tests)
1. `test_convert_to_affine_returns_new_object` - 转换返回新对象
2. `test_convert_from_affine_returns_new_object` - 反向转换返回新对象
3. `test_convert_roundtrip_data_consistency` - 往返转换一致性
4. `test_convert_preserves_original` - 原对象不变
5. `test_convert_non_rectangle_raises` - 非矩形抛异常
6. `test_already_affine_returns_self` - 已是 affine 返回 self

---

## 重要修复

### 修复 1: Y 轴方向
**问题：** 保存的 affine GeoTiff 显示为镜像
**原因：** Geo Y 向上，Pixel row 向下
**修复：**
```python
# 错误
geo_x = origin_x + col * dx * cos_a - row * dy * sin_a
geo_y = origin_y + col * dx * sin_a + row * dy * cos_a

# 正确
geo_x = origin_x + col * dx * cos_a + row * dy * sin_a
geo_y = origin_y + col * dx * sin_a - row * dy * cos_a
```

### 修复 2: 闭合点检测
**问题：** 矩形被误判为非矩形
**原因：** `np.allclose` 默认 `rtol=1e-5`，大坐标值（~3955477）产生 ~40 的容差
**修复：**
```python
# 错误
pts = polygon[:-1] if np.allclose(polygon[0], polygon[-1]) else polygon

# 正确
pts = polygon[:-1] if np.allclose(polygon[0], polygon[-1], rtol=0, atol=1e-6) else polygon
```

### 修复 3: 对象保护
**问题：** `save(use_affine=True)` 修改了对象状态
**修复：** 移除 `self._use_affine = True`，只影响保存的文件

### 修复 4: 转换方法返回新对象
**问题：** `convert_to_affine()` 修改原对象
**修复：** 返回新 GeoTiff 对象，保护原数据

---

## API 变更总结

### 新增公开 API
- `GeoTiff.mask_polygon` (property)
- `GeoTiff.mask_polygon_geo` (property)
- `GeoTiff.mask_polygon_pixel` (property)
- `GeoTiff.use_affine` (property)
- `GeoTiff.set_mask_polygon(polygon, is_geo)`
- `GeoTiff.convert_to_affine()` -> GeoTiff
- `GeoTiff.convert_from_affine(target_bounds)` -> GeoTiff

### 修改的 API
- `GeoTiff.save(..., use_affine=False)` - 新增参数
- `back2raw2geotiff(..., use_affine=False)` - 新增参数

### 新增内部方法
- `_polygon_pixel_to_geo()`
- `_polygon_geo_to_pixel()`
- `_polygon_to_binary()`
- `_binary_to_polygon()`
- `_is_valid_rectangle()`
- `_prepare_affine_storage()`

---

## 性能考虑

### 优化点
1. **Lazy 计算：** Binary mask 从 polygon 按需计算，缓存结果
2. **向量化：** 使用 NumPy 向量操作，避免循环
3. **稀疏存储：** Polygon 比 binary mask 占用更少内存

### 性能影响
- **Polygon -> Binary：** O(n * m)，n=polygon 点数，m=图像像素数
- **Affine 转换：** O(w * h * bands)，需要重采样整个图像
- **Metadata 读写：** 可忽略（WKT 字符串很小）

---

## 兼容性

### 向后兼容
- ✅ 旧代码无需修改
- ✅ 旧 GeoTiff 文件正常读取
- ✅ Legacy mask 方法仍然工作

### QGIS 兼容
- ✅ Affine 旋转 GeoTiff 正确显示
- ✅ WKT polygon 可被 QGIS 读取（如果支持自定义 tag）

### 依赖
- `rasterio` - GeoTiff I/O
- `shapely` - Polygon 操作和 WKT
- `scikit-image` - Polygon 绘制和轮廓提取
- `scipy` - 图像重采样

---

## 未来改进方向

1. **性能优化：**
   - 使用 Cython 加速 polygon -> binary 转换
   - GPU 加速图像重采样

2. **功能扩展：**
   - 支持多个 polygon（多 ROI）
   - 支持 polygon 编辑（添加/删除顶点）
   - 支持更多 polygon 格式（GeoJSON）

3. **可视化：**
   - 添加 `plot_polygon()` 方法
   - 集成到 Jupyter notebook 显示

4. **文档：**
   - 添加更多使用示例
   - 创建教程视频

---

## 文件清单

### 修改的文件
- `src/easyidp/geotiff.py` - 核心实现（+600 行）
- `tests/test_geotiff.py` - 测试用例（+17 个测试）

### 新增依赖
- 无（使用已有依赖）

### 文档
- `.agent/SUMMARY.md` - 本文档

---

## 开发者注意事项

### 调试技巧
1. **检查 polygon：** `print(gtiff.mask_polygon_geo)`
2. **检查 transform：** `print(gtiff.header['profile']['transform'])`
3. **可视化 mask：** `plt.imshow(gtiff.mask)`
4. **检查 use_affine：** `print(gtiff.use_affine)`

### 常见陷阱
1. **坐标系混淆：** 始终明确 geo vs pixel
2. **闭合点：** 注意 polygon 是否闭合
3. **Y 轴方向：** Geo Y 向上，Pixel row 向下
4. **大坐标值：** 使用绝对容差，不用相对容差

### 代码风格
- PEP 8 规范
- Numpy 风格 docstring
- 函数 < 50 行
- 缩进深度 ≤ 3 层（使用 Guard Clauses）

---

**最后更新：** 2026-02-04  
**版本：** EasyIDP 2.0.2+  
**作者：** Antigravity (Google Deepmind)  
**测试状态：** ✅ 42/42 passed

Nya~♡
