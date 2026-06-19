# User Prompt

只读背景调查，不要修改任何文件。工作目录 /home/crest/Documents/Github/EasyIDP。请重点分析 EasyIDP 当前 GeoTiff/cvtools 与未来基于 rasterio 的架构重设计。

上下文：用户希望 GeoTiff 能切块和保存，支持多种 mask 格式，保存时可应用 Affine Transform 来保存旋转裁剪结果，减少倾斜矩形 ROI 的外部像素浪费。用户怀疑 GeoTiff 不应完全自写，可继承或封装 rasterio 对应 raster 类，并提供 EasyIDP 风格接口。

请做：

1. 阅读 src/easyidp/geotiff.py、src/easyidp/cvtools.py、src/easyidp/visualize.py 中 raster/裁剪/mask/affine 相关部分。
2. 总结当前 GeoTiff 的主要职责、状态、调用链。
3. 评估继承 rasterio DatasetReader/DatasetWriter 是否现实，或更适合组合 rasterio dataset/path/profile/window/mask。
4. 对 mask、多边形裁剪、rotated affine crop、save、lazy read、窗口读取、大图性能提出新架构建议。
5. 给出如何兼容现有 idp.GeoTiff API 的迁移路径。

返回中文结构化 Markdown，只做分析，不写文件。

---

# EasyIDP GeoTiff / cvtools 架构分析与 rasterio 重设计评估

## 1. 当前架构全貌

### 1.1 文件职责总览

| 文件                         | 行数 | 角色                                        |
| ---------------------------- | ---- | ------------------------------------------- |
| `src/easyidp/geotiff.py`   | 3237 | GeoTiff 类 + 10 个模块级函数                |
| `src/easyidp/cvtools.py`   | 275  | 纯 numpy 的裁剪/多边形光栅化工具            |
| `src/easyidp/visualize.py` | 562  | matplotlib 可视化，与 raster/裁剪无直接关系 |

### 1.2 GeoTiff 类的状态与职责

`GeoTiff` 是一个**自包含的 Python 对象**，不继承任何类（仅 `object`），它有 3 种构造方式：

```python
# 方式1: 从文件懒加载
gtiff = GeoTiff("path/to/file.tif")  # -> 触发 self.open()

# 方式2: 从内存 ndarray + header dict
gtiff = GeoTiff(imarray=ndarray, header=header_dict)

# 方式3: 空对象 + 后续 open()
gtiff = GeoTiff()
gtiff.open("path/to/file.tif")
```

**内部状态字段**（属性）：

| 字段                     | 类型                | 说明                                                   |
| ------------------------ | ------------------- | ------------------------------------------------------ |
| `file_path`            | `Path\|None`       | 关联的磁盘文件路径                                     |
| `header`               | `dict`            | 所有元数据（宽、高、波段、CRS、transform、profile 等） |
| `_imarray`             | `np.ndarray\|None` | 懒加载的像素数据，shape=(H,W,C)                        |
| `_mask`                | `np.ndarray\|None` | bool mask，(H,W)，True=有效像素                        |
| `_mask_polygon`        | `np.ndarray\|None` | 多边形 mask，(N,2)，geo 或 pixel 坐标                  |
| `_mask_polygon_is_geo` | `bool`            | mask 坐标系类型标记                                    |
| `_use_affine`          | `bool`            | 是否处于 affine 旋转存储模式                           |

**header dict 结构**（`get_header()` 生成）：

```python
{
    "height": 5752, "width": 5490, "dim": 4,
    "nodata": 0, "dtype": dtype('uint8'),
    "scale": [0.00738, 0.00738],
    "tie_point": [368014.54157, 3955518.27477],
    "crs": <pyproj.CRS>,
    "transform": <affine.Affine>,
    "profile": <dict: rasterio DatasetReader.profile 的深拷贝>,
    "colorinterp": <tuple of ColorInterp>,
    "has_alpha": bool,
}
```

### 1.3 核心 API 方法清单（23 个）

**IO 层**：

- `open(tif_path)` — 读取头信息，检测 affine 模式，读取自定义标签（EASYIDP_MASK_POLYGON WKT）
- `save(save_path, overwrite, apply_mask, use_affine)` — 保存为 GeoTIFF，按数据类型选择 nodata/alpha 策略
- 模块函数 `get_header()`, `get_imarray()` — 底层 IO

**裁剪层**（调用链：`crop_polygon` -> `crop_shapely_polygon` / `crop_rectangle` -> `crop_shapely_polygon`）：

- `crop_polygon(polygon_hv, is_geo, ...)` — 裁剪单个多边形
- `crop_shapely_polygon(shapely_polygon, ...)` — 底层裁剪，使用 `rasterio.mask.mask`
- `crop_rectangle(left, top, w, h, ...)` — 矩形裁剪
- `crop_rois(roi, ...)` — 批量裁剪多个 ROI

**坐标转换**：

- `geo2pixel(polygon_hv, return_index)` — 使用 `~src.transform`
- `pixel2geo(polygon_hv)` — 使用 `src.transform` / `src.xy()`

**查询与统计**：

- `point_query(points_hv, is_geo)` — 使用 `src.sample()`
- `polygon_math(polygon_hv, kernel)` — 区域内统计（mean/min/max/percentile）

**Mask 管理（polygon-first 架构）**：

- `set_mask_polygon(polygon, is_geo)` — 设置多边形 mask
- `mask_polygon` / `mask_polygon_geo` / `mask_polygon_pixel` — 属性访问
- `mask` — 懒计算属性，优先级：polygon > affine > legacy
- `_compute_mask()` — 3 级 fallback: GDAL internal mask > alpha channel > nodata

**Affine 旋转裁剪**（核心新功能）：

- `convert_to_affine()` — 将标准存储转为 affine 旋转存储，返回新对象
- `convert_from_affine(target_bounds)` — 反向转换
- `_prepare_affine_storage(imarray, profile, rect_info)` — 采样+构建旋转 transform
- `_get_rectangle_affine_info(polygon)` — 从多边形提取矩形几何信息
- `_is_valid_rectangle()` / `_sort_rectangle_points()` / `_is_ordered_rectangle()` — 矩形验证

**辅助**：

- `has_data()`, `has_alpha`, `use_affine` — 状态查询
- 属性简写：`crs`, `height`, `width`, `dim`, `nodata`, `scale`, `tie_point`

**模块级独立函数**（10 个）：

- `one_raw_roi2geotiff()` — 将单张原始图像 ROI 投影变换为 GeoTiff（使用 skimage ProjectiveTransform + warp）
- `back2raw2geotiff()` — 并行批量处理
- `create_binary_mask_for_geotiff()` — 从 shapefile/ROI 生成格网对齐的二值 mask
- 已废弃的 `geo2pixel()` / `pixel2geo()`（旧实现，仍保留向后兼容）

### 1.4 cvtools.py 的职责

纯 numpy 操作，不依赖 rasterio，供 `roi.crop()` 和 `reconstruct.save_back2raw_json_and_png()` 调用：

- `imarray_crop(imarray, mask, nodata)` — 通过多边形坐标(N,2)或 bool mask(H,W)进行 numpy 裁剪，返回 (cropped_im, offset_xy, mask_out)
- `poly2mask(image_shape, poly_coord)` — 多边形转 bool mask（skimage 后端）
- `rgb2gray(rgb)` — 灰度转换

### 1.5 关键调用链

```
roi.crop(dom)                                    # roi.py:1263
  └─ dom.crop_rois(roi)                          # geotiff.py:1633
       └─ dom.crop_polygon(polygon)              # geotiff.py:1874
            └─ geo2pixel 转换 (如需要)
            └─ dom.crop_shapely_polygon(Polygon) # geotiff.py:1747
                 └─ rio.open(src)                # 打开源文件
                 └─ rasterio.mask.mask()         # 地理空间裁剪
                 └─ np.moveaxis 调整轴序
                 └─ 构建新 GeoTiff(imarray, header)
                 └─ 设置 mask_polygon
                 └─ convert_to_affine() (可选)
                 └─ save(save_path) (可选)

roi.get_z_from_dsm(dsm)                          # roi.py:960
  └─ dsm.polygon_math(polygon, kernel)           # geotiff.py:2137
       └─ dsm.crop_polygon()                     # 获取局部 ndarray
       └─ _compute_mask()                        # 识别有效像素
       └─ roi.calculate_kernel_stats()           # 统计计算

dsm.point_query(points)                          # geotiff.py:1518
  └─ rio.open(src)
  └─ src.sample(adjusted_geo_points)             # rasterio 采样
```

---

## 2. 当前设计的优点

1. **API 直观** — `idp.GeoTiff(file)` 一行构造，`.crop_polygon()`, `.save()` 等方法语义清晰
2. **polygon-first mask** — 用多边形而非栅格存储边界，精度无损，支持 geo/pixel 双坐标系
3. **affine 旋转裁剪** — 已实现核心逻辑：矩形多边形检测、`map_coordinates` 重采样、旋转 Affine 构建、QGIS 兼容显示
4. **懒加载** — `_imarray` 首次访问时才从磁盘读取，节省内存
5. **mask 仅用于保存** — 裁剪后保留完整矩形数据，mask 不破坏原始像素值，后续计算仍可访问边缘
6. **rasterio 正确使用** — IO 全部通过 rasterio，不再依赖 tifffile

---

## 3. 当前设计的问题与瓶颈

### 3.1 架构问题

| 问题                                    | 详情                                                                                                                                                                        |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **超大类（God Class）**           | 3237 行，混合 IO、裁剪、mask、坐标转换、affine、统计等多种职责                                                                                                              |
| **header dict 是冗余抽象层**      | header 中存在 `"scale"`/`"tie_point"` 等字段，同时 `profile["transform"]` 也有完整 Affine；`"height"` 同时存在于 `header` 顶层和 `profile` 子 dict 中，同步困难 |
| **每次操作都重新 `rio.open()`** | `geo2pixel`, `pixel2geo`, `point_query`, `crop_shapely_polygon` 各自独立打开文件，无法共享句柄                                                                      |
| **懒加载与显式 IO 混搭**          | 有些方法从 `self.imarray`（懒加载触发），有些直接 `rio.open(self.file_path)`，数据来源不统一                                                                            |
| **mask 计算有歧义**               | `_compute_mask()` 既可从文件 GDAL mask 读，也可从 ndarray 的 alpha/nodata 算，两者 shape 不一致时行为不明确                                                               |
| **无窗口/分块读取**               | `get_imarray()` 总是全图读取，`crop_shapely_polygon` 虽然用 `rasterio.mask.mask` 做了空间裁剪，但没有抽象的 window read 接口                                          |

### 3.2 性能瓶颈

- **全图读取** — `imarray` 属性始终触发全图 `src.read()`，大图（>4GB）很容易 OOM
- **重复 IO** — `crop_rois` 中每个 ROI 独立打开文件、独立调用 `mask.mask`
- **缺少 overview 支持** — 无金字塔层级读取
- **map_coordinates 逐波段采样** — `_prepare_affine_storage` 中多波段使用 Python for 循环逐波段采样，未向量化

### 3.3 cvtools 的问题

- `imarray_crop()` 与 `crop_shapely_polygon()` 功能重叠但实现路径完全不同（前者纯 numpy，后者用 rasterio mask）
- `poly2mask()` 不支持 polygon 带洞（仅简单多边形）
- 两者通过 `roi.crop()` 间接关联，调用者需自行选择

---

## 4. 继承 vs 组合：rasterio 集成策略评估

### 4.1 继承 rasterio DatasetReader/DatasetWriter 是否现实？

**不现实，理由如下：**

1. **rasterio DatasetReader 是 C 扩展对象** — `rasterio.io.DatasetReader` 不是纯 Python 类，它封装了 GDAL 数据集句柄。构造函数不是 Python 风格（通过 `rasterio.open()` 工厂函数创建），不支持无文件初始化。
2. **DatasetReader 生命周期与文件句柄绑定** — `with rio.open() as src:` 协议管理资源。如果在 `GeoTiff.__init__` 中打开文件并持有 `src`，需要管理上下文管理器生命周期，破坏懒加载语义。
3. **内存 ndarray 构造路径不合** — `GeoTiff(imarray=ndarray, header=header)` 无对应文件，DatasetReader 无法从此构造。
4. **多文件/多窗口场景** — 一个 EasyIDP 工作流可能涉及多个源文件（DOM + DSM + mask），继承单一 DatasetReader 无法表达。
5. **只读语义** — DatasetReader 是只读的，`save()` 需要 DatasetWriter，但两者是不同的类（非继承关系）。

**结论：继承 rasterio 类不适合 EasyIDP 的语义。**

### 4.2 组合模式更合适

推荐**组合 rasterio 的路径(Path)、profile(dict)、transform(Affine)、window(Window)、mask(mask function)**：

```python
class GeoTiff:
    """组合模式：持有 rasterio 相关对象而非继承"""
    _path: Path | None            # 文件路径
    _profile: dict                # rasterio profile（单一真相来源）
    _transform: Affine            # 快捷引用
    _crs: CRS | None
    _imarray: np.ndarray | None   # 懒加载缓存
    _mask_polygon: np.ndarray | None
    _window: Window | None        # 当前读取窗口（支持分块）
    _src: DatasetReader | None    # 可选：持久的文件句柄（with 管理）
```

`_profile` 作为**单一真相来源**，消除 header 中 `scale`/`tie_point`/`height`/`width` 的冗余。

---

## 5. 新架构建议

### 5.1 分层设计

```
┌─────────────────────────────────────────────────────┐
│  EasyIDP Public API (向后兼容层)                     │
│  idp.GeoTiff(...)                                   │
│  .crop_polygon() .save() .point_query() ...         │
├─────────────────────────────────────────────────────┤
│  GeoTiff Core (组合 rasterio 对象)                   │
│  _path, _profile, _transform, _crs, _mask_polygon   │
│  - lazy read via window                             │
│  - unified mask resolution                          │
│  - affine transform management                      │
├─────────────────────────────────────────────────────┤
│  RasterIO Adapter (薄封装)                          │
│  _open_reader() -> DatasetReader (上下文管理)        │
│  _read_window(window) -> ndarray                    │
│  _write(path, imarray, profile)                     │
├─────────────────────────────────────────────────────┤
│  rasterio / GDAL (底层)                              │
│  DatasetReader, DatasetWriter, mask, transform...    │
├─────────────────────────────────────────────────────┤
│  cvtools (纯 numpy 工具，独立使用)                   │
│  imarray_crop(), poly2mask(), rgb2gray()            │
└─────────────────────────────────────────────────────┘
```

### 5.2 窗口读取（Window Read）

替代全图懒加载，引入分块读取：

```python
class GeoTiff:
    def read(self, window: Window | None = None) -> np.ndarray:
        """按窗口读取数据，默认全图。
      
        现有 `imarray` 属性改为调用 `self.read()`。
        增加 `window` 参数支持分块处理大图。
        """
        with self._open_reader() as src:
            if window is None:
                data = src.read()
            else:
                data = src.read(window=window)
        return np.moveaxis(data, 0, -1)  # (B,H,W) -> (H,W,B)
  
    def iter_windows(self, block_size=512):
        """生成器：按 block 迭代全图窗口"""
        for row in range(0, self.height, block_size):
            for col in range(0, self.width, block_size):
                w = min(block_size, self.width - col)
                h = min(block_size, self.height - row)
                yield Window(col, row, w, h)
```

### 5.3 统一 Mask 体系

```python
class MaskType(Enum):
    NONE = "none"           # 无 mask
    POLYGON = "polygon"     # 多边形 mask（优先）
    BOOL_ARRAY = "bool"     # 布尔数组 mask
    ALPHA_BAND = "alpha"    # alpha 通道
    NODATA_VALUE = "nodata" # nodata 值

class GeoTiff:
    def resolve_mask(self) -> tuple[np.ndarray, MaskType]:
        """统一 mask 解析入口，消除当前 _compute_mask 中的歧义"""
        # 优先级：polygon > bool_array > alpha > nodata > all_valid
```

### 5.4 Affine 旋转裁剪增强

当前实现已较完整，建议：

1. **向量化多波段采样** — 用 `scipy.ndimage.map_coordinates` 的广播替代 for 循环（或使用 `rasterio.warp.reproject` 处理多波段）
2. **`_prepare_affine_storage` 支持直接读取窗口** — 当源图巨大时，不应先全图加载再采样，而应通过 geo->pixel 映射只读取旋转 ROI 覆盖区域
3. **支持非矩形多边形的最小外接矩形 (MBR) 优化** — 用户可能传入任意倾斜 ROI，自动计算 MBR 并裁剪

### 5.5 保存策略统一

当前 `save()` 根据 `_get_data_type()` 分流为 dsm/rgb/rgba/ms/msa 五条路径。建议：

```python
def save(self, path, profile_overrides=None, **kwargs):
    """保存策略由 profile 决定，而非手动分支
  
    - dtype 为 float + nodata 存在 -> DSM 风格
    - 有 alpha band -> 合并 mask 到 alpha
    - 无 alpha -> 添加 alpha band
    """
    profile = self._prepare_save_profile(profile_overrides)
    data = self._prepare_save_data(profile)
    self._write_geotiff(path, data, profile)
```

### 5.6 cvtools 重定位

`cvtools.imarray_crop()` 应用于不依赖 georeference 的纯像素裁剪场景（如 `back2raw` 流程中的原始 JPG 裁剪）。它与 GeoTiff 的地理裁剪并行存在即可，无需合并。

### 5.7 大图性能建议

| 技术                | 说明                                                            |
| ------------------- | --------------------------------------------------------------- |
| Window read         | 替代全图加载，按需读取                                          |
| Overview (金字塔)   | 利用 rasterio 的 overview 层级做低分辨率预览/快速统计           |
| 块迭代 (Tiled read) | 处理超大图时按 block 迭代，内存 footprint 可控                  |
| 并行 crop           | `crop_rois` 中复用同一次 `rio.open()` 的句柄，减少重复 open |
| 内存映射            | rasterio 底层支持 `GDAL_CACHEMAX` 等配置，大图应暴露此配置    |

---

## 6. 兼容现有 API 的迁移路径

### 6.1 阶段 1：内部重构，API 不变（v2.x）

1. **引入 `_profile` 作为单一真相来源**

   - 在 `__init__` 中从现有 `header["profile"]` 同步 `_profile`
   - 添加 property 别名：`self.height` 代理到 `self._profile["height"]`
   - 保留 `self.header` dict（标记 deprecated），内部全部改用 `_profile`
2. **提取 `_RasterIOAdapter` 内部类**

   - 封装 `_open_reader()`, `_read_window()`, `_write()`
   - 让 `geo2pixel`, `pixel2geo`, `point_query`, `crop_shapely_polygon` 通过适配器访问
   - 可选：在需要多次 IO 时复用同一个 `src` 句柄
3. **添加 `read(window)` 方法**

   - `imarray` property 改为 `return self.read()`（无窗口参数 = 全图），行为不变
   - 保留现有 `get_imarray()` 模块函数（标记为内部）
4. **保留 `header` dict 作为兼容层**

   - `header` 改为 property，从 `_profile` 动态生成
   - 移除 `scale`/`tie_point` 的独立存储，改为 property 计算自 `transform`

### 6.2 阶段 2：公开新接口（v3.0）

| 旧 API                              | 新 API                             | 说明                      |
| ----------------------------------- | ---------------------------------- | ------------------------- |
| `gtiff.header`                    | `gtiff.profile`                  | 直接暴露 rasterio profile |
| `gtiff.imarray`                   | `gtiff.read()`                   | 支持 window 参数          |
| `gtiff.get_imarray()`             | 废弃                               | 用 `gtiff.read()`       |
| `gtiff.scale`                     | `gtiff.pixel_size`               | 更清晰命名                |
| `gtiff.tie_point`                 | `gtiff.origin`                   | 语义对齐                  |
| `gtiff.crop_polygon(...)`         | `gtiff.crop(polygon, ...)`       | 统一裁剪入口              |
| `gtiff.crop_shapely_polygon(...)` | 内部方法                           | 标记 `_crop_shapely`    |
| `cvtools.imarray_crop(...)`       | 保持不变                           | numpy 工具独立            |
| - (新)                              | `gtiff.iter_windows(block_size)` | 块迭代                    |
| - (新)                              | `gtiff.reproject(target_crs)`    | rasterio warp 封装        |

### 6.3 兼容性保证

- `GeoTiff(file_path)` 构造行为不变
- `GeoTiff(imarray=ndarray, header=header_dict)` 在一个过渡版本中同时接受 `header` 和 `profile` keyword
- `.crop_polygon()`, `.save()`, `.point_query()` 等方法签名不变（可增加 keyword 参数）
- 旧代码 `dom.header["crs"]` 在过渡期内继续工作（property 代理）

### 6.4 风险与注意事项

| 风险                                     | 缓解                                                                   |
| ---------------------------------------- | ---------------------------------------------------------------------- |
| `header` dict 被外部代码直接修改       | 过渡期通过 `__getitem__`/`__setitem__` 代理同步到 `_profile`     |
| `_mask` / `_mask_polygon` 状态一致性 | 重构为单一 `_mask_source` (polygon/array/auto) + `_resolve_mask()` |
| 多波段轴序 (H,W,C) vs rasterio (B,H,W)   | 统一在适配器层做 `np.moveaxis` 转换，类内部始终保持 (H,W,C)          |
| `scipy.ndimage.map_coordinates` 性能   | 大图时改为 `rasterio.warp.reproject` 或 blocksize 分块               |

---

## 7. 总结

| 维度               | 当前状态                            | 推荐方向                                        |
| ------------------ | ----------------------------------- | ----------------------------------------------- |
| 与 rasterio 的关系 | 薄封装，每次操作独立 `rio.open()` | 组合 profile/transform/window，薄适配层         |
| Mask 体系          | polygon-first + 3 级 fallback       | 统一 `resolve_mask()` 消除歧义                |
| 裁剪               | 通过 `rasterio.mask.mask`         | 增加 window-based crop 作为补充                 |
| Affine 旋转        | 已实现核心逻辑，正确                | 向量化多波段、支持 MBR、大图分块                |
| 懒加载             | 仅全图                              | 改为 window read，支持 overview                 |
| 保存               | 5 路分支                            | profile 驱动策略，减少分支                      |
| cvtools            | numpy 层独立                        | 保持独立，不与 geotiff 耦合                     |
| 代码规模           | 3237 行 God Class                   | 拆分为 GeoTiff + _RasterIOAdapter + mask 子模块 |

**核心结论**：不应继承 rasterio 类，应组合其对象。重构重点是 (1) 引入 `_profile` 作为单一真相来源、(2) 添加 window read 和块迭代支持大图、(3) 统一 mask 解析、(4) 向量化 affine 采样，同时保持外部 API 向后兼容。
