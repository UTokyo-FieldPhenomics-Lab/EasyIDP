# EasyIDP 源码分析与重构建议报告

日期：2026-06-08

分析方式：主代理协调，4 个子代理并行分析。范围覆盖 `src/easyidp`、`tests`、`docs/python_api`、`README.md`、`pyproject.toml` 与 `.agents/references/20260608_refactor_analysis/repomix-output.xml`。

子代理分工：

- 子代理 A：`structures`、`roi.py`、`shp.py`、`jsonfile.py`、`geotools.py`。
- 子代理 B：`geotiff.py`、`cvtools.py`、`pointcloud.py`、`visualize.py`。
- 子代理 C：`reconstruct.py`、`pix4d.py`、`metashape.py`。
- 子代理 D：`__init__.py`、`data.py`、`logger.py`、文档、测试、CI、依赖。

## 1. 总体结论

EasyIDP 当前已经具备稳定的农业 ROI 工作流主干：ROI 读写、CRS 转换、DSM/点云取高程、GeoTIFF 裁剪、Pix4D/Metashape 项目解析、back2raw 投影、可视化和真实数据测试。项目的功能面较完整，测试覆盖也明显强于普通科研代码。

主要短板集中在可维护性和工具化边界：多个核心类承担过多职责，导入阶段存在网络和动态安装副作用，文档和 API 有多处不一致，几何/CRS/投影逻辑仍有不少手写分支。若未来要兼容 OpenDroneMap、DJI Terra，并提供 MCP 和 skills 支持，建议从“软件类内部混合解析、投影、IO”逐步重构为“数据模型、Parser Adapter、Projection Engine、产品 IO、工具接口”分层架构。

最高优先级修复项：

- `ROI.open()` 不支持 `.geojson`，但项目已有 `ROI.read_geojson()`。
- `shp.py` 中 `Path(shp_proj).exists` 缺少 `()`，路径存在判断错误。
- shapefile polygon 闭合判断使用 `.all()`，单轴不同时可能漏闭合。
- `jsonfile._check_file_encoding()` 返回 chardet dict，而不是 encoding 字符串。
- `GeoTiff.save()` 使用 `self._mask` 而不是 `self.mask`，可能跳过已设置掩膜。
- `PointCloud.write_point_cloud()` 无后缀路径会用旧 `file_ext` 判断，可能写错格式。
- `Metashape.read_chunk_zip()` 用 `bool("false")` 解析 enabled，结果永远为 True。
- `Pix4D/Metashape.get_photo_position()` 缓存不区分目标 CRS，可能返回错误坐标。
- `__init__.py` 导入时进行 Google 访问和动态 `pip install oss2`，对库、CI、MCP server 都不安全。
- `data.py` 使用 `ZipFile.extractall()`，缺少 Zip Slip 防护。

## 2. 九维评分

评分标准：1 为最差，5 为最好。

| 维度 | 总分 | 说明 |
|---|---:|---|
| 功能正确性 | 3/5 | 主流程可用且测试覆盖真实场景，但存在多个确定性 bug 和格式边界问题。 |
| 可读性 | 3/5 | 注释和 docstring 较多，但 ROI、GeoTiff、Pix4D、Metashape、data 模块过大，命名缩写多。 |
| 健壮性 | 2/5 | CRS、MultiPolygon、无颜色点云、损坏 XML、下载中断、并发缓存等边界处理不足。 |
| 性能 | 3/5 | 已使用 rasterio、KDTree、Metashape batch 投影等优化，但仍有整图加载、重复 open、ROI x photo 循环和高内存 batch 风险。 |
| 测试 | 3/5 | 功能测试丰富，含真实数据；但单元/集成/网络/慢测试未分层，缺少边界、并发、基准测试。 |
| 风格/规范 | 2/5 | 多个函数超过 50 行，缺少 ruff/mypy 配置，存在可变默认参数、print/logger 混用、拼写错误。 |
| 文档/注释 | 3/5 | 文档体系完整，但签名、返回值、属性名和过时 API 有多处不一致。 |
| 可维护性 | 2/5 | 现有架构不利于新增 ODM/DJI Terra Adapter，也不利于 MCP/skills 稳定调用。 |
| 并发/线程安全 | 2/5 | 无显式线程模型，存在全局 logger、全局 numpy print options、原地 CRS 修改、共享下载目录等风险。 |

## 3. 核心模块与调用链

### 3.1 ROI、矢量数据与 CRS

核心对象：`Container`、`ROI`、`shp.read_shp()`、`jsonfile.read_geojson()`、`geotools.convert_proj()`、`geotools.convert_proj3d()`。

主要调用链：

```text
ROI(path)
  -> ROI.open(path)
     -> ROI.read_shp(path)
        -> idp.shp.read_shp(path, return_proj=True)
        -> read_shp_field_schema()
        -> rename_by_fields()
     -> ROI.read_labelme_json(path)
```

GeoJSON 目前只能显式调用：

```text
ROI.read_geojson(path)
  -> idp.jsonfile.read_geojson(path, return_proj=True)
  -> self[k] = polygon
```

CRS 转换链：

```text
ROI.change_crs(target_crs)
  -> geotools.convert_proj(self.id_item, self.crs, target_crs)
     -> pyproj.Transformer
     -> axis order 判断
```

主要结论：

- ROI 作为容器、矢量 IO、CRS、高程采样、裁剪、回投影入口的聚合类，职责过重。
- shapefile 和 GeoJSON 当前主要把 geometry 转为 numpy array，MultiPolygon、holes、multipart polygon 支持弱。
- 建议引入 Shapely 作为内部 geometry 标准层，再按现有 API 输出 numpy array。
- 建议统一 `pyproj.Transformer.from_crs(..., always_xy=True)`，减少手写 axis order 逻辑。

### 3.2 GeoTiff、cvtools、PointCloud、visualize

核心对象：`GeoTiff`、`PointCloud`、`cvtools.imarray_crop()`、`cvtools.poly2mask()`、`visualize.draw_*()`。

GeoTiff 裁剪调用链：

```text
GeoTiff.crop_rois(roi)
  -> for each ROI
  -> GeoTiff.crop_polygon()
     -> shapely.geometry.Polygon
     -> GeoTiff.crop_shapely_polygon()
        -> rasterio.mask.mask(...)
        -> GeoTiff(imarray=..., header=...)
        -> set_mask_polygon()
        -> _compute_mask()
```

PointCloud 裁剪有两个重复实现：

```text
PointCloud.crop_polygon()
  -> KDTree bbox 预筛
  -> matplotlib.path.Path.contains_points
  -> ndarray

PointCloud.crop_point_cloud()
  -> self.points 全量拷贝
  -> bbox bool 过滤
  -> matplotlib Polygon.contains_points
  -> PointCloud
```

主要结论：

- GeoTiff 代码功能强，但 `save`、`crop_shapely_polygon`、`one_raw_roi2geotiff`、`back2raw2geotiff` 过长且职责混合。
- 建议更多使用 `rasterio.windows`、`rasterio.features.geometry_mask`、`rasterio.warp.reproject`。
- PointCloud 应读取 LAS CRS VLR，支持无颜色 LAS/PLY，并使用 `laspy.open(...).chunk_iterator()` 支持大点云。
- visualize 应避免全局 pyplot 状态，改为返回 `(fig, ax)` 或引入 plotter 类。

### 3.3 Pix4D、Metashape 与重建项目抽象

公共基类：`Recons`。当前保存 sensors、photos、DOM/DSM/PCD、CRS、相机位置缓存，但没有明确抽象接口。

Pix4D 调用链：

```text
Pix4D.open_project(project_path)
  -> parse_p4d_project()
  -> parse_p4d_param_folder()
  -> read_xyz()
  -> read_ccp()
  -> read_cicp()
  -> read_cam_ssk()
  -> read_pmat()
  -> build Sensor and Photo
  -> load_pcd/load_dom/load_dsm
```

Pix4D 投影：

```text
Pix4D.back2raw(roi)
  -> for each ROI
  -> back2raw_crs(points_xyz)
     -> for each photo
     -> _pmatrix_calc(points_xyz, photo)
     -> Calibration.calibrate()
     -> Sensor.in_img_boundary()
```

Metashape 调用链：

```text
Metashape.open_project(project_path)
  -> read_project_zip()
  -> read_chunk_zip()
  -> _chunk_dict_to_object()
```

Metashape batch 投影：

```text
Metashape.back2raw(roi)
  -> 临时 self.crs = roi.crs
  -> 去重所有 ROI 顶点
  -> CRS -> local/world
  -> _prepare_camera_transforms()
  -> _batch_project_to_cameras()
  -> 按 ROI 重组输出
```

主要结论：

- `Recons` 应重命名并扩展为 `ReconstructionProject` 抽象基类。
- `Photo.transform` 当前在 Pix4D 中是 3x4 projection matrix，在 Metashape 中是 4x4 camera transform，语义冲突，应拆成 `projection_matrix`、`camera_to_world`、`world_to_camera`。
- 新增 ODM/DJI Terra 时，不应复制 Pix4D/Metashape 的解析和投影逻辑，应建立 Parser Adapter 与 Projection Engine。
- `get_photo_position()` 应使用 CRS-aware cache，缓存 key 可用 `target_crs.to_string()`。

### 3.4 包级 API、数据、日志、测试与文档

包级导入链：

```text
import easyidp
  -> init_easyidp_logger()
  -> Google Drive 可用性检测
  -> Google 不可用时尝试 import oss2
  -> 缺 oss2 时动态 pip install
  -> import data/reconstruct/geotiff/pointcloud/etc
```

数据下载链：

```text
Lotus()/ForestBirds()/TestData()
  -> EasyidpDataSet.__init__()
  -> load_data()
     -> _download_data()
        -> gdown or AliyunDownloader
     -> _unzip_data()
        -> ZipFile.extractall()
```

主要结论：

- 导入 `easyidp` 不应触发网络、动态安装或 banner 输出。
- 数据集对象实例化即下载，不适合文档、测试、MCP、服务端环境。
- `logger.py` 设计较完整，但全局 `_STATE` 和 handler reset 缺少锁。
- 测试依赖真实网络和共享缓存，CI 使用 `max-parallel: 1` 暗示并发不稳定。

## 4. 第三方库替代与扩展建议

### 4.1 pyshp、shapely、geojson、pyproj

建议：

- 用 `shape.__geo_interface__` 加 `shapely.geometry.shape()` 标准化 shapefile geometry。
- 用 `shapely.geometry.shape(feature["geometry"])` 读取 GeoJSON，支持 MultiPolygon 和 holes。
- 用 `pyproj.Transformer.from_crs(src, dst, always_xy=True)` 统一 CRS 转换。
- 用 Shapely 2.x ufunc 或 prepared geometry 辅助点云、ROI 空间筛选。

收益：

- 减少自定义 geometry 判断和闭合逻辑。
- 更好支持 multipart polygon、holes、invalid polygon 修复。
- 降低 EPSG:4326 axis order 错误概率。

风险：

- Shapely geometry 到 numpy array 的兼容层需要明确外壳、内环、多 polygon 的 API 语义。
- 对大量小 polygon，Shapely 转换有额外开销，需要 benchmark。

### 4.2 rasterio

建议：

- 用 `rasterio.windows.from_bounds` 做窗口读取，避免整图加载。
- 用 `rasterio.mask.mask(filled=False)` 保留 masked array 语义。
- 用 `rasterio.features.geometry_mask` 或 `rasterize` 替代部分自定义 mask 逻辑。
- 用 `rasterio.warp.reproject` 替代 GeoTIFF 仿射重采样中的部分 `scipy.ndimage.map_coordinates`。

### 4.3 laspy、lazrs、plyfile

建议：

- 用 `laspy.open(...).chunk_iterator()` 支持大点云流式裁剪。
- 用 `las.header.parse_crs()` 读取 LAS/LAZ 内置 CRS。
- 用 `header.add_crs(crs)` 写入 CRS。
- 检查 point format 是否包含 RGB，不要假设 red/green/blue 一定存在。
- PLY 读取时支持无颜色文件，不要对 `None` 设置 dtype。

### 4.4 XML、文本和矩阵解析

建议：

- 对外部 Metashape XML 使用 `defusedxml.ElementTree`。
- Pix4D 文本解析建立 `Pix4DProjectFiles`、`read_matrix_blocks()`、`read_key_value_blocks()` 中间层。
- 矩阵/旋转可局部使用 `scipy.spatial.transform.Rotation`。
- OpenCV 或 pycolmap 可作为可选依赖评估，不建议直接成为核心强依赖。

## 5. 文档与函数一致性问题

高优先级不一致：

- `docs/python_api/geotiff.rst` 写 `easyidp.pointcloud.GeoTiff`，实际应为 `easyidp.geotiff.GeoTiff`。
- `docs/python_api/geotiff.rst` 列出 `tifffile_crop`、`save_geotiff`，源码当前未定义这些函数。
- `docs/python_api/visualize.rst` 未列出源码和测试已使用的 `show_subplots`。
- `GeoTiff.__init__` docstring 参数仍写 `tif_path`、`Transparent_layer`，实际签名是 `file_path, imarray, header, mask`。
- `cvtools.imarray_crop` 返回文档写 `(W, H, 3)`，实际是 `(height, width[, bands])`。
- `PointCloud.write_point_cloud` 文档说支持 ply/las/laz，但无后缀路径存在写错格式风险。
- `jsonfile.read_geojson()` 的 `name_field` 文档仍说 shp fields，应改为 GeoJSON properties。
- `Pix4D.back2raw_crs()` 文档写 `distortion_correct`，签名使用 `distort_correct`。
- `Sensor.in_img_boundary()` 支持 `ignore="as_point"`，文档和错误信息未列。
- `Metashape.back2raw()` 的 `ignore/log` 文档说明未实现，但通过 `**kwargs` 静默吞掉。
- `docs/python_api/data.rst` 示例使用 `lotus.metashape.proj`，源码是 `lotus.metashape.project`。
- README 依赖列表提到 `tifffile`、`lasio`，与 `pyproject.toml` 不一致。

建议：

- 建立文档一致性测试，至少检查 autosummary 中的对象能 import。
- 将数据集 manifest 作为文档表格来源，避免手写路径表过期。
- 将大型下载示例改成 `download=False` 或显式 `.ensure_available()`。

## 6. 命名与可读性专项

建议重命名：

| 当前名称 | 建议名称 | 理由 |
|---|---|---|
| `Recons` | `ReconstructionProject` | 基类缩写不直观，未来作为 Adapter 基座应语义完整。 |
| `ProjectPool` | `ReconstructionProjectCollection` | Pool 含义不清，且当前未实现。 |
| `Photo.transform` | `projection_matrix` 或 `camera_to_world` | Pix4D 与 Metashape 中语义不同，应拆分字段。 |
| `shp_dict` | `geometry_dict` 或 `coord_dict` | `geotools.convert_proj()` 不只处理 shp。 |
| `keyring` | `selected_field_names` | 当前名称不表达字段名集合。 |
| `field_id` | `field_ids` 或 `name_field_ids` | 可能是 int 或 list，单复数混淆。 |
| `polygon_hv` | `polygon_xy` 或 `polygon_colrow` | hv 不直观，需区分地理坐标和像素坐标。 |
| `points_hv` | `points_xy` 或 `points_colrow` | 同上。 |
| `cls` | `colors` | 点云颜色缩写易误解为 class。 |
| `nms` | `normals` | 缩写降低可读性。 |
| `crop_polygon` in PointCloud | `query_points_in_polygon` | 当前返回 ndarray，不是 PointCloud。 |
| `crop_point_cloud` | `crop_to_pointcloud` | 与返回 ndarray 的函数区分。 |
| `poly2mask` | `polygon_to_mask` | 更清晰，可保留旧名作 alias。 |
| `EasyidpDataSet` | `EasyIDPDataset` | 品牌缩写统一。 |
| `ReconsProj` | `ReconsProjectPaths` | 实际是路径集合，不是项目对象。 |
| `url_checker` | `is_url_reachable` | bool 返回值应使用谓词命名。 |
| `download_auth` | `confirm_download_cost` | 实际职责是确认下载费用。 |
| `logged_input` | `prompt_user` 或 `_logged_input` | 交互函数不宜作为普通包级 API。 |

建议优先拆分的超过 50 行或职责不单一函数：

- `ROI.read_shp()`：拆为 `_reset_from_shp()`、`_load_attrs()`、`rename_by_fields()`。
- `ROI.get_z_from_dsm()`：拆为 `_prepare_query_polygons()`、`_sample_dsm_face()`、`_sample_dsm_points()`、`_assign_z_values()`。
- `ROI.get_z_from_pcd()`：与 DSM 重复，应抽象 `ZSampler`。
- `shp.read_shp()`：拆为 `_resolve_shp_crs()`、`_iter_shape_records()`、`_normalize_geometry()`。
- `jsonfile.read_geojson()`：拆为 `_read_geojson_crs()`、`_get_geojson_fields()`、`_feature_key()`、`_feature_geometry()`。
- `GeoTiff.save()`：拆为 `_normalize_save_path()`、`_prepare_save_array()`、`_apply_mask_to_array()`、`_write_raster()`。
- `GeoTiff.crop_shapely_polygon()`：拆为 `_mask_crop_with_rasterio()`、`_header_from_profile()`、`_build_cropped_geotiff()`。
- `one_raw_roi2geotiff()`：拆为 `_load_raw_image()`、`_crop_raw_with_buffer()`、`_estimate_output_grid()`、`_warp_raw_to_grid()`。
- `back2raw2geotiff()`：拆为 `_group_back2raw_by_image()`、`_estimate_worker_count()`、`_run_geotiff_tasks()`。
- `Pix4D.open_project()`：拆为 `parse_project_files()`、`build_sensor()`、`build_photos()`、`load_outputs()`。
- `Metashape.back2raw()`：拆为 `prepare_roi_points()`、`project_rois_batch()`、`reconstruct_projection_results()`。
- `read_chunk_zip()`：拆为 `parse_chunk_doc()`、`parse_frame_docs()`、`resolve_photo_paths()`。
- `AliYunDownloader.download_auth()`：拆为 `build_cost_notice()`、`prompt_confirmation()`、`confirm_download_cost()`。
- `EasyidpDataSet._download_data()`：拆为 `choose_downloader()`、`download_archive()`。

接口草案：

```python
@dataclass
class ROIFeature:
    name: str
    geometry: shapely.Geometry
    attrs: dict[str, Any]
    crs: pyproj.CRS | None = None


@dataclass
class ROICollection:
    features: list[ROIFeature]
    crs: pyproj.CRS | None = None
    source: Path | None = None
```

```python
class ReconstructionProject(ABC):
    software: str
    crs: pyproj.CRS | None
    sensors: Container[Sensor]
    photos: Container[Photo]

    @abstractmethod
    def open_project(self, project_path: PathLike, **kwargs) -> None:
        ...

    @abstractmethod
    def world_to_local(self, points: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def project_local_to_photo(
        self,
        points_local: np.ndarray,
        photo: Photo,
        *,
        distortion_correct: bool = True,
    ) -> np.ndarray:
        ...
```

```python
class ZSampler(Protocol):
    crs: pyproj.CRS | None

    def sample_face(self, polygon_xy: np.ndarray, kernel: str, buffer: float) -> float:
        ...

    def sample_points(self, points_xy: np.ndarray, kernel: str, buffer: float) -> np.ndarray:
        ...
```

## 7. 性能与复杂度热点

### 7.1 `shp.read_shp()`

复杂度：时间 `O(N + V)`，空间 `O(N + V)`。`N` 是 feature 数，`V` 是总顶点数。

瓶颈：全量 polygons 和 records 入内存，geometry 转 numpy 需要拷贝。

优化策略：

- 提供 generator API：`iter_shp_features(path) -> Iterator[FeatureGeometry]`。
- 支持 bbox/filter，避免读取全部 feature。
- 对比 `shape.points` 与 `shapely.shape(shape.__geo_interface__)` 的性能和完整性。

基准：1k、10k、100k polygons，单 polygon 5、100、1000 vertices。

### 7.2 `jsonfile.read_geojson()`

复杂度：时间 `O(N + V)`，空间 `O(N + V)`。

瓶颈：`geojson.load()` 全量加载，大文件内存压力高，properties 只从第一条 feature 推断字段。

优化策略：

- 大文件可选 `ijson` 流式解析。
- 统一 shp/geojson 的 key 生成和 duplicate 检测逻辑。
- 用 Shapely 标准化 geometry。

基准：10MB、100MB、1GB GeoJSON，Polygon 与 MultiPolygon 分别测试。

### 7.3 `geotools.convert_proj()`

复杂度：时间 `O(V)`，空间 `O(V)`。

瓶颈：每个 item 重复判断 axis order，按 polygon 分批 transform 造成调用开销。

优化策略：

- 使用 `Transformer.from_crs(..., always_xy=True)`。
- axis order 计算移出循环。
- 对大量小 polygon，将所有点 concatenate 后一次转换，再 split 回去。

### 7.4 `ROI.rename_by_fields()`

复杂度：当前重复 key 检测最坏 `O(N^2)`。

优化策略：

```python
from collections import Counter

duplicates = [key for key, count in Counter(generated_keys).items() if count > 1]
```

### 7.5 `GeoTiff.get_imarray()` 与裁剪

复杂度：整图读取时间和空间 `O(H * W * B)`。

瓶颈：大图整图入内存，每个 ROI 裁剪重复打开 dataset。

优化策略：

- 增加 window read API。
- `crop_rois` 内只打开一次 rasterio dataset。
- 相邻 ROI 可合并 window 或按 tile cache。

### 7.6 GeoTiff 仿射转换

复杂度：时间 `O(out_H * out_W * B)`，空间约为多个完整 float grid 加输出。

瓶颈：`np.mgrid` 和 `map_coordinates` 创建多个大数组。

优化策略：

- 分块生成坐标并写输出。
- 用 `rasterio.warp.reproject` 或 GDAL warp。
- 插值阶数可配置，分类/掩膜数据用 nearest。

### 7.7 PointCloud 裁剪

复杂度：KDTree 构建 `O(N log N)`，查询平均 `O(log N + K)`。

瓶颈：`PointCloud.points` 每次访问返回 `_points + _offset`，造成 `O(N)` 分配；`crop_point_cloud()` 多次调用。

优化策略：

- 函数内缓存 `points = self._points + self._offset`。
- 只为裁剪计算 XY。
- `crop_point_cloud()` 复用 KDTree 版 `crop_polygon()`。
- 大点云用 laspy chunk 流式裁剪。

### 7.8 Pix4D back2raw

复杂度：时间 `O(R * C * P)`，空间主要为输出。

瓶颈：ROI x photo Python 双层循环，每次构造齐次坐标。

优化策略：

- stack 所有 photo 的 3x4 projection matrix，使用 `einsum` 批量投影。
- 对 ROI 顶点去重，复用 Metashape batch 思路。
- 用相机位置、距离阈值或视锥粗筛候选照片。

基准：`R=10/1000/10000`，`C=100/300/1000`，`P=5`。

### 7.9 Metashape batch back2raw

复杂度：时间 `O(C * U)`，空间 `O(C * U)`。`U` 是唯一点数。

瓶颈：一次性创建 `uv` 和 `valid`，大项目内存压力高。

优化策略：

- 按照片块或点块分块 batch。
- ROI bbox/中心和相机位置预筛选。
- sparse output，避免保存完整 `C * U` 矩阵。

### 7.10 数据下载与测试

复杂度：下载 `O(S)`，解压 `O(U)`，zip entry 创建 `O(N)`。

瓶颈：导入时网络检测、真实下载测试、共享缓存目录。

优化策略：

- 下载时 lazy check，不在 import 阶段访问网络。
- 使用 `.part` 临时文件和原子 rename。
- 增加 sha256，避免重复下载。
- 将 `network`、`integration`、`slow` 测试分离。

## 8. 并发与线程安全审查

当前项目没有高并发服务代码，但工具化和 MCP 场景会暴露共享状态风险。

主要风险：

- `Container._btf_print()` 修改全局 `np.set_printoptions()`，多线程打印会互相污染。
- `ROI.change_crs()`、`get_z_from_dsm()`、`get_z_from_pcd()` 会原地修改 ROI 坐标和 CRS。
- `Metashape.back2raw()`、`get_photo_position()` 临时修改 `self.crs`，异常时缺少 `finally` 恢复。
- `_photo_position_cache` 不区分 CRS，且没有锁。
- `Photo.position` 在查询函数中被写入，有隐藏副作用。
- `logger.setup_logger()` 修改全局 handler 和 `_STATE`，缺少锁。
- `DuplicateThrottleFilter` 内部 `_last_msg` 和 `_last_times` 无锁。
- `__init__.py` 全局 `aliyun_down` 懒初始化无锁。
- 下载、解压、删除共享用户目录无文件锁。
- `visualize.py` 使用全局 `matplotlib.pyplot`，多线程绘图不安全。
- rasterio dataset 当前多用 context manager 是安全的，但不应跨线程或进程共享打开的 dataset。

修复建议：

- 使用 `with np.printoptions(...)` 替代全局 `np.set_printoptions()`。
- 对会修改状态的方法增加 `_inplace` 命名，默认提供纯函数式 `to_crs()`、`with_z_from_dsm()`。
- `get_photo_position()` 不写 `Photo.position`，或将写入变为显式参数。
- 缓存 key 改成 `(target_crs.to_string(), project_revision)`。
- `Metashape.back2raw()` 使用 `try/finally`，更优是完全不修改 `self.crs`。
- `setup_logger()` 和 `DuplicateThrottleFilter` 增加 `threading.RLock`。
- 数据下载、解压、删除使用文件锁和原子写入。
- MCP 和服务端场景默认 `progress=False`、`interactive=False`。

并发测试建议：

- 多线程同时调用 `repr(roi)`，检查 numpy print options 不泄露。
- 连续和并发调用 `get_photo_position(EPSG:4326)` 与 `get_photo_position(EPSG:32654)`，检查缓存不串 CRS。
- 构造异常 ROI，调用 `Metashape.back2raw()` 后检查 `self.crs` 是否恢复。
- 多进程同时下载同一数据集，检查文件锁和最终文件完整性。
- 多线程访问同一 `GeoTiff.imarray` 和 `mask`，检查结果一致。
- 并发写同一路径时明确报错或通过文件锁序列化。

## 9. 面向 ODM、DJI Terra、MCP 和 skills 的架构基座

建议目标架构：

```text
easyidp.geometry
  ROIFeature
  ROICollection
  Shapely adapters
  geometry validation

easyidp.crs
  CRSResolver
  TransformerCache
  always_xy policy

easyidp.io
  ShpReader/ShpWriter
  GeoJSONReader/GeoJSONWriter
  LabelmeReader

easyidp.raster
  RasterDataset
  RasterCropper
  RasterMask
  RasterWriter

easyidp.pointcloud
  PointCloudData
  PointCloudReader
  PointCloudCropper
  PointCloudWriter

easyidp.reconstruction
  ReconstructionProject
  ReconstructionBundle
  Sensor
  Photo
  CameraModel
  ProjectionEngine

easyidp.reconstruction.adapters
  Pix4DAdapter
  MetashapeAdapter
  ODMAdapter
  DJITerraAdapter
```

Adapter 输入输出：

```python
@dataclass
class ReconstructionBundle:
    label: str
    crs: pyproj.CRS | None
    reference_crs: pyproj.CRS | None
    sensors: list[Sensor]
    photos: list[Photo]
    products: ReconstructionProducts
    metadata: dict[str, Any]
```

自动识别：

```python
PROJECT_ADAPTERS = {
    "pix4d": Pix4DAdapter,
    "metashape": MetashapeAdapter,
    "odm": ODMAdapter,
    "dji_terra": DJITerraAdapter,
}


def open_reconstruction_project(path, software="auto", **kwargs):
    adapter_cls = detect_adapter(path) if software == "auto" else PROJECT_ADAPTERS[software]
    return adapter_cls.open(path, **kwargs)
```

ODM 适配重点：

- `odm_orthophoto/odm_orthophoto.tif`。
- `odm_dem/dsm.tif`。
- `opensfm/reconstruction.json`。
- `camera_models.json`、shots、OpenSfM camera models。
- OpenSfM 内部 ENU/topocentric 到项目 CRS 的映射。

DJI Terra 适配重点：

- DOM/DSM GeoTIFF。
- LAS/LAZ point cloud。
- 相机姿态和空三结果 parser。
- DJI 相机内参和畸变模型映射。
- 版本差异较大，路径解析必须插件化。

MCP/skills 友好 API：

```python
def inspect_project(path: str) -> dict:
    ...


def validate_project(project: ReconstructionProject) -> list[Diagnostic]:
    ...


def project_roi_to_raw(
    project_path: str,
    roi_geojson: dict,
    *,
    software: str = "auto",
    chunk_id: str | int | None = None,
    max_images: int | None = None,
) -> dict:
    ...
```

MCP 工具建议：

- `roi.read`
- `roi.validate`
- `roi.to_crs`
- `roi.sample_z`
- `raster.crop_polygon`
- `pointcloud.crop_polygon`
- `recons.inspect`
- `recons.back_project_roi`
- `visualize.draw_roi`
- `data.list_datasets`
- `data.download_dataset`

MCP 设计原则：

- 导入零副作用。
- 不直接返回大 ndarray，返回文件路径、统计摘要和结构化 warning。
- 所有文件写入支持 dry-run 和 overwrite policy。
- 所有交互参数化，禁止底层调用 `input()`。
- 错误返回稳定 code，便于 agent 处理。

## 10. 阶段性重构路线图

### Phase 0：风险止血

- 修复 `.geojson` open、`Path.exists()`、polygon 闭合、encoding fallback、`GeoTiff.save()` mask、`PointCloud.write_point_cloud()` 后缀、Metashape enabled、photo position CRS cache。
- 移除 import 阶段动态安装 `oss2`。
- 增加环境变量 `EASYIDP_NO_NETWORK=1` 和 `EASYIDP_QUIET=1`。
- `user_data_dir()` 改为 `Path.mkdir(parents=True, exist_ok=True)`。
- `_unzip_data()` 增加 Zip Slip 防护。

### Phase 1：测试分层与 CI

- 添加 pytest markers：`unit`、`integration`、`network`、`slow`。
- 真实下载测试改为 `network` marker，默认 PR 不跑。
- `tests/out` 写入迁移到 `tmp_path`。
- CI 增加 `ruff check`、`ruff format --check`、`mypy`。
- 文档构建 job 单独运行，且禁用网络。

### Phase 2：ROI 与 geometry 重构

- 引入 `ROIFeature` 和 `ROICollection`。
- 建立 Shapely geometry adapter。
- 保持现有 `ROI[...] -> np.ndarray` 兼容输出。
- shp/geojson/labelme reader 独立成 IO 层。
- CRS 转换统一 `always_xy=True`。

### Phase 3：Raster 与 PointCloud 重构

- GeoTiff 拆成 dataset、cropper、mask、writer。
- 裁剪使用 rasterio windows 和 geometry_mask。
- PointCloud 读写和裁剪拆分，支持 LAS CRS 和 chunk iterator。
- visualize 改为无全局状态的 plotter 函数或类。

### Phase 4：Reconstruction Adapter 基座

- `Recons` 迁移为 `ReconstructionProject`。
- Parser 与 Project 分离，产出 `ReconstructionBundle`。
- Projection Engine 独立，Pix4D/Metashape 只提供相机模型、pose 和坐标转换。
- 拆分 `Photo.transform` 语义。
- 引入 ODM 和 DJI Terra Adapter scaffold。

### Phase 5：MCP 和 skills 支持

- 提供无交互、零副作用、结构化 JSON API。
- 新增 `docs/mcp/` 和 `docs/agent_api/`。
- 提供 dataset manifest 和 project inspection API。
- 将常用工作流包装为 skills：项目检查、ROI 投影、数据下载、文档一致性检查、性能基准。

## 11. 子代理摘要

### 子代理 A：ROI、矢量、CRS

结论：主流程完整，但 ROI 类职责过重，geometry 基于 numpy array 导致 MultiPolygon、holes、multipart 支持弱。应尽快修复 `.geojson open`、`Path.exists()`、闭合判断、encoding fallback、`Container.__delitem__`，并引入 Shapely 与 `pyproj always_xy=True`。

评分：功能正确性 3，可读性 3，健壮性 2，性能 3，测试 3，风格/规范 2，文档/注释 3，可维护性 2，并发/线程安全 2。

### 子代理 B：GeoTiff、cvtools、PointCloud、visualize

结论：GeoTiff 和 PointCloud 功能强，但大函数多且热路径存在内存和重复 IO 问题。优先修复 `GeoTiff.save()` 掩膜、`PointCloud.write_point_cloud()` 后缀、无颜色点云、文档过期项，并逐步使用 rasterio windows、LAS chunk iterator、无全局 pyplot 的绘图接口。

评分：功能正确性 3，可读性 2，健壮性 3，性能 3，测试 4，风格/规范 2，文档/注释 3，可维护性 2，并发/线程安全 2。

### 子代理 C：Pix4D、Metashape、reconstruct

结论：Pix4D/Metashape 已有完整项目解析和投影能力，Metashape batch 是重要性能优化。但当前没有足够清晰的 Reconstruction Adapter 基座，`Photo.transform` 语义冲突，缓存不区分 CRS，临时修改 `self.crs` 非线程安全。应优先修复 enabled 解析、CRS cache、batch `ignore` 行为差异，并拆分 Parser、Project、Projection Engine。

评分：功能正确性 3.5，可读性 3，健壮性 2.5，性能 3.5，测试 3.8，风格/规范 2.8，文档/注释 3.2，可维护性 2.8，并发/线程安全 2。

### 子代理 D：包级 API、data、logger、docs、tests

结论：最大风险是导入时副作用、数据下载副作用、真实网络测试、共享缓存和文档/API 不一致。建议优先移除 import 阶段网络检测和动态 pip install，重构数据下载 API，加入测试分层、ruff/mypy/pytest 配置，并为 MCP 提供零副作用、非交互、结构化接口。

评分：功能正确性 3，可读性 3，健壮性 2，性能 3，测试 2，风格/规范 2，文档/注释 3，可维护性 2，并发/线程安全 2。

## 12. 建议立即新增的测试

- `Container.__delitem__` 删除 int 和 str 后索引一致性。
- `ROI.open("xxx.geojson")`。
- shapefile MultiPart、holes、empty shape、缺失 `.prj`、未闭合 polygon 单轴不同。
- GeoJSON MultiPolygon、holes、properties 字段不一致、非 UTF-8 fallback。
- EPSG:4326 CRS 转换 always_xy 和单点转换。
- `GeoTiff.save()` 在只调用 `set_mask_polygon()` 后能正确应用 mask。
- `PointCloud.write_point_cloud("out")` 后缀推断正确。
- 无颜色 PLY/LAS/LAZ 读取。
- Metashape `enabled="false"` chunk 解析。
- `get_photo_position()` 不同 CRS cache 不串。
- `Metashape.back2raw()` 异常后 CRS 恢复。
- 数据下载 Zip Slip 防护。
- 多线程 logger setup 和 duplicate throttle。
- 文档 autosummary 对象可 import。

## 13. 最小可执行改进顺序

1. 修 bug：矢量 IO、GeoTiff mask、PointCloud 后缀、Metashape enabled、CRS cache、import 副作用、Zip Slip。
2. 建测试：为上述 bug 每项补回归测试，并引入 `unit/integration/network/slow` markers。
3. 建规范：添加 ruff、mypy、pytest 配置和 CI job。
4. 拆 data：下载器、dataset spec、测试数据路径、交互确认分层。
5. 拆 ROI：geometry adapter、IO reader/writer、CRS transformer、ZSampler。
6. 拆 raster/pointcloud：窗口读取、mask、writer、chunk pointcloud。
7. 拆 reconstruct：Parser Adapter、Projection Engine、统一 `ReconstructionProject`。
8. 增 ODM/DJI Terra scaffold：先实现 project inspect 和产品路径发现，再实现投影。
9. 增 MCP/skills：稳定 JSON schema、dry-run、结构化错误、无大数组返回。
