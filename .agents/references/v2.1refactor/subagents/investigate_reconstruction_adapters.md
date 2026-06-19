# User Prompt

你在 /home/crest/Documents/Github/EasyIDP 中做只读代码审查，不要修改文件。项目索引在 repomix-output.xml，源码重点在 src/easyidp。请详细分析以下范围：src/easyidp/reconstruct.py、src/easyidp/pix4d.py、src/easyidp/metashape.py，以及对应 tests/test_reconstruct.py、tests/test_pix4d.py、tests/test_metashape.py、docs/python_api 下相关 rst。

目标：

1. 梳理核心功能和主要类/函数调用链，尤其 Recons、Photo/Sensor/Calibration、Pix4D/Metashape 项目解析、坐标/投影、back2raw/forward/backward 流程。
2. 判断是否可抽象为 ReconstructionProject adapter 基类，以兼容 OpenDroneMap、DJI Terra；也评估能否用第三方库替代 XML/文本/矩阵解析逻辑。
3. 检查文档与函数签名/行为一致性。
4. 对功能正确性、可读性、健壮性、性能、测试、风格/规范、文档/注释、可维护性、并发/线程安全九个维度各给 1-5 分，并说明理由。
5. 命名与可读性专项：列出不良命名、建议替代名、理由；列出超过 50 行或职责不单一函数，给重构切分建议和接口草案。
6. 性能复杂度：对热点函数给 Big O、瓶颈、2-3 个优化策略、基准测试建议，重点关注按 ROI x photo 的投影循环、路径解析、重复坐标转换。
7. 并发与线程安全：若无并发代码，也要说明 project 对象共享状态、副作用、缓存策略相关风险和测试思路。
8. 给出面向 ODM/DJI Terra 兼容基座和 MCP/skills 支持的模块化重构建议。

返回中文结构化 Markdown，包含具体文件/函数名和尽量具体的行号或定位描述。不要写文件，只返回结果。

# EasyIDP 只读代码审查报告

审查范围：

- `/home/crest/Documents/Github/EasyIDP/src/easyidp/reconstruct.py`
- `/home/crest/Documents/Github/EasyIDP/src/easyidp/pix4d.py`
- `/home/crest/Documents/Github/EasyIDP/src/easyidp/metashape.py`
- `/home/crest/Documents/Github/EasyIDP/tests/test_reconstruct.py`
- `/home/crest/Documents/Github/EasyIDP/tests/test_pix4d.py`
- `/home/crest/Documents/Github/EasyIDP/tests/test_metashape.py`
- `/home/crest/Documents/Github/EasyIDP/docs/python_api/{reconstruct,pix4d,metashape}.rst`
- 对应 `docs/python_api/autodoc/*.rst`

说明：本次审查只读，未修改文件。项目索引未在 `.agents/references/**/repomix-output.xml` 找到，但根目录存在 `/home/crest/Documents/Github/EasyIDP/repomix-output.xml`。

---

## 1. 核心功能与调用链梳理

### 1.1 公共抽象层：`reconstruct.py`

核心对象位于 `/home/crest/Documents/Github/EasyIDP/src/easyidp/reconstruct.py`。

#### `Recons`，行 25-157

作用：Pix4D 与 Metashape 项目的共享基类。

主要状态：

- `label`：项目或 chunk 名。
- `meta`：项目元信息。
- `enabled`：项目或 chunk 是否可用。
- `sensors`：`idp.Container`，存储 `Sensor`。
- `photos`：`idp.Container`，存储 `Photo`。
- `_world_crs`：地心坐标 CRS。
- `_dom` / `_dsm` / `_pcd`：输出产品。
- `_crs`：项目 CRS。
- `_photo_position_cache`：相机位置缓存。

主要职责：

- 管理 DOM / DSM / PCD 加载属性。
- CRS 修改时清空相机位置缓存，见行 80-89。
- 目前不定义统一的 `open_project` / `back2raw` / `get_photo_position` 抽象接口。

#### `Sensor`，行 159-287

作用：相机模型。

关键字段：

- `id`, `label`, `type`
- `width`, `height`
- `w_mm`, `h_mm`
- `pixel_width`, `pixel_height`, `pixel_size`
- `focal_length`
- `calibration`

关键方法：

- `in_img_boundary()`，行 192-287
  判断 ROI 投影点是否落在影像边界内；支持 `ignore=None/"x"/"y"/"as_point"`。

问题点：

- 文档与错误信息只列 `None/'x'/'y'`，但代码支持 `"as_point"`，行 274-284。
- `ignore="x"` / `"y"` 会原地修改传入数组，存在副作用。
- 边界判断使用 `x_max > w` / `y_max > h`，像素最大索引通常应是 `w - 1` / `h - 1`，需确认坐标定义。

#### `Photo`，行 289-374

作用：单张影像的内外参、路径、传感器引用。

关键字段：

- `id`, `path`, `_path`, `label`
- `sensor_id`, `sensor`
- `enabled`
- `cam_matrix`, `location`, `rotation`
- `transform`
- `translation`
- `position`
- `master_id`

问题点：

- `_img_exists` 装饰器行 357-367 条件疑似写反：
  `if self.path != "" or not os.path.exists(self.path): raise FileNotFoundError`
  只要 `path` 非空就报错，应为 `if self.path == "" or not os.path.exists(self.path)`。目前相关方法被注释，暂未暴露。

#### `Calibration`，行 376-611

作用：镜头畸变模型。

调用链：

```text
Calibration.calibrate()
  -> _calibrate_pix4d_frame()
  -> _calibrate_metashape_frame()
```

Pix4D：

- 行 500-537，基于 Pix4D frame 模型。
- 输入 `u, v` 实际是无畸变或投影坐标。
- 输出原始图像像素坐标。

Metashape：

- 行 539-611，支持批量 `xh/yh` 输入。
- 使用 `k1-k4`, `t1/t2` 作为 Metashape 的 `p1/p2`，`b1/b2` 修正。
- 支持 1D 或 2D 数组。

问题点：

- `p1-p4` 在 `__init__` 中只是 `self.p1 = self.t1` 的数值拷贝，后续 `t1` 更新不会同步到 `p1`。文档列出 `p1-p4`，但实际计算使用 `t1-t4`。
- 行 497 错误信息为 `not {self.type}`，因多了一层花括号，实际不会插值。
- `calibrate()` 分发依赖 `software` 字符串，不利于扩展 ODM / DJI Terra / OpenSfM。

#### 公共工具函数

##### `sort_img_by_distance()`，行 646-759

作用：根据 ROI 中心到相机位置距离排序和过滤 `back2raw` 结果。

调用链：

```text
sort_img_by_distance()
  -> recons.get_photo_position(to_crs=roi.crs)
  -> ROI bbox center
  -> broadcast 计算 ROI x camera 距离矩阵
  -> 对每个 ROI 已有候选影像排序/截断
  -> 可选 save_back2raw_json_and_png()
```

特点：

- 距离矩阵一次性向量化，复杂度 `O(R * C)` 空间也为 `O(R * C)`。
- ROI 中心用 bbox 中心，不是 polygon centroid。

##### `save_back2raw_json_and_png()`，行 762-847

作用：保存 JSON 与裁剪 PNG。

问题点：

- 名称表明只保存 back2raw，但实际还裁剪图像。
- 会创建目录、读取原图、写 PNG，属于 IO 重函数。
- `tqdm(desc=f"Processing image [{img_name}]")` 在循环变量未绑定前使用 `img_name`，行 829-831，运行时可能引用上一次外层变量或报错风险；应改为固定描述。
- 文档写 `save_folder : str`，代码接受 `Path`，行 797-799。

---

### 1.2 Pix4D 项目解析与投影流程

文件：`/home/crest/Documents/Github/EasyIDP/src/easyidp/pix4d.py`

#### `Pix4D`，行 12-1009

初始化调用链：

```text
Pix4D.__init__()
  -> Recons.__init__()
  -> self.software = "pix4d"
  -> 若 project_path 不为空：
       open_project(project_path, raw_img_folder, param_folder)
```

#### `Pix4D.open_project()`，行 127-327

核心调用链：

```text
open_project()
  -> parse_p4d_project()
       -> parse_p4d_param_folder()
  -> read_xyz()        # offset
  -> read_ccp()        # 每张照片内外参
  -> read_cicp()       # 标定内参
  -> read_cam_ssk()    # sensor 元数据
  -> read_pmat()       # Pix4D pmatrix
  -> 构造 Sensor
  -> 构造 Photo
  -> 读取 CRS
  -> load_pcd/load_dom/load_dsm
```

核心映射：

- Pix4D offset：`self.meta["p4d_offset"]`，行 171-173。
- Sensor：默认只支持一个相机模型，行 213-258。
- Photo：
  - `transform` 保存 3x4 pmatrix，行 293。
  - `cam_matrix`, `location`, `rotation` 从 CCP 读取，行 296-298。
- CRS：
  - `self.crs` 与 `_proj_crs` 都来自 `*_wkt.prj`，行 311-314。

问题点：

- 行 302-306 warning 字符串中 `"[{raw_img_folder}]"` 不是 f-string，路径不会插值。
- `raw_img_folder` 用 `os.listdir()` 做线性查找，行 265-290，对大量照片成本较高。
- Pix4D 只支持单 Sensor，扩展多相机或多光谱较弱。

#### Pix4D back2raw 流程

单 ROI：

```text
Pix4D.back2raw_crs(points_xyz)
  -> points_xyz - p4d_offset
  -> 遍历 self.photos
       -> _pmatrix_calc(points_xyz, photo)
            -> 追加齐次坐标
            -> dot photo.transform.T
            -> 除以 z 得到 u/v
            -> calibration.calibrate(u, v) 可选畸变
       -> sensor.in_img_boundary()
  -> {photo.label: coords}
```

关键位置：

- `back2raw_crs()`：行 522-609。
- `_pmatrix_calc()`：行 489-520。
- `_external_internal_calc()`：行 441-487，标记 deprecated 且 “seems not correct”。

多 ROI：

```text
Pix4D.back2raw(roi)
  -> 遍历 roi.items()
  -> 校验 shape[1] == 3
  -> back2raw_crs()
  -> 可选 save_back2raw_json_and_png()
```

位置：行 611-701。

复杂度：

- `R` 个 ROI，平均 `P` 个顶点，`C` 张照片。
- 当前复杂度 `O(R * C * P)`，每次 ROI 都遍历全部照片。
- 内存低，但 Python 循环开销大。

#### Pix4D 相机位置

```text
get_photo_position()
  -> enabled photos
  -> p.location + p4d_offset
  -> 如目标 CRS 与 _proj_crs 不同，convert_proj3d()
  -> 写入 p.position
  -> 缓存 _photo_position_cache
```

位置：行 703-783。

问题点：

- 缓存只按单个结果保存，不区分 `to_crs`。若先 `get_photo_position(to_crs=EPSG:4326)`，再不刷新地请求默认 CRS，可能返回错误缓存。
- 返回 `out.copy()` 是浅拷贝，内部 ndarray 仍共享。

#### Pix4D 文件解析函数

- `_match_suffix()`，行 1016-1053
- `parse_p4d_param_folder()`，行 1056-1202
- `parse_p4d_project()`，行 1205-1379
- `read_xyz()`，行 1382-1429
- `read_pmat()`，行 1432-1520
- `read_cicp()`，行 1523-1610
- `read_ccp()`，行 1613-1777
- `read_campos_geo()`，行 1780-1853
- `read_cam_ssk()`，行 1856-1964

主要问题：

- `parse_p4d_project()` 行 1295-1304：`param_folder` 在行 1296 被赋默认值后，`param_folder is None` 分支不可达，错误信息逻辑不清。
- `_match_suffix()` 返回字符串，其他路径多用 `Path`，类型不一致。
- 文本解析多依赖固定空格位置，例如 `read_cicp()` 行 1604-1609、`read_cam_ssk()` 行 1939-1963，格式稍变易失败。
- `read_pmat()` 对只有单行时 `np.loadtxt` 可能返回 1D，`pmat_nb[i, :]` 会失败。

---

### 1.3 Metashape 项目解析与投影流程

文件：`/home/crest/Documents/Github/EasyIDP/src/easyidp/metashape.py`

#### `Metashape`，行 16-1386

初始化调用链：

```text
Metashape.__init__()
  -> Recons.__init__()
  -> self.software = "metashape"
  -> self.transform = ChunkTransform()
  -> open_project(project_path, chunk_id)
  -> 可选 change_photo_folder()
```

#### 项目打开流程

```text
open_project()
  -> _open_whole_project()
       -> _check_is_software()
       -> _split_project_path()
       -> read_project_zip()
       -> read_chunk_zip(..., return_label_only=True)
       -> 构建 chunk id/label 映射
  -> open_chunk()
       -> read_chunk_zip()
       -> _chunk_dict_to_object()
```

关键位置：

- `open_project()`：行 165-202。
- `open_chunk()`：行 203-290。
- `_open_whole_project()`：行 291-360。
- `_chunk_dict_to_object()`：行 361-391。

#### Metashape zip/xml 解析链

```text
read_project_zip()
  -> _get_xml_str_from_zip_file(project.zip, doc.xml)
  -> ElementTree.fromstring()
  -> 读取 chunks
```

位置：行 1393-1463。

```text
read_chunk_zip()
  -> _get_xml_str_from_zip_file(chunk.zip, doc.xml)
  -> ElementTree.fromstring()
  -> _decode_chunk_transform_tag()
  -> _sensorxml2object()
       -> _decode_sensor_tag()
            -> _decode_calibration_tag()
  -> _photoxml2object()
       -> _decode_camera_tag()
  -> 读取 frame.zip/doc.xml
       -> _decode_frame_xml()
       -> parse_relative_path()
  -> _decode_chunk_reference_tag()
```

位置：

- `read_chunk_zip()`：行 1466-1654。
- `_sensorxml2object()`：行 1657-1677。
- `_photoxml2object()`：行 1680-1753。
- `_decode_chunk_transform_tag()`：行 1825-1862。
- `_decode_chunk_reference_tag()`：行 1865-1953。
- `_decode_sensor_tag()`：行 1956-2044。
- `_decode_calibration_tag()`：行 2047-2131。
- `_decode_camera_tag()`：行 2134-2276。
- `_decode_frame_xml()`：行 2279-2357。

问题点：

- `read_chunk_zip()` 行 1603：`bool(xml_tree.attrib["enabled"])` 对字符串 `"false"` 仍为 `True`。应显式比较 `"true"`。
- XML 使用 `xml.etree.ElementTree` 直接解析，若面对不可信工程文件，建议 `defusedxml`。
- `_decode_chunk_reference_tag()` 行 1885 假设 reference 一定存在；本地或损坏项目会 `IndexError`。
- `_decode_chunk_reference_tag()` 行 1950 假设 `datum.to_json_dict()` 一定有 `"id"` 与 `"ellipsoid"`，某些 CRS 可能不满足。
- `_decode_sensor_tag(debug_meta={})` 使用可变默认参数，行 1956。
- `_photoxml2object()` 行 1724-1725：master photo 可能尚未解析或已 disabled，直接引用 `photos[camera.master_id].transform` 有顺序和完整性风险。

#### 坐标转换流程

```text
_local2world()
  -> apply_transform_matrix(points, chunk.transform.matrix)

_world2local()
  -> 如 matrix_inv 为 None，np.linalg.inv()
  -> apply_transform_matrix(points, matrix_inv)

_world2crs()
  -> world geocentric -> self.crs 或 _reference_crs

_crs2world()
  -> self.crs 或 _reference_crs -> world geocentric
```

位置：行 458-481。

#### Metashape 单张投影流程

```text
_back2raw_one2one(points_np, photo_id)
  -> 取 Photo 与 Sensor
  -> t = camera.transform[0:3, 3]
  -> r = camera.transform[0:3, 0:3]
  -> xyz = (points_np - t).dot(r)
  -> xh = x / z, yh = y / z
  -> distortion_correct=True:
       sensor.calibration.calibrate(xh, yh)
     else:
       用 f/cx/cy 计算无畸变像素
```

位置：行 483-552。

#### Metashape 旧 back2raw 流程

```text
back2raw_crs(points_xyz)
  -> CRS -> world -> local
  -> 遍历 photos
       -> _back2raw_one2one()
       -> in_img_boundary()
```

位置：行 554-659。

`back2raw_old()` 行 661-758 是多 ROI 旧实现，会临时修改 `self.crs` 再恢复。

#### Metashape 新 batch back2raw 流程

位置：行 891-1063。

调用链：

```text
back2raw(roi)
  -> 临时 self.crs = roi.crs
  -> Step 1: 去重所有 ROI 顶点
  -> Step 2: CRS -> local
  -> Step 3: _prepare_camera_transforms()
  -> Step 4: _batch_project_to_cameras()
  -> Step 5: 按 ROI 重构输出
  -> 恢复 self.crs
  -> 可选保存
```

辅助函数：

- `_prepare_camera_transforms()`，行 764-799。
- `_batch_project_to_cameras()`，行 801-889。

优点：

- 将 `ROI x photo x point` 中的大部分矩阵计算向量化。
- 多光谱按 sensor group 批量标定。

问题点：

- `ignore` / `log` 在 docstring 说明 “未实现”，但函数签名仍接受 `**kwargs` 并静默忽略，行 891-936。
- 与 `back2raw_crs()` 行 554-659 行为不完全一致：batch 仅支持全顶点都在边界内，不支持 `ignore="x"/"y"/"as_point"`。
- `all_points` / `split_indices` 在行 967-992 收集但后续未使用。
- 使用 `tuple(pt)` 做浮点去重，行 984-988；对微小数值误差不鲁棒。
- `try/finally` 缺失：如果中途异常，`self.crs` 与 progress bar 可能无法恢复/关闭。
- 一次性创建 `uv` 和 `valid`，空间复杂度 `O(C * U)`，大项目可能内存过高。

#### Metashape 相机位置

```text
get_photo_position()
  -> 临时切换 self.crs
  -> t_vecs = camera.transform[0:3, 3]
  -> local -> world -> crs
  -> 写 p.position
  -> 恢复 self.crs
  -> 缓存
```

位置：行 1065-1162。

问题点同 Pix4D：

- `_photo_position_cache` 不区分目标 CRS。
- 临时修改 `self.crs`，非线程安全。
- 异常时没有 `finally` 恢复。

---

## 2. 是否可抽象为 `ReconstructionProject` adapter 基类

结论：**可以，而且很有必要**。

当前 `Recons` 已有共同状态，但缺少明确接口，导致 Pix4D / Metashape 重复实现：

- `open_project()`
- `back2raw()`
- `back2raw_crs()`
- `get_photo_position()`
- `sort_img_by_distance()`
- `show_roi_on_img()`
- DOM / DSM / PCD 加载
- 坐标转换与相机投影语义

### 建议抽象结构

```python
class ReconstructionProject(ABC):
    software: str
    label: str
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
    def local_to_world(self, points: np.ndarray) -> np.ndarray:
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

    def project_crs_to_photo(...):
        # 通用 CRS -> world -> local -> photo

    def back2raw(...):
        # 通用遍历或 batch 策略

    def get_photo_position(...):
        # 通用缓存策略，由 adapter 提供 local/world/photo center
```

### Adapter 分层建议

```text
easyidp.reconstruction/
  base.py
    ReconstructionProject
    Sensor
    Photo
    Calibration
    CameraModel
    ProjectionResult
  adapters/
    pix4d.py
      Pix4DProjectAdapter
      Pix4DParamParser
    metashape.py
      MetashapeProjectAdapter
      MetashapeXmlParser
    odm.py
      ODMProjectAdapter
      OpenSfMParser
    dji_terra.py
      DJITerraProjectAdapter
      TerraParser
  projection/
    camera_models.py
    distortion.py
    transforms.py
    batch.py
  io/
    geotiff_products.py
    pointcloud_products.py
```

### 兼容 ODM / DJI Terra 的关键接口

#### ODM / OpenDroneMap

常见来源：

- OpenSfM `reconstruction.json`
- `camera_models.json`
- `shots`
- ODM outputs 的 `odm_georeferencing`, `odm_orthophoto`, `odm_dem`
- 可能的 `opensfm/undistorted/reconstruction.json`

Adapter 重点：

- 将 OpenSfM camera model 映射到 `Sensor` / `Calibration`。
- 将 shot pose 映射到 `Photo.transform`。
- 处理 coordinate reference：OpenSfM 内部 ENU / topocentric 与 ODM georeferencing。
- 支持 Brown / perspective / fisheye / spherical 模型。

#### DJI Terra

可能来源随版本差异较大：

- 工程 XML/JSON。
- Aerotriangulation result。
- `terra` 输出内外参文本。
- DOM/DSM/point cloud 标准 GeoTIFF/PLY/LAS。

Adapter 重点：

- 路径解析要插件化，不能硬编码目录结构。
- 标定模型需要兼容 DJI 相机内参/畸变字段。
- CRS 与 offset 要独立建模，避免 Pix4D 的 `p4d_offset` 特化泄漏到通用层。

---

## 3. 第三方库替代 XML / 文本 / 矩阵解析逻辑评估

### XML / zip 解析

当前：

- `zipfile`
- `xml.etree.ElementTree`
- `xml.dom.minidom`

建议：

- **安全性**：若读取用户提供项目文件，使用 `defusedxml.ElementTree` 替代 `xml.etree.ElementTree`。
- **XPath / 容错**：复杂 XML 可用 `lxml.etree`，但会增加依赖。
- **数据模型**：建议用 `pydantic` 或 `dataclasses` 建立中间 schema，而不是直接返回 dict。

### 文本解析

当前 Pix4D 文本解析大量手写。

可替代或增强：

- `numpy.loadtxt` / `genfromtxt`：适合规则矩阵，如 `pmatrix`。
- `pandas.read_csv(delim_whitespace=True)`：适合列式文件，但增加依赖和开销。
- `pyparsing` / `lark`：对 `.ssk` 这种 key-value 块结构更健壮，但可能过重。
- 简洁方案：写通用 `read_key_value_file()`、`read_matrix_blocks()`，封装容错与错误上下文。

### 矩阵 / 相机模型

可参考或替代：

- `scipy.spatial.transform.Rotation`：旋转矩阵、欧拉角、四元数互转。
- `opencv-python`：`cv2.projectPoints` 可处理常见 pinhole + distortion，但 Pix4D / Metashape 畸变公式和坐标中心差异仍需适配。
- `pycolmap`：对 COLMAP / OpenSfM 类相机模型有帮助，但作为 EasyIDP 核心依赖可能偏重。
- `opensfm`：ODM/OpenSfM 格式可直接参考，但稳定 API 与安装成本需评估。

结论：**不建议完全替换现有解析逻辑**，建议抽象为 parser adapter，并在内部局部使用 `defusedxml`、`scipy Rotation`、可选 OpenCV/pycolmap。

---

## 4. 文档与函数签名 / 行为一致性检查

### 主要不一致

| 位置                       | 问题                                                                                   | 影响                           |
| -------------------------- | -------------------------------------------------------------------------------------- | ------------------------------ |
| `reconstruct.py:192-287` | `in_img_boundary()` 支持 `"as_point"`，文档和错误信息未列                          | 用户不知道该模式，错误信息误导 |
| `pix4d.py:611-701`       | `save_folder` 文档默认 `""`，签名默认 `None`                                     | 小问题                         |
| `pix4d.py:522-532`       | 文档参数写 `distortion_correct`，签名是 `distort_correct`                          | 用户按文档传参会失败           |
| `pix4d.py:529-532`       | “If back to software corrected images ... set it to True” 语义疑似相反               | 投影结果可能错误               |
| `metashape.py:891-936`   | `ignore` / `log` 文档写未实现，但签名 `**kwargs` 静默吞掉                        | 用户以为生效，实际无效         |
| `metashape.py:933-936`   | See Also 中 `back2raw : Original implementation` 指向自己，旧实现是 `back2raw_old` | 文档误导                       |
| `reconstruct.py:762-847` | `save_folder` 文档写 str，代码接受 `Path`                                          | 小问题                         |
| `reconstruct.py:646-759` | `roi` 文档写 `idp.ROI`，实际只需 `.keys()` / `__getitem__` / `.crs`          | 可补充 Protocol                |
| `docs/python_api/*.rst`  | 多处 “preocessing / definately / reconstructin / coordiantes”等拼写错误              | 专业度下降                     |
| `autodoc`                | 仅列 autosummary，不解释新 batch 行为差异                                              | 用户难以理解性能与限制         |

---

## 5. 九维评分

满分 5 分。

| 维度          | 分数 | 理由                                                                                                                              |
| ------------- | ---: | --------------------------------------------------------------------------------------------------------------------------------- |
| 功能正确性    |  3.5 | Pix4D/Metashape 主流程有测试和实测期望值；但存在 chunk enabled 解析 bug、缓存 CRS 不区分、batch back2raw 与旧语义不一致等风险。   |
| 可读性        |  3.0 | 注释丰富，但函数过长、命名混杂、dict 结构隐式、Pix4D/Metashape 重复逻辑多。                                                       |
| 健壮性        |  2.5 | 文件格式强假设较多，XML/文本缺容错；异常时临时状态不一定恢复；部分路径和单行输入边界未处理。                                      |
| 性能          |  3.5 | Metashape 新 batch 显著优化；Pix4D 仍是 ROI x photo 循环；缓存策略粗糙；大规模内存风险未控制。                                    |
| 测试          |  3.8 | 覆盖项目解析、投影数值、多光谱、错误路径；但缺少 cache CRS、ignore batch、单行 pmat、false chunk enabled、线程/并发、副作用测试。 |
| 风格/规范     |  2.8 | PEP8 基本可读，但函数长度、可变默认参数、f-string 漏用、拼写、职责混合较多。                                                      |
| 文档/注释     |  3.2 | 文档体量足，示例多；但多处签名/行为不一致，旧行为与新 batch 行为差异说明不足。                                                    |
| 可维护性      |  2.8 | 支持 ODM/DJI Terra 前需重构 adapter/parser/projection 分层；当前新增软件会复制大量模式。                                          |
| 并发/线程安全 |  2.0 | 无并发设计；`self.crs` 临时修改、`_photo_position_cache`、`matrix_inv` 懒缓存、`Photo.position` 写入均非线程安全。        |

---

## 6. 命名与可读性专项

### 6.1 不良命名与建议

| 当前名称                             | 位置                     | 建议名称                                                                             | 理由                                                    |
| ------------------------------------ | ------------------------ | ------------------------------------------------------------------------------------ | ------------------------------------------------------- |
| `Recons`                           | `reconstruct.py:25`    | `ReconstructionProject`                                                            | 缩写不直观，作为基类应语义完整。                        |
| `ProjectPool`                      | `reconstruct.py:12`    | `ReconstructionProjectCollection`                                                  | 当前未实现，Pool 含义不清。                             |
| `pcd`                              | 多处                     | `point_cloud` 或保留属性但内部用 `_point_cloud`                                  | 对新用户不如 point_cloud 清晰。                         |
| `dom` / `dsm`                    | 多处                     | 可保留，但文档首次明确 Orthomosaic / Surface Model                                   | 领域缩写需统一说明。                                    |
| `ccp`, `cicp`, `ssk`, `pmat` | `pix4d.py`             | `camera_params`, `internal_params`, `sensor_settings`, `projection_matrices` | Pix4D 文件缩写对维护者不友好。                          |
| `read_ccp()`                       | `pix4d.py:1613`        | `read_calibrated_camera_parameters()`                                              | 明确文件含义。                                          |
| `read_cicp()`                      | `pix4d.py:1523`        | `read_internal_camera_parameters()`                                                | 明确文件含义。                                          |
| `read_cam_ssk()`                   | `pix4d.py:1856`        | `read_sensor_settings()`                                                           | 更通用。                                                |
| `points_hv` 文档                   | 多处                     | `points_xyz` / `polygon_xyz`                                                     | 文档写 hv，但实际 shape 是 nx3 xyz。                    |
| `distort_correct`                  | Pix4D                    | `distortion_correct`                                                               | 与文档和通用英语一致。                                  |
| `img_dict_all`                     | `sort_img_by_distance` | `back2raw_results`                                                                 | 明确数据来源。                                          |
| `roi`                              | 多处                     | `rois` 用于多 ROI，`roi_points` 用于单 ROI                                       | 避免对象/单 polygon 混淆。                              |
| `out_dict`                         | 多处                     | `projection_results`                                                               | 更能说明结构。                                          |
| `photo_name` vs `photo.label`    | 多处                     | 统一 `photo_key` / `photo_label`                                                 | Container key 与 label 有时是否含后缀不一致。           |
| `transform` 在 `Photo`           | `reconstruct.py:340`   | `camera_to_world` / `projection_matrix` 分类型字段                               | Pix4D 是 3x4 pmatrix，Metashape 是 4x4 pose，语义冲突。 |
| `software` 字符串分发              | `Calibration`          | `CameraModel` 子类                                                                 | 避免字符串条件扩散。                                    |

---

### 6.2 超过 50 行或职责不单一函数

以下按代码实际职责评估，不含 docstring 也大多偏长或职责复合。

#### `/src/easyidp/reconstruct.py`

| 函数                             |    行号 | 问题                                              | 切分建议                                                                                            |
| -------------------------------- | ------: | ------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `Sensor.in_img_boundary()`     | 192-287 | 判断、裁剪、日志、弃点模式混在一起；原地修改输入  | 拆成 `check_polygon_bounds()`, `clip_polygon_to_bounds()`, `filter_points_in_bounds()`        |
| `Calibration.calibrate()`      | 448-498 | 字符串分发；错误信息 bug                          | 用 `CameraModel` 多态替代                                                                         |
| `sort_img_by_distance()`       | 646-759 | 位置获取、ROI 中心、距离矩阵、过滤、保存混合      | 拆成 `compute_roi_centers()`, `compute_camera_distances()`, `filter_projection_by_distance()` |
| `save_back2raw_json_and_png()` | 762-847 | JSON 保存、数据转置、图像读取、裁剪、PNG 保存混合 | 拆成 `invert_back2raw_results()`, `save_back2raw_json()`, `crop_and_save_roi_images()`        |

接口草案：

```python
def compute_roi_centers(rois: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    ...

def filter_back2raw_by_distance(
    results: Mapping,
    roi_centers: Mapping[str, np.ndarray],
    camera_positions: Mapping[str, np.ndarray],
    *,
    max_distance: float | None = None,
    limit: int | None = None,
) -> dict:
    ...
```

#### `/src/easyidp/pix4d.py`

| 函数                           |      行号 | 问题                                                    | 切分建议                                                                              |
| ------------------------------ | --------: | ------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `Pix4D.open_project()`       |   127-327 | 打开项目、解析文件、构造 sensor/photo、加载产品全部混合 | `parse_project_files()`, `build_sensor()`, `build_photos()`, `load_outputs()` |
| `Pix4D.back2raw()`           |   611-701 | ROI 校验、投影、保存混合                                | 复用公共 base `back2raw()`                                                          |
| `Pix4D.get_photo_position()` |   703-783 | 缓存、转换、写 Photo 状态混合                           | 抽出 `compute_photo_positions()`，缓存由 base 管理                                  |
| `Pix4D.show_roi_on_img()`    |  932-1008 | 校验与可视化混合                                        | 公共 visualization helper                                                             |
| `parse_p4d_param_folder()`   | 1056-1202 | 多文件匹配重复代码                                      | 用 spec 表驱动                                                                        |
| `parse_p4d_project()`        | 1205-1379 | 目录结构、fallback、输出产品匹配混合                    | `find_param_folder()`, `find_point_cloud()`, `find_geotiff_outputs()`           |
| `read_ccp()`                 | 1613-1777 | 固定 block parser，可读性差                             | 写 `iter_camera_param_blocks()`                                                     |
| `read_cam_ssk()`             | 1856-1964 | key-value 解析硬编码                                    | 通用 `parse_ssk_key_values()`                                                       |

接口草案：

```python
@dataclass
class Pix4DProjectFiles:
    project_name: str
    offset_file: Path
    pmatrix_file: Path
    internal_camera_file: Path
    calibrated_camera_file: Path
    sensor_file: Path
    crs_file: Path
    point_cloud: Path | None = None
    dom: Path | None = None
    dsm: Path | None = None
```

#### `/src/easyidp/metashape.py`

| 函数                              |      行号 | 问题                                                       | 切分建议                                                                                      |
| --------------------------------- | --------: | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `_open_whole_project()`         |   291-360 | 解析 chunk、过滤、默认选择、状态写入混合                   | `load_project_chunks()`, `build_chunk_label_maps()`, `select_chunk()`                   |
| `_back2raw_one2one()`           |   483-552 | photo 解析、相机坐标、畸变/无畸变计算混合                  | `resolve_photo()`, `world_to_camera()`, `camera_to_pixel()`                             |
| `back2raw_crs()`                |   554-659 | CRS 转换、遍历、边界判断混合                               | base 复用                                                                                     |
| `back2raw_old()`                |   661-758 | 旧实现保留，临时状态修改                                   | 标记 deprecated，改用 context manager                                                         |
| `_batch_project_to_cameras()`   |   801-889 | 坐标投影、按 sensor 标定、边界判断混合                     | `batch_world_to_camera()`, `batch_apply_calibration()`, `batch_bounds_mask()`           |
| `back2raw()`                    |  891-1063 | 去重、CRS、batch、重组、保存、进度条混合                   | 拆成 `prepare_roi_points()`, `project_rois_batch()`, `reconstruct_projection_results()` |
| `get_photo_position()`          | 1065-1162 | 缓存、状态切换、计算、写 Photo 混合                        | base cache + pure compute                                                                     |
| `read_chunk_zip()`              | 1466-1654 | zip/xml、sensor/photo/frame/path/crs 全部混合              | `parse_chunk_doc()`, `parse_frame_docs()`, `resolve_photo_paths()`                      |
| `_photoxml2object()`            | 1680-1753 | 普通 camera、group、duplicate、multispectral 混合          | 拆 group parser 与 camera parser                                                              |
| `_decode_chunk_reference_tag()` | 1865-1953 | CRS 解析和 WGS84 修正混合                                  | `parse_metashape_crs()`, `normalize_wgs84_crs()`                                          |
| `_decode_sensor_tag()`          | 1956-2044 | 解析、选择 adjusted、错误格式化混合                        | `select_adjusted_calibration_tag()`                                                         |
| `_decode_camera_tag()`          | 2134-2276 | 基本字段、enabled、master、transform、rolling shutter 混合 | 多个小 parser                                                                                 |

---

## 7. 性能复杂度与优化建议

### 7.1 Pix4D `back2raw`

位置：

- `Pix4D.back2raw()`：`pix4d.py:611-701`
- `Pix4D.back2raw_crs()`：`pix4d.py:522-609`
- `_pmatrix_calc()`：`pix4d.py:489-520`

复杂度：

- 时间：`O(R * C * P)`
- 空间：单 ROI 单 photo 为 `O(P)`，总体输出取决于命中数量。
- `R` = ROI 数，`C` = photo 数，`P` = ROI 顶点数。

瓶颈：

1. Python 双层循环 `ROI x photo`。
2. 每次 `_pmatrix_calc()` 都构造齐次坐标。
3. `Sensor.in_img_boundary()` 对每张图做 min/max。
4. `back2raw()` 没有像 Metashape 一样对 ROI 顶点去重或批量投影。

优化策略：

1. **批量 Pix4D pmatrix 投影**将所有 photo 的 `3x4` pmatrix stack 为 `(C, 3, 4)`，所有点为 `(U, 4)`，用 `einsum` 得到 `(C, U, 3)`。
2. **ROI 点去重**复用 Metashape batch 里的 unified points，但建议用 `np.unique(axis=0)` 或带容差量化。
3. **预过滤候选照片**先用相机位置、视场粗 bbox、距离阈值过滤，再做精确投影。
4. **缓存 offset 后的 ROI 点**
   对同一 ROI 多次投影时避免重复 `points_xyz - offset`。

基准建议：

- 数据规模：
  - 小：`R=10, C=100, P=5`
  - 中：`R=1_000, C=300, P=5`
  - 大：`R=10_000, C=1_000, P=5`
- 指标：
  - 总耗时。
  - 每百万 `point-photo` 投影耗时。
  - 峰值内存。
  - 命中照片数量。
- 对比：
  - 当前循环版。
  - batch pmatrix 版。
  - batch + 候选照片预过滤版。

---

### 7.2 Metashape batch `back2raw`

位置：

- `Metashape.back2raw()`：`metashape.py:891-1063`
- `_batch_project_to_cameras()`：`metashape.py:801-889`

复杂度：

- 设唯一点数 `U`，照片数 `C`，sensor 数 `S`。
- 时间：`O(C * U)`。
- 空间：`uv` 为 `O(C * U * 2)`，`valid` 为 `O(C * U)`。
- 去重：当前 dict tuple 点，约 `O(R * P)`。

瓶颈：

1. `xyz_batch`、`uv`、`valid` 对大项目内存占用高。
2. `valid` 只支持严格全点在图内，不支持 ignore 裁剪。
3. 每个 sensor group 内仍有 Python loop 写回 `u_all/v_all` 和边界 mask，行 873-884。
4. CRS 转换一次性处理所有点，通常好，但超大 ROI 可能内存峰值高。

优化策略：

1. **分块 batch**按照片块或点块处理，例如 `camera_chunk_size=128`，降低峰值内存。
2. **候选照片预筛选**用 ROI bbox/中心与相机位置做距离或视锥粗筛，减少 `C`。
3. **边界判断向量化到 sensor group**对 group 直接生成 `(group_size, U)` mask，避免内层逐 photo loop。
4. **可选 sparse 输出**
   不保存完整 `uv`，直接对每个 ROI 的 point indices 判断和提取，适合超大项目。

基准建议：

- 对比 `back2raw_old()` 与 batch `back2raw()`：
  - `R=100/1000/10000`
  - `C=150/1000/5000`
  - 多 sensor 项目：RGB 单 sensor、多光谱 4 sensor。
- 记录：
  - 去重耗时。
  - CRS 转换耗时。
  - batch 投影耗时。
  - 重构结果耗时。
  - 峰值内存。
- 工具：
  - `pytest-benchmark`
  - `memory_profiler` 或 `tracemalloc`
  - 固定随机 ROI seed。

---

### 7.3 路径解析

热点：

- Pix4D `open_project()` 行 265-290：`os.listdir` + `img_label in img_list`。
- Metashape `read_chunk_zip()` 行 1630-1651：逐 frame / camera 路径解析。
- Pix4D `_match_suffix()` 行 1016-1053：多次 `os.listdir()`。

复杂度：

- Pix4D raw image 匹配：当前 list membership 为 `O(C * N)`，`N` 为文件夹图片数。
- 应改为 set：`O(C + N)`。

优化：

1. `img_set = set(os.listdir(raw_img_folder))`。
2. `_match_suffix()` 一次扫描所有文件，不要 ext list 嵌套扫描。
3. 路径使用 `Path` 统一，减少 str/Path 混用。

---

### 7.4 重复坐标转换

热点：

- `sort_img_by_distance()` 每次调用 `get_photo_position(to_crs=roi.crs)`。
- `get_photo_position()` 缓存不区分 CRS。
- `back2raw_old()` / `get_photo_position()` 临时修改 `self.crs`。

优化：

1. 缓存 key 用 `(target_crs.srs 或 to_epsg, project_revision)`。
2. 纯函数式 CRS 转换，不修改 `self.crs`。
3. ROI 对象 CRS 转换结果可缓存到 ROI 层或 projection session 层。

---

## 8. 并发与线程安全分析

当前没有显式并发代码，但对象含共享可变状态。

### 风险点

| 状态                             | 位置                                                           | 风险                                                  |
| -------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------- |
| `self.crs`                     | `Recons.crs`, `Metashape.back2raw`, `get_photo_position` | 多线程同时调用会互相覆盖 CRS。                        |
| `_photo_position_cache`        | `Recons.__init__`, Pix4D/Metashape `get_photo_position`    | 不区分 CRS；并发读写可能返回错误 CRS 结果。           |
| `Photo.position`               | `get_photo_position`                                         | 查询函数有写副作用。                                  |
| `transform.matrix_inv`         | `Metashape._world2local`                                     | 懒计算无锁，多线程重复写入风险小但不纯。              |
| `Sensor.in_img_boundary()`     | `reconstruct.py:252-270`                                     | 对输入 polygon 原地裁剪，调用者复用数组会受污染。     |
| `Metashape.back2raw()`         | 行 947-1057                                                    | 临时修改 `self.crs`，异常时无 finally，线程不安全。 |
| `save_back2raw_json_and_png()` | 文件 IO                                                        | 并发写同一目录会冲突。                                |

### 测试思路

1. **CRS cache 测试**

   - 连续调用：
     - `get_photo_position(to_crs=EPSG:4326)`
     - `get_photo_position(to_crs=EPSG:32654)`
   - 不传 `refresh=True`，检查是否错误复用缓存。
2. **并发读测试**

   - 用 `ThreadPoolExecutor` 同时调用：
     - `ms.get_photo_position(EPSG:4326)`
     - `ms.get_photo_position(EPSG:32654)`
     - `ms.back2raw(roi_a)`
     - `ms.back2raw(roi_b)`
   - 验证 `ms.crs` 结束后未被污染。
3. **异常恢复测试**

   - 构造一个 ROI 中途 shape 错误。
   - 调用 `Metashape.back2raw()` 后检查 `self.crs` 是否恢复。
4. **输入数组副作用测试**

   - 给 `Sensor.in_img_boundary(polygon, ignore="x")`。
   - 检查原始 polygon 是否被修改。
   - 若希望无副作用，应测试不修改。

### 建议

- 用 context manager 管理临时 CRS：

```python
@contextmanager
def temporary_crs(project, crs):
    old_crs = project.crs
    project.crs = crs
    try:
        yield
    finally:
        project.crs = old_crs
```

- 更优：不要修改 `self.crs`，所有转换函数显式传 `source_crs` / `target_crs`。
- 缓存改为：

```python
self._photo_position_cache: dict[str, dict[str, np.ndarray]]
# key = target_crs.to_string()
```

---

## 9. 测试覆盖评估

### 已覆盖较好的部分

#### `tests/test_reconstruct.py`

- `Recons` DOM/DSM/PCD 属性。
- `Sensor.in_img_boundary()` 的 `None/x/y`。
- `Calibration.calibrate()` 错误分支。
- `sort_img_by_distance()` Pix4D / Metashape。
- `save_back2raw_json_and_png()` 基本输出。

#### `tests/test_pix4d.py`

- Pix4D 标准/非标准项目结构解析。
- 参数文件读取：`xyz`, `pmat`, `cicp`, `ccp`, `campos`, `ssk`。
- Pix4D 项目加载。
- Pix4D back2raw 数值检查。
- get_photo_position CRS 转换。
- 短文件名访问。

#### `tests/test_metashape.py`

- transform matrix。
- geocentric/geodetic 互转。
- Metashape 初始化、多 chunk、缺 chunk。
- 多目录、嵌套目录路径。
- 单点/多点投影数值。
- CRS 投影。
- 缺 calibration、多 calibration、重复 sensor 名。
- disordered image xml。
- 多光谱 backward projection。

### 缺口

- `Metashape.back2raw()` batch 对 `ignore` 的行为差异未测试。
- `read_chunk_zip()` 的 `enabled="false"` chunk 解析 bug 未直接测试。
- `get_photo_position()` 缓存按 CRS 错误复用未测试。
- Pix4D `read_pmat()` 单行文件未测试。
- Pix4D raw image folder 大量文件匹配性能未测试。
- `Sensor.in_img_boundary(ignore="as_point")` 未测试。
- XML 安全/损坏 XML/缺 reference 未测试。
- `save_back2raw_json_and_png()` tqdm desc bug未测试。
- 并发和异常恢复未测试。

---

## 10. 面向 ODM / DJI Terra 与 MCP/skills 支持的模块化重构建议

### 10.1 模块化重构路线

#### 第一阶段：稳定公共模型

引入：

```text
reconstruction/base.py
reconstruction/models.py
reconstruction/camera.py
reconstruction/transforms.py
```

核心 dataclass：

```python
@dataclass
class CameraPose:
    camera_to_world: np.ndarray | None
    world_to_camera: np.ndarray | None
    projection_matrix: np.ndarray | None

@dataclass
class Photo:
    id: int
    label: str
    path: Path | None
    sensor_id: int
    pose: CameraPose
    enabled: bool = True
```

重点：不要再让 `Photo.transform` 同时表示 Pix4D pmatrix 与 Metashape 4x4 transform。

#### 第二阶段：Parser 与 Project 分离

当前 `Pix4D.open_project()` / `Metashape.read_chunk_zip()` 同时解析与构造对象。建议改为：

```text
Pix4DParser.parse(project_path) -> ReconstructionBundle
MetashapeParser.parse(project_path, chunk_id) -> ReconstructionBundle
ODMParser.parse(project_path) -> ReconstructionBundle
DJITerraParser.parse(project_path) -> ReconstructionBundle
```

`ReconstructionBundle`：

```python
@dataclass
class ReconstructionBundle:
    label: str
    crs: pyproj.CRS | None
    reference_crs: pyproj.CRS | None
    sensors: list[Sensor]
    photos: list[Photo]
    products: ReconstructionProducts
    metadata: dict
```

#### 第三阶段：投影引擎独立

```text
projection/engine.py
  project_points_to_photo()
  project_rois_to_photos()
  project_rois_to_photos_batch()
```

Adapter 只负责提供：

- 坐标转换。
- 相机 pose。
- 相机模型。
- 产品路径。

#### 第四阶段：可插拔软件支持

注册表：

```python
PROJECT_ADAPTERS = {
    "pix4d": Pix4DAdapter,
    "metashape": MetashapeAdapter,
    "odm": ODMAdapter,
    "dji_terra": DJITerraAdapter,
}
```

自动识别：

```python
def open_reconstruction_project(path, software="auto", **kwargs):
    adapter_cls = detect_adapter(path) if software == "auto" else PROJECT_ADAPTERS[software]
    return adapter_cls.open(path, **kwargs)
```

### 10.2 MCP / skills 支持建议

目标：让外部工具或 agent 能稳定读取项目结构、调用投影、检查数据质量。

建议提供清晰的 “tool-friendly” API：

```python
def inspect_project(path: str) -> dict:
    """返回软件类型、chunk、照片数、sensor、CRS、输出产品路径。"""

def validate_project(project: ReconstructionProject) -> list[Diagnostic]:
    """检查缺图、缺 CRS、缺 transform、sensor 不一致等。"""

def project_roi_to_raw(
    project_path: str,
    roi_geojson: dict,
    *,
    software: str = "auto",
    chunk_id: str | int | None = None,
    max_images: int | None = None,
) -> dict:
    """MCP 友好的 JSON 输入输出，不暴露 ndarray。"""
```

输出应避免 numpy 不可序列化对象：

```json
{
  "roi_id": {
    "image_name": [[x, y], [x, y]]
  }
}
```

为 skills/agent 提供：

- 项目结构检查 skill。
- 投影结果诊断 skill。
- 文档一致性检查 skill。
- 性能基准 skill。
- 新 adapter scaffold skill。

---

## 11. 关键风险清单

优先级从高到低：

1. **Metashape chunk enabled 解析错误**`/src/easyidp/metashape.py:1603``bool("false") == True`，应显式解析字符串。
2. **相机位置缓存未按 CRS 区分**`/src/easyidp/pix4d.py:744-783``/src/easyidp/metashape.py:1126-1162`可能返回错误坐标系的位置。
3. **Metashape batch `back2raw()` 静默忽略 `ignore/log`**`/src/easyidp/metashape.py:891-936`与旧实现和文档预期不一致。
4. **`self.crs` 临时修改非异常安全、非线程安全**`/src/easyidp/metashape.py:947-1057``/src/easyidp/metashape.py:1130-1157`
5. **Pix4D / Metashape 投影矩阵语义混用在 `Photo.transform`**`/src/easyidp/reconstruct.py:335-340`新 adapter 扩展会越来越难。
6. **文本/XML 解析强依赖格式**Pix4D `read_ccp/read_cam_ssk/read_cicp` 与 Metashape XML parser 均需更强错误上下文。
7. **大项目 batch 内存风险**
   `/src/easyidp/metashape.py:1027-1030`
   `uv` 与 `valid` 对 `C * U` 线性增长。

---

## 12. 总体结论

EasyIDP 当前 Pix4D 与 Metashape 支持已经具备完整主流程：项目解析、相机/传感器建模、CRS 转换、back2raw 投影、距离筛图和结果保存。测试覆盖不少真实场景，尤其 Metashape 多光谱和多 chunk 支持较有价值。

但若目标是兼容 OpenDroneMap、DJI Terra，并支撑 MCP/skills 自动化能力，当前结构需要从“软件类里混合解析+投影+IO”重构为：

```text
Parser Adapter -> 中间 ReconstructionBundle -> 通用 Projection Engine -> 产品/可视化/导出层
```

最优先应修复：

1. Metashape `enabled` 字符串解析。
2. `get_photo_position()` CRS-aware cache。
3. Metashape batch `back2raw()` 与 `ignore` 行为一致性。
4. `Photo.transform` 语义拆分。
5. `self.crs` 临时修改改为纯函数参数或异常安全 context manager。

Nya~♡
