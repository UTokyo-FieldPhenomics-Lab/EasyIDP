# User Prompt

只读背景调查，不要修改任何文件。工作目录 /home/crest/Documents/Github/EasyIDP。请重点分析 EasyIDP 当前 ROI/Container/矢量 IO 与未来基于 GeoPandas/Shapely 的架构重设计。

上下文：用户希望 EasyIDP 的 ROI 支持预处理/生成，根据 boundary polygon(s) 和规则生成 subplots/subgrids，并带预览功能。未来希望作为 /home/crest/Documents/Github/EasyPlantFieldID/src/utils/subplot_generate 的底座。用户怀疑 ROI/Container 本质更适合基于 GeoPandas GeoDataFrame，因为可记录 attribute、geometry、编辑 API、保存功能。

请做：

1. 阅读 src/easyidp/structures/container.py、src/easyidp/roi.py、src/easyidp/shp.py、src/easyidp/jsonfile.py、src/easyidp/geotools.py。
2. 阅读 /home/crest/Documents/Github/EasyPlantFieldID/src/utils/subplot_generate/io.py 及同目录相关文件（只读），理解 subplot_generate 现有输入/输出接口。
3. 总结当前 ROI 主要职责、调用链、数据模型。
4. 评估是否应继承 GeoDataFrame、组合 GeoDataFrame、还是保留自定义类但内部持有 GeoDataFrame。说明利弊。
5. 针对 str/int 双索引、attrs、geometry、多 CRS、保存 shp/geojson、生成 subplots 和预览，提出推荐架构和最小迁移路径。
6. 明确哪些 EasyIDP 旧 API 应兼容，哪些可 deprecated。

返回中文结构化 Markdown，只做分析，不写文件。

---

# EasyIDP ROI/Container 架构分析与 GeoPandas/Shapely 重设计评估

## 1. 当前架构速览

### 1.1 核心数据模型 (Container + ROI)

```
Container(dict)
  ├── id_item: Dict[int, ndarray]        # 整型索引 -> polygon ndarray
  ├── item_label: Dict[str, int]          # 字符串标签 -> id_item 的 int key
  └── 支持 int/str/slice 三种 __getitem__/__setitem__/__delitem__

ROI(Container)
  ├── crs: pyproj.CRS | None             # 统一的 CRS（所有 polygon 共享）
  ├── source: Path | None                # 来源文件路径
  ├── _attrs: List[dict]                 # 与 polygon 并行的属性行
  ├── _field_schema: dict                # DBF 字段模式 {name: (type,size,decimal)}
  ├── _subplot_meta: dict | None         # 动态挂载的 subplot 元数据（row, col, status）
  └── 方法: read_shp, read_labelme_json, read_geojson(BUG: line 432 有 pass),
            change_crs, save/save_shp, get_z_from_dsm, get_z_from_pcd,
            crop, back2raw, rename_by_fields
```

关键观察：

- **几何数据存储为裸 `numpy.ndarray`** (shape Nx2 或 Nx3)，不是 Shapely geometry 对象。
- `_attrs` 与 geometry 是**平行数组**关系，没有列式查询能力。
- `_subplot_meta` 是一个**临时 hack**：用 `setattr` 动态挂在 ROI 实例上，而不是正式的数据列。
- `read_geojson` 方法第 432 行有一个 **`pass` 语句导致后续代码永远不会执行**（Bug）。

### 1.2 矢量 IO 模块

| 模块            | 依赖库            | 输入格式                | 输出格式                        |
| --------------- | ----------------- | ----------------------- | ------------------------------- |
| `shp.py`      | pyshp (shapefile) | .shp                    | `{str: ndarray}` + attributes |
| `jsonfile.py` | geojson, json     | .geojson, labelme .json | `{str: ndarray}`              |
| `geotools.py` | pyproj, shapely   | dict 或 idp.ROI         | 坐标转换 / subplot ROI          |

**关键问题**：EasyIDP 同时依赖 pyshp + geojson 两个第三方库做 IO，而 GeoPandas 已经用 `fiona`/`pyogrio` 统一了所有矢量格式读写（shp, geojson, gpkg, kml 等 50+ 格式）。

### 1.3 subplot 生成调用链

```
EasyIDP:
  geotools.generate_subplots(boundary: idp.ROI, ...)
    -> _validate_boundary()            # 要求 idp.ROI, 恰好 1 个 polygon
    -> Polygon(coords[:, :2])          # numpy -> Shapely（临时转换）
    -> _compute_mar_info()             # MAR 方向向量
    -> _generate_subplot_grid()        # 矩形网格 -> numpy
    -> _classify_subplots()            # inside/touch/outside
    -> _create_roi_from_subplots()     # 封装回 idp.ROI，挂 _subplot_meta

EasyPlantFieldID:
  io.generate_subplots_gdf(boundary_gdf: gpd.GeoDataFrame, ...)
    -> gpd.GeoDataFrame direct
    -> Shapely Polygon 全生命周期
    -> 返回 gpd.GeoDataFrame({"id": ints}, geometry=polygons, crs=crs)
    -> 通过 .to_file() 保存
```

EasyPlantFieldID 版本**已经是纯 GeoPandas 工作流**，没有中间 numpy 转换，CRS 随 GeoDataFrame 流转。

---

## 2. 当前 ROI 主要职责与调用链

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  shp.py      │    │  jsonfile.py │    │  labelme     │
│  pyshp IO    │    │  geojson IO  │    │  json parse  │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ▼
                   ┌──────────────┐
                   │   ROI 对象    │
                   │  (Container)  │
                   └──────┬───────┘
                          │
          ┌───────────────┼───────────────────┐
          ▼               ▼                    ▼
   ┌─────────────┐ ┌─────────────┐    ┌──────────────┐
   │ GeoTiff     │ │ PointCloud  │    │ reconstruct  │
   │ .crop_rois()│ │ .crop_rois()│    │ .back2raw()  │
   │ polygon_math│ │ crop_polygon│    │              │
   │ point_query │ │             │    │              │
   └─────────────┘ └─────────────┘    └──────────────┘
          │               │                    │
          ▼               ▼                    ▼
       dict[str,       dict[str,         {roi_id: {img_id:
       ndarray]        ndarray]           pixel_coords}}
```

ROI 的核心职责：

1. **多源加载**：shp / geojson / labelme json
2. **属性管理**：字段名、记录、schema 的读写
3. **CRS 管理**：统一坐标系，按需转换
4. **几何运算转发**：作为 dict 容器传给 GeoTiff/PointCloud/reconstruct
5. **序列化**：保存为 shp（仅支持 shp）
6. **subplot 生成**：boundary → subplots

---

## 3. 三种架构方案利弊分析

### 方案 A：继承 GeoDataFrame (`class ROI(gpd.GeoDataFrame)`)

```python
class ROI(gpd.GeoDataFrame):
    # 关键列：geometry (Shapely), row, col, status, ...
    _metadata = ['_source', ...]  # 需处理 GeoDataFrame 元数据传递
```

| 利                                                                        | 弊                                                                           |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 免费获得 `.to_file()`, `.plot()`, `.explore()` 等全部 GeoPandas API | GeoDataFrame 内部极复杂（多重继承链）                                        |
| 天然支持 50+ 矢量格式                                                     | `__getitem__` 语义冲突：`roi[0]` 在 DataFrame 中是取列（column），不是行 |
| CRS 嵌入在对象内部，随切片自动传递                                        | 自定义 `__init__` 极易踩坑（需调用 `_constructor` 等）                   |
| `.sjoin()`, `.overlay()`, `.dissolve()` 等空间操作即开即用          | 老旧代码 `roi[str_key]` 需改为 `.loc[]`                                  |
| `_attrs` 变成普通列，可查询过滤                                         | 继承 DataFrame 的子类化历史悠久、bug 多                                      |

### 方案 B：组合 GeoDataFrame (`ROI._gdf`)

```python
class ROI:
    def __init__(self, ...):
        self._gdf = gpd.GeoDataFrame(columns=['id', 'geometry', ...])
  
    def __getitem__(self, key):  # 保持旧 API
        return self._gdf.loc[key, 'geometry'].coords[...]  # 转回 ndarray
  
    @property
    def gdf(self):  # 新 API
        return self._gdf
```

| 利                                            | 弊                                 |
| --------------------------------------------- | ---------------------------------- |
| 完全向后兼容现有 `roi["N1W1"]` 返回 ndarray | 内外两套接口，增加维护成本         |
| 可逐步暴露 GeoDataFrame 功能                  | 需要双写同步逻辑                   |
| 避免继承 GeoDataFrame 的复杂性                | 迁移期间对象有两份状态             |
| 旧代码零修改即可运行                          | `roi[0]` 的 int 索引语义需要适配 |

### 方案 C：保留自定义类，内部用 dict

| 利           | 弊                           |
| ------------ | ---------------------------- |
| 完全不改代码 | 无法获得 GeoPandas 任何优势  |
| 无迁移成本   | 维护 pyshp + geojson 两套 IO |
|              | _subplot_meta hack 持续      |
|              | _attrs 平行数组容易错位      |

### 推荐：**方案 B（组合）作为中间态，最终目标方案 A（继承）**

理由：

1. 方案 B 可以**渐进式迁移**：先增加 `.gdf` 属性，旧 API 加 `DeprecationWarning`
2. 方案 B 避免一次性破坏所有调用方（GeoTiff.crop_rois, PointCloud.crop_rois, reconstruct.back2raw 都依赖 `roi.items()` 返回 `(str, ndarray)`）
3. 最终可演化到方案 A，届时旧 dict API 全部废弃

---

## 4. 关键技术点分析

### 4.1 str/int 双索引

当前 Container 实现了 `__getitem__` 同时接受 int（按插入顺序）和 str（按标签）。若迁移到 GeoDataFrame：

- `gdf.iloc[0]` 替代 int 索引
- `gdf.loc["N1W1"]` 替代 str 索引
- `gdf.iloc[0:3]` 替代 slice 索引

建议保留 `ROI.__getitem__` 作为兼容层，内部转发到 `_gdf.loc[]`。

### 4.2 attrs（属性表）

当前 `_attrs: List[dict]` 是平行数组。GeoDataFrame 直接以列存储：

```python
# 旧:  roi._attrs[5]["CROPTYPE"]
# 新:  roi._gdf.at[5, "CROPTYPE"]
# 或:  roi._gdf["CROPTYPE"].iloc[5]
```

优势：可进行列式操作 `roi._gdf[roi._gdf["CROPTYPE"] == "小麦"]`。

### 4.3 geometry（几何类型）

当前 `ndarray` -> Shapely 转换散落在 5+ 处（geotools.py line 434, 1043, rois.py 等）。应统一在 IO 层转换为 Shapely Polygon，内部全生命周期使用 Shapely。

### 4.4 多 CRS

当前 `ROI.crs` 是一个共享属性，隐含"所有 polygon 同一 CRS"。这正好匹配 GeoDataFrame 的 `.crs`。若未来需要每个 polygon 不同 CRS，GeoDataFrame 支持 `gdf.estimate_utm_crs()` 或手动 `.to_crs()`。

### 4.5 保存 shp/geojson

当前仅支持 shp（`save_shp`），且绕过 GeoPandas。迁移后直接 `_gdf.to_file(path)` 即可支持 driver 自动探测（shp/geojson/gpkg 等）。

### 4.6 subplot 生成与预览

当前 `geotools.generate_subplots()` → `idp.ROI` with `_subplot_meta` hack。
`visualize.show_subplots()` 手动读取 `_subplot_meta["status"]` 来决定颜色。

改进后：

```python
# 返回 GeoDataFrame，status/row/col 是普通列
subplots_gdf = generate_subplots_gdf(boundary_gdf, ...)
# 列: id, geometry, row, col, status
# 预览: subplots_gdf.plot(column="status", cmap={...}, ax=ax)
# 保存: subplots_gdf.to_file("output.gpkg")
```

EasyPlantFieldID 的 `io.py` 已经是这个模式，可以直接作为底座。

---

## 5. 推荐架构与最小迁移路径

### 5.1 目标架构

```
                    ┌──────────────────────┐
                    │   GeoPandas / Fiona   │
                    │  (shp, geojson, gpkg, │
                    │   parquet, ...)       │
                    └──────────┬───────────┘
                               │ gpd.read_file()
                               ▼
                    ┌──────────────────────┐
                    │  GeoDataFrame (内部)  │
                    │  columns:             │
                    │   - id (str)          │
                    │   - geometry (Polygon)│
                    │   - ...attrs...       │
                    │   - row, col, status  │
                    │  crs: pyproj.CRS      │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼─────────────────┐
              ▼                ▼                  ▼
        ROI.__getitem__   ROI.gdf           ROI.to_file()
        (兼容层)           (GeoPandas API)   (导出)
```

### 5.2 最小迁移路径（3 阶段）

**Phase 1：引入 GeoDataFrame 内部存储（非破坏性）**

- `ROI.__init__` 增加 `self._gdf = gpd.GeoDataFrame()`
- `read_shp` 改用 `gpd.read_file()` 替代 `shapefile.Reader`
- `read_geojson` 改用 `gpd.read_file()` 替代 `geojson.load()`
- 旧的 `id_item`/`item_label` 从 `_gdf` 同步生成（向后兼容）
- `_attrs` 映射到 `_gdf` 的列
- 公开 `ROI.gdf` 属性供新代码使用

**Phase 2：subplot 生成重构**

- `geotools.generate_subplots()` 重构为返回 `gpd.GeoDataFrame`（参考 EasyPlantFieldID `io.py`）
- 新函数 `generate_subplots_gdf()` 作为主入口
- 旧函数保留但加 `DeprecationWarning`，内部调用新函数后转 ROI
- `visualize.show_subplots()` 改为接受 GeoDataFrame，读取列而非 `_subplot_meta`
- 增加 `ROI.plot_subplots()` 快捷方法调用 `.plot()`

**Phase 3：废弃旧接口**

- `read_labelme_json` 标记 deprecated（labelme 非标准格式，建议外部预处理）
- `_subplot_meta` 动态挂载移除
- `Container` 基类标记 deprecated
- `save_shp` → `save(to_file 兼容)`
- pyshp 依赖标记为可选

### 5.3 需兼容的旧 API

| API                                   | 兼容策略                                      |
| ------------------------------------- | --------------------------------------------- |
| `roi["N1W1"]` 返回 ndarray          | Phase 1-2 保留，内部从 `_gdf` 提取          |
| `roi[0]` int 索引                   | 保留，转发 `_gdf.iloc[0]`                   |
| `roi[0:3]` slice                    | 保留，返回新 ROI(shallow copy)                |
| `roi.items()` → `(str, ndarray)` | 保留，Phase 3 废弃                            |
| `roi.crs`                           | 保留，代理到 `_gdf.crs`                     |
| `roi.source`                        | 保留                                          |
| `roi.read_shp(...)`                 | 保留，内部改用 `gpd.read_file()`            |
| `roi.change_crs(crs)`               | 保留，转发 `_gdf.to_crs(crs, inplace=True)` |
| `roi.save_shp(...)`                 | Phase 1-2 保留，转发 `_gdf.to_file()`       |
| `roi.get_z_from_dsm(...)`           | 保留，操作逻辑不变                            |
| `roi.crop(...)`                     | 保留，外部接口不变                            |

### 5.4 可废弃的 API

| API                                                                                           | 废弃原因                                                             |
| --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `Container` 基类                                                                            | str/int 双索引由 GeoDataFrame 的 `loc`/`iloc`/`set_index` 替代 |
| `roi.read_labelme_json()`                                                                   | 非标准格式，应由预处理脚本处理                                       |
| `_attrs` / `_field_schema` 直接访问                                                       | 改为 `_gdf.columns`                                                |
| `_subplot_meta`                                                                             | 改为 GeoDataFrame 列                                                 |
| `idp.shp.read_shp()` 返回裸 list/dict                                                       | 改为返回 GeoDataFrame                                                |
| `idp.shp.write_shp()`                                                                       | `gdf.to_file()` 替代                                               |
| `idp.jsonfile.read_geojson()` 返回 dict                                                     | 改为返回 GeoDataFrame                                                |
| `jsonfile.dict2json` / `write_json` / `save_json`                                       | 三个完全重复的函数，保留一个即可                                     |
| `shp.py` 中的 `_get_field_key`、`_get_field_schema`、`_infer_field_schema_from_attrs` | GeoPandas 自动处理 schema                                            |

### 5.5 EasyPlantFieldID 作为底座

EasyPlantFieldID 的 `io.py` 已经是理想的 subplot 生成参考实现：

- 输入：`gpd.GeoDataFrame` (1 row boundary)
- 输出：`gpd.GeoDataFrame` with `id`, `geometry`, `crs`
- 支持 grid mode 和 size mode、x/y spacing、keep mode
- 提供 `calculate_optimal_rotation()` 辅助函数

EasyIDP 可以直接 import 或 vendoring 这份逻辑，删除 `geotools.py` 中 ~100 行的 `generate_subplots` 及其 6 个私有辅助函数。

---

## 6. 总结

| 维度           | 现状                            | 推荐                                           |
| -------------- | ------------------------------- | ---------------------------------------------- |
| 几何存储       | numpy ndarray                   | Shapely Polygon（GeoDataFrame 列）             |
| IO 库          | pyshp + geojson (2个)           | GeoPandas/Fiona (50+ 格式)                     |
| 属性表         | `_attrs: List[dict]` 平行数组 | GeoDataFrame 列                                |
| CRS            | 独立属性 `crs`                | GeoDataFrame.crs                               |
| subplot meta   | `_subplot_meta` 动态挂载      | GeoDataFrame 列 (row, col, status)             |
| 预览           | matplotlib 手动绘制             | `gdf.plot()` / `.explore()`                |
| 保存           | 仅 .shp                         | `.to_file()` 自动探测格式                    |
| 空间操作       | 手动调用 shapely                | `.sjoin()`, `.overlay()`, `.dissolve()`  |
| str/int 双索引 | 自定义 Container                | GeoDataFrame `.loc`/`.iloc` + 兼容 wrapper |

**核心结论**：ROI 在当前形态下本质是一个"没有类型安全、没有空间索引、没有列式查询的 GeoDataFrame 等价物"。采用组合 GeoDataFrame 的方案 B 作为中间态，逐步暴露 GeoPandas API，最终可演化到轻量继承方案 A。该迁移可在不破坏外部接口的前提下分 3 个阶段完成，第一阶段（引入 `_gdf`）即可带来最大的功能增益。Nya~♡
