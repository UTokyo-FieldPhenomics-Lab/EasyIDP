# User Prompt

只读调查，不要修改任何文件。工作目录 /home/crest/Documents/Github/EasyIDP。

任务：调查 EasyIDP 的 structures/Container 是否需要纳入 v2.1 新架构重构。重点回答：Container 除 ROI 外是否还服务 Metashape Photos/Sensors？如果 Reconstruction 也要重构，Container 是否还有保留价值？

请阅读：

- src/easyidp/structures/container.py
- src/easyidp/structures/__init__.py
- src/easyidp/roi.py 中继承 Container 的用法
- src/easyidp/reconstruct.py 中 sensors/photos/ProjectPool 相关用法
- src/easyidp/metashape.py、src/easyidp/pix4d.py 中 photos/sensors Container 的访问方式
- tests/test_init_class_func.py、tests/test_reconstruct.py、tests/test_pix4d.py、tests/test_metashape.py 中 Container 相关行为
- docs/python_api/index.rst、reconstruct.rst、manualdoc/easyidp.roi.ROI.rst 中 Container/ROI 说明
- .agents/rules/architecture.md 与 .agents/references/v2.1refactor/reconstruction_adapters.md、roi_geopandas_architecture.md

请输出中文结构化 Markdown：

1. 当前 Container 的职责、行为、调用链和真实使用点。
2. Container 在 ROI、Recons.sensors、Recons.photos、ProjectPool 中分别承担什么职责。
3. 当前问题：继承 dict 是否合理、双索引语义、删除/切片/copy/pretty print、全局 numpy printoptions、类型安全、文档/API 风险。
4. 如果 ROI 改用 GeoDataFrame composition、Reconstruction 改用 Bundle/list/dict/dataclass，Container 是否还有存在必要。
5. 推荐 v2.1 策略：保留为 internal? 改为 LabelledCollection? 完全废弃？请给 2-3 个方案和推荐方案。
6. 哪些旧 API 应兼容或直接破坏。
7. 建议是否写入 .agents/references/v2.1refactor/container_collection_architecture.md，以及建议内容提纲。

返回结果即可，不写文件。

---

# Container 调查与 v2.1 策略报告

## 1. Container 的职责、行为、调用链与实现

### 1.1 核心实现 (`src/easyidp/structures/container.py`)

`Container(dict)` 是一个继承自 `dict` 的自定义集合类，通过两个平行字典实现 int-key 与 str-label 的双索引：

```python
class Container(dict):
    def __init__(self, suffix=""):
        self.id_item: dict[int, Any] = {}      # int → value
        self.item_label: dict[str, int] = {}    # str → int key
        self._suffix: str = str(suffix)         # for fuzzy match (e.g. ".JPG")
```

**核心行为表**：

| 操作                                | 行为                                                     | 复杂度         |
| ----------------------------------- | -------------------------------------------------------- | -------------- |
| `ctn[0]` (int)                    | `id_item[key]` 直接取值                                | O(1)           |
| `ctn["label"]` (str)              | `id_item[item_label[key]]` 两级查找                    | O(1)           |
| `ctn["name.JPG"]` (str + suffix)  | 若精确匹配失败，strip `_suffix` 后再匹配               | O(1)           |
| `ctn[0:3]` (slice)                | 取 id_item 子集，创建新 Container（含 deepcopy）         | O(n)           |
| `ctn[0] = obj` (int set)          | 索引处已有则覆盖、等于 `len()` 则追加，否则 IndexError | O(1)           |
| `ctn["lbl"] = obj` (str set)      | 若 label 已存在追加，否则创建新条目                      | O(1)           |
| `del ctn[0]` / `del ctn["lbl"]` | 删除后全量重编号（`idx-1` 移位）                       | **O(n)** |
| `for x in ctn`                    | 迭代 `id_item.values()`（int 序）                      | O(n)           |
| `ctn.keys()`                      | 返回 `item_label.keys()`（str 序）                     | O(n)           |
| `ctn.copy()`                      | `deepcopy(self)` ── 代价高昂                         | O(n)           |
| `repr/str`                        | 调用 `_btf_print()`                                    | O(n)           |

**`_btf_print()` 的严重副作用**：打印时**临时修改全局 numpy printoptions**（`np.set_printoptions(threshold=4, suppress=True)`），打印后恢复。这对多线程/异步环境有隐患。

### 1.2 所有构造点

```
src/easyidp/structures/__init__.py:3    → from .container import Container
src/easyidp/__init__.py:17              → from .structures import Container
src/easyidp/reconstruct.py:12           → class ProjectPool(idp.Container)
src/easyidp/reconstruct.py:59           → self.sensors = idp.Container()
src/easyidp/reconstruct.py:61           → self.photos = idp.Container()
src/easyidp/roi.py:13                   → class ROI(idp.Container)
src/easyidp/metashape.py:1658           → sensors = idp.Container()  (_sensorxml2object)
src/easyidp/metashape.py:1681           → photos = idp.Container()   (_photoxml2object)
```

### 1.3 内部状态直接访问点（`id_item` / `item_label` 被外部直接读写）

`roi.py` 中大量出现（共 20+ 处）：

```python
# roi.py:235-236 — read_shp() 里直接重置
self.id_item = {}
self.item_label = {}

# roi.py:310 — rename_by_fields() 直接替换 item_label
self.item_label = {key: idx for idx, key in enumerate(generated_keys)}

# roi.py:584 — change_crs() 直接替换 id_item
self.id_item = idp.geotools.convert_proj(self.id_item, self.crs, target_crs)

# roi.py:1000/1013/1153 — 多处 .copy() 然后直接用
poly_dict = self.id_item.copy()

# roi.py:1032/1189 — item_label 用于内部遍历
poly = poly_dict[self.item_label[roi_name]]
```

## 2. 在各上下文中的职责

### 2.1 ROI (`roi.py`)

- **模式**：`ROI(Container)` ── ROI **继承** Container。
- **存储内容**：值 = `np.ndarray` 坐标数组 (nx2 或 nx3)；key = str label（如 "N1W1"），额外通过 `_attrs` 存属性表。
- **使用的 Container 特性**：`__getitem__` (int/str/slice)、`__setitem__` (str)、`items()`、`keys()`、`copy()`、`__len__`。
- **对外暴露的内部状态风险**：用户可以直接 `roi.id_item` / `roi.item_label` 读取/破坏内部一致性。

### 2.2 Recons.sensors (`reconstruct.py:59`)

- **模式**：`self.sensors = idp.Container()` ── **组合** (非继承)。
- **存储内容**：key = `sensor.id` (int)；值 = `Sensor` 对象；label = `sensor.label`。
- **使用方式**：`self.sensors[camera_i.sensor_id]` (int lookup)、`sensors_dict[sid]`（遍历）。
- **不需要的特性**：slice、copy、suffix matching。注意：Recons 基类自己没有一个迭代传感器的公共 API，只有子类在内部使用。

### 2.3 Recons.photos (`reconstruct.py:61`)

- **模式**：`self.photos = idp.Container()` ── **组合**。
- **存储内容**：key = `photo.id` (int)；值 = `Photo` 对象；label = `photo.label`。
- **使用方式**：
  - `self.photos[photo_id]` (int lookup) — 有 `_back2raw_one2one`、`Metashape.back2raw`、`Pix4D.back2raw_crs`
  - `self.photos[img_name]` (str lookup) — 有 `Pix4D._check_photo_type`、`show_roi_on_img`
  - `self.photos["DJI_0174"]` 匹配 `self.photos["DJI_0174.JPG"]`（通过 Pix4D 的 `_suffix` hack）
  - `for photo in self.photos` 迭代 — 多处使用
  - `for photo_name, photo in self.photos.items()` — `Pix4D.back2raw_crs`、`Metashape.back2raw_crs`、`Metashape.back2raw`
  - `self.photos.keys()` — `Pix4D._check_photo_type` 的 str→Photo 转换
  - `len(self.photos)` — 测试断言

### 2.4 ProjectPool (`reconstruct.py:12`)

- **模式**：`ProjectPool(Container)` ── **继承**。
- **状态**：未实现。`add_pix4d()` 与 `add_metashape()` 都只有 `pass`。
- **roi.py `back2raw()` 中对 `idp.ProjectPool` 的 `isinstance` 检查**：触发 `NotImplementedError`。
- **结论**：从未实现的死代码。

## 3. 问题分析

### 3.1 继承 dict 不合理

- 文档明确标注 "This object can not be saved by pickle"。
- dict 继承带来大量意外行为：`dict.get()`、`dict.pop()`、`dict.update()`、`dict.setdefault()`、`dict.__contains__`、`dict.__eq__` 都不会正常工作（因为父类 dict 存储为空，实际数据在 `id_item`/`item_label` 中）。
- `copy()` 调用 `deepcopy` —— 对大量照片极其昂贵。

### 3.2 双索引语义不一致

- `id_item` 和 `item_label` 由两个独立 dict + 增/删时的手动同步来维持一致性。
- 如果用户直接赋值 `ctn.item_label = {...}`（ROI 内已这样做），两个索引会漂移。
- `__delitem__` 中的重编号是 O(n) 的，当项目包含 936 张照片（Metashape 测试中）时成本陡增。

### 3.3 删除/切片/copy/pretty print 的实现缺陷

| 操作             | 问题                                                                                            |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| `del ctn[i]`   | O(n) 重编号所有后续键——对包含 150-1000 张照片的 Container 是反模式。                          |
| `ctn[0:3]`     | 返回一个 `deepcopy` 的新 Container。内存与时间消耗都很大，而且由于是 copy，无法用于局部修改。 |
| `ctn.copy()`   | `deepcopy(self)` ── 在内部循环中大量使用（ROI 中约 10 次 `.copy()` 调用）。               |
| `_btf_print()` | **修改全局 `np.set_printoptions`**。这在多线程/异步 Jupyter 场景中是不安全的。          |

### 3.4 Suffix matching 是个 hack

- Pix4D 将 `img_label_no_suffix` 为 Photo.label，但将 `img_suffix` 存入 `self.photos._suffix`，以便 `photos["DJI_0174.JPG"]` 能匹配 `photos["DJI_0174"]`。
- 这个魔术 `__getitem__` 绕过正规的 key 系统，并且仅在 Pix4D 路径上生效。

### 3.5 类型安全

- `Container` 存储任意类型。没有编译期或运行时保证其内容为 `Sensor` 或 `Photo`。
- "label" 属性检查在运行时通过 `if "label" in dir(item)` 而非显式 protocol/ABC/duck-typing。

### 3.6 文档/API 风险

- 公共 API 表面暴露了危险字段 `id_item`、`item_label`、`_suffix`。
- 文档说不能 pickle，但并未屏蔽相关路径。
- `ROI` 同时继承 Container 又暴露 `_attrs`、`_field_schema` 等平行结构，整个 API 显得凌乱。

## 4. 若 ROI 改用 GeoDataFrame、Reconstruction 改用 Bundle/list/dict/dataclass，Container 是否还有存在必要？

**答案：完全没有必要。**

### 4.1 ROI 路径

`roi_geopandas_architecture.md` 已明确规划：

- `ROICollection` 以私有 `_gdf: GeoDataFrame` 组合。
- 通过 `.gdf` 供高级用户与内部模块访问。
- `loc`/`iloc` 风格的访问取代 Container 的双索引。
- 属性与几何不再漂移（同一 gdf 列）。
- CRS 由 GeoDataFrame 原生携带。

**需求矩阵**：

| Container 特性              | GeoDataFrame 替代方案                                           |
| --------------------------- | --------------------------------------------------------------- |
| `roi["N1W1"]` 标签访问    | `_gdf.loc["N1W1"]`                                            |
| `roi[0]` 位置访问         | `_gdf.iloc[0]`                                                |
| `roi[0:3]` 切片           | `_gdf.iloc[0:3]`                                              |
| `for k, v in roi.items()` | `_gdf.iterrows()` 或 `_gdf["geometry", "label"].iterrows()` |
| `roi.keys()`              | `_gdf.index`                                                  |
| `del roi["N1W1"]`         | `_gdf.drop("N1W1")`                                           |
| CRS 转换                    | `_gdf.to_crs()`（原生）                                       |

### 4.2 Reconstruction 路径

`reconstruction_adapters.md` 已明确规划：

- `ReconstructionBundle` 使用 `list[Photo]` 和 `list[Sensor]`，而非 Container。
- `Photo` 和 `Sensor` 为 `@dataclass`，携带 `id: str`、`label: str` 等字段。

**需求矩阵**：

| Container 特性                       | 替代方案                                                                                                            |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `photos[0]` 按 id 访问             | `photos_by_id: dict[int, Photo]` 或 `id_photo_map`                                                              |
| `photos["DJI_0422"]` 按 label 访问 | `photos_by_label: dict[str, Photo]`                                                                               |
| `for p in photos` 迭代             | `for p in bundle.photos`                                                                                          |
| `len(photos)`                      | `len(bundle.photos)`                                                                                              |
| 两个索引都在                         | `ReconstructionBundle` 可在构建时生成两个辅助 dict：`_photos_by_id`、`_photos_by_label`，比双平行 dict 更简单 |

### 4.3 ProjectPool 路径

- 从未实现，存在仅是为了占位。
- `roi.back2raw()` 中的 `isinstance(recons, idp.ProjectPool)` 分支直接抛 `NotImplementedError`。
- 可以安全移除。

## 5. v2.1 策略建议

### 方案 A（推荐）：完全废弃 Container

| 维度           | 行动                                                                                                                                          |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| ROI            | 迁移到 `ROICollection._gdf: GeoDataFrame`。`id_item`/`item_label` 变为 `_gdf.index` + `.geometry` 列。                              |
| Reconstruction | 迁移到 `ReconstructionBundle` 中含 `photos: list[Photo]` + `_photos_by_label: dict[str, Photo]` + `_photos_by_id: dict[int, Photo]`。 |
| ProjectPool    | 删除。                                                                                                                                        |
| Container      | 在 `__init__.py` 中移除导入。可在 `structures/container.py` 文件头标注 `# @deprecated since v2.1` 并保留文件供旧代码参考。              |
| 测试           | `test_class_container*`、`test_class_container_photo` → 重写以测试新的 Collection/Bundle 接口。                                          |

**优点**：与 v2.1 架构方向 100% 一致；消除所有上述问题；类型安全提高；pickle 兼容。

**缺点**：破坏性变更；需要重写 ROI 和 Reconstruction 的所有容器访问。

### 方案 B（保守折中）：退化为内部实现细节

- 将 `Container` 保留但标记为 `_Container`，仅供 `ReconstructionBundle` 内部使用。
- ROI 仍然迁移到 GeoDataFrame。
- Reconstruction 使用 `Bundle`，但内部用 `_Container` 管理 sensors/photos 的双索引。

**优点**：重构 churn 较少。

**缺点**：RoI Container 问题不解决；`_btf_print` 全局副作用问题依然存在；`_suffix` hack 依然存在。

### 方案 C（激进）：重写为 `LabelledCollection[T]`

- 用 `list[int]` 索引 + `dict[str, int]` label 映射创建一个新的、**非 dict 继承**的类型化集合。
- 纯组合（不继承任何东西）。仅提供 `by_id(0)`、`by_label("name")`、`__iter__`、`__len__`。
- 不含 suffix matching、不含全局打印副作用、不含 O(n) 删除重编号。
- ROI 仍然迁移到 GeoDataFrame；Reconstruction 使用此类型。

**优点**：处于废弃与保留之间；解决了 dict 继承和副作用问题。

**缺点**：又是一个需要维护的自定义集合类；ROI 迁移使其不需要；类型参数化需要 `typing.Generic`，在没有运行时检查的情况下价值有限。

### 推荐：方案 A（完全废弃）

理由：

1. 架构规则明确要求 "Prefer explicit data objects over parallel dict/list state"——`id_item`/`item_label` 正是被反对的模式。
2. ROI 目标 (`roi_geopandas_architecture.md`) 和 Reconstruction 目标 (`reconstruction_adapters.md`) 都不含 Container。
3. Container 的每一个设计选择（dict 基类、全局 numpy 副作用、suffix hack、O(n) 删除）都是技术债务，不适合在新架构中重用。
4. 没有其他模块依赖 Container——它是一个仅有 4 个消费点的内部类型。

## 6. 向后兼容性建议

### 应破坏（breaks）

| 旧 API                                       | 新 API                                                                  |
| -------------------------------------------- | ----------------------------------------------------------------------- |
| `roi["N1W1"]` → ndarray                   | `roi.gdf.loc["N1W1"]` 或 `roi_collection["N1W1"]`（shim）           |
| `roi[0]` → ndarray                        | `roi.gdf.iloc[0]`                                                     |
| `roi[0:3]` → Container                    | `roi.gdf.iloc[0:3]` → GeoDataFrame                                   |
| `roi.id_item` / `roi.item_label`         | 废弃——使用 `.gdf`                                                   |
| `recons.sensors[0]` / `recons.photos[0]` | `bundle.sensors[0]` / `bundle.photos[0]` 或 `bundle.get_photo(0)` |
| `recons.photos["DJI_0422"]`                | `bundle.get_photo_by_label("DJI_0422")`                               |
| `for photo in recons.photos`               | `for photo in bundle.photos`                                          |
| `idp.Container()`                          | 删除                                                                    |
| `idp.ProjectPool()`                        | 删除                                                                    |

### 应保留兼容性 / Shim 层（临时）

- `roi["N1W1"]` 和 `roi[0]` 可保留为 `ROICollection.__getitem__` 的向后兼容 shim，但最终迁移到 `.loc`/`.iloc`。
- `recons.photos` 和 `recons.sensors` 可保留为兼容属性（返回轻量 dict wrapper），但应从 `ReconstructionBundle` 代理。

## 7. 架构文档建议

**是**，建议创建 `.agents/references/v2.1refactor/container_collection_architecture.md`。

### 推荐提纲

```markdown
# Container / Collection Architecture Notes

## Current Model (v2.0)
- Container(dict) + id_item/item_label 的双索引设计
- 消费方：ROI、Recons.sensors、Recons.photos、ProjectPool
- 已识别问题：dict 继承、全局 printoptions 副作用、_suffix hack、
  O(n) 删除、不可 pickle、并行状态漂移

## v2.1 Direction
- 废弃 Container；无新的通用集合基类

## ROI Replacement
- ROICollection 以 GeoDataFrame 为后端的组合模式
- .loc/.iloc 取代 Container 索引
- 属性存储在 gdf 列中

## Reconstruction Replacement
- ReconstructionBundle 含 list[Photo] + list[Sensor]
- 通过 dict 属性提供 label/id 查找
- Pix4D _suffix hack 替换为显式 label 规范化

## Migration
- Container 类加 @deprecated 并保留文件以供参考
- test_class_container* 重写以适配新接口
- 旧 Recons._show_chunk 等 display 方法也一并重构

## ProjectPool Removal
- 未实现，直接移除
- roi.back2raw() 中 isinstance(ProjectPool) 分支移除
```

---

*调查完毕。Container 在 v2.1 架构中完全没有保留价值——ROI 移向 GeoDataFrame，Reconstruction 移向 dataclass + list/dict，两条路径都不需要它。建议完全废弃。Nya~♡*
