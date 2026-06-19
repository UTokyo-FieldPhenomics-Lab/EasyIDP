# User Prompt

你在 /home/crest/Documents/Github/EasyIDP 中做只读代码审查，不要修改文件。项目索引在 repomix-output.xml，源码重点在 src/easyidp。请详细分析以下范围：src/easyidp/__init__.py、src/easyidp/data.py、src/easyidp/logger.py、pyproject.toml、README.md、docs/python_api/index.rst、docs/python_api/**/*.rst、tests/test_init_class_func.py、tests/test_data.py，以及整体 tests 组织。

目标：

1. 梳理包级 API 暴露、数据集下载、日志/告警机制、依赖与测试组织。
2. 检查文档和目前函数/类的一致性，列出 autodoc/manualdoc 缺口、过时 API、示例风险。
3. 评估项目可维护性、类型标注、lint/mypy/ruff、CI/测试策略，并针对未来 MCP 和 skills 支持提出文档/API 设计建议。
4. 对功能正确性、可读性、健壮性、性能、测试、风格/规范、文档/注释、可维护性、并发/线程安全九个维度各给 1-5 分，并说明理由。
5. 命名与可读性专项：列出不良命名、建议替代名、理由；列出超过 50 行或职责不单一函数，给重构切分建议和接口草案。
6. 性能复杂度：对数据下载、目录扫描、测试性能、文档构建热点给 Big O 或瓶颈分析，提供优化与基准建议。
7. 并发与线程安全：重点分析下载、全局 logger、warnings、缓存/用户目录共享状态。
8. 输出一份可执行的阶段性重构路线图，能被主代理合并到总报告。

返回中文结构化 Markdown，包含具体文件/函数名和尽量具体的行号或定位描述。不要写文件，只返回结果。

# EasyIDP 只读代码审查报告

审查范围基于 `/home/crest/Documents/Github/EasyIDP/repomix-output.xml` 定位，并只读查看了目标源码、文档、测试与 CI 配置；未修改文件。

## 0. 范围与关键文件

- `/home/crest/Documents/Github/EasyIDP/src/easyidp/__init__.py`
- `/home/crest/Documents/Github/EasyIDP/src/easyidp/data.py`
- `/home/crest/Documents/Github/EasyIDP/src/easyidp/logger.py`
- `/home/crest/Documents/Github/EasyIDP/pyproject.toml`
- `/home/crest/Documents/Github/EasyIDP/README.md`
- `/home/crest/Documents/Github/EasyIDP/docs/python_api/index.rst`
- `/home/crest/Documents/Github/EasyIDP/docs/python_api/**/*.rst`
- `/home/crest/Documents/Github/EasyIDP/tests/test_init_class_func.py`
- `/home/crest/Documents/Github/EasyIDP/tests/test_data.py`
- `/home/crest/Documents/Github/EasyIDP/tests/` 整体组织
- `/home/crest/Documents/Github/EasyIDP/.github/workflows/pytest.yml`

---

## 1. 包级 API 暴露、数据集下载、日志/告警、依赖与测试组织

### 1.1 包级 API 暴露

文件：`src/easyidp/__init__.py`

主要暴露：

- 版本：
  - `__version__ = "2.0.3"`，行 1。
- 工具函数：
  - `_find_key()`，行 20-28。
  - `get_full_path()`，行 36-42。
  - `parse_relative_path()`，行 45-53。
  - `user_data_dir()`，行 56-109。
  - `logged_input()`，行 119-140。
- 包级对象/模块：
  - `Container`，行 17。
  - `visualize`, `cvtools`, `geotools`, `shp`, `jsonfile`, `reconstruct`, `data`，行 147-155。
  - `PointCloud`, `GeoTiff`, `Pix4D`, `Metashape`, `ProjectPool`, `ROI`，行 157-162。
- 日志：
  - `init_easyidp_logger`, `logger`, `setup_logger`，行 11。
- 数据源状态：
  - `aliyun_down = None`，行 169。
  - `GOOGLE_AVAILABLE = True`，行 170。

主要问题：

1. **导入时副作用很重**

   - 行 116：导入包时立即初始化 logger 并输出 banner。
   - 行 172：导入包时立即访问 Google Drive 测试 URL。
   - 行 175-204：Google 不可达时尝试导入 `oss2`，若缺失则自动执行 `pip install oss2`。
   - 风险：导入 `easyidp` 可能阻塞、失败、污染环境、触发网络访问，不利于库、CI、MCP server、skills agent 或离线文档构建使用。
2. **循环依赖/隐式导入**

   - `data.py` 行 14：`from easyidp import user_data_dir, logged_input`。
   - `__init__.py` 行 154 又导入 `data`。
   - 当前可运行依赖于导入时序，但设计较脆弱。建议将 `user_data_dir` 与 `logged_input` 下沉到独立模块，如 `easyidp._paths`、`easyidp._io`。
3. **公共 API 边界不清**

   - 没有 `__all__`。
   - `_find_key` 私有命名却在包级文件中；`get_full_path`、`parse_relative_path` 未进入 API 文档。
   - `logged_input` 是交互式 I/O，应避免包级默认暴露或至少标记为内部工具。

---

### 1.2 数据集下载机制

文件：`src/easyidp/data.py`

核心类/函数：

- `show_data_dir()`，行 17-36。
- `url_checker()`，行 39-65。
- `_can_access_google_cloud()`，行 68-69。
- `download_all()`，行 72-77。
- `AliYunDownloader`，行 79-223。
- `EasyidpDataSet`，行 225-377。
- `Lotus`，行 379-465。
- `ForestBirds`，行 467-523。
- `GDownTest`，行 526-534。
- `TestData`，行 537-909。

下载流程：

1. `EasyidpDataSet.__init__()` 行 270-279 设置 `name/gdrive_url/size/data_dir/zip_file/pix4d/metashape` 并立即调用 `load_data()`。
2. `load_data()` 行 281-299：
   - 若数据目录不存在，先下载 zip，再解压。
3. `_download_data()` 行 311-355：
   - 若 `idp.GOOGLE_AVAILABLE` 为真，使用 `gdown.download()`。
   - 否则交互询问是否位于中国大陆。
   - 若确认，使用全局 `idp.aliyun_down` 下载。
4. `_unzip_data()` 行 357-368：
   - `zipfile.ZipFile.extractall(self.data_dir)` 解压。
   - 解压后删除 zip。

主要问题：

1. **实例化即下载**

   - `Lotus()`、`ForestBirds()`、`TestData()`、`GDownTest()` 都可能触发网络和磁盘写入。
   - 对文档示例、测试、MCP 工具调用非常危险。
2. **`download_all()` 逻辑不完整**

   - 行 74-76 只实例化 `Lotus`, `GDownTest`, `TestData`，未包含 `ForestBirds`。
   - 局部变量 `lotus/gd/test` 未使用。
   - 函数无返回值、无错误汇总、无参数控制。
3. **Aliyun 认证与下载存在健壮性问题**

   - `AliYunDownloader.__init__()` 行 87：`requests.get(access_url)` 无 timeout。
   - 行 93：`content.split("\r\n")` 对换行格式过于脆弱。
   - 行 144-186：`download_auth()` 混合成本计算、用户提示、ANSI 格式、输入校验，职责过多。
   - 行 207-223：`download()` 没有临时文件/原子替换，也没有异常时关闭进度条的 `finally`。
4. **Zip Slip 安全风险**

   - `_unzip_data()` 行 359-360 直接 `extractall(self.data_dir)`。
   - 若 zip 内含 `../` 或绝对路径，可能写出目标目录。
   - 当前数据源可信但库级实现仍应防护。
5. **并发冲突**

   - 所有数据集默认写入同一个 `user_data_dir()`。
   - 多进程测试、多个 Python 进程、多个 MCP 调用同时下载/解压/删除同一目录会竞态。

---

### 1.3 日志/告警机制

文件：`src/easyidp/logger.py`

主要结构：

- `BraceStyleAdapter`，行 38-109。
- `logger = BraceStyleAdapter(_base_logger, {})`，行 112。
- `DuplicateThrottleFilter`，行 115-186。
- `TqdmHandler`，行 189-209。
- `_default_log_file()`，行 212-229。
- `setup_logger()`，行 274-337。
- `init_easyidp_logger()`，行 340-360。

优点：

- 使用专用 logger 名称 `easyidp`，行 20。
- `propagate = False`，行 313，避免污染 root logger。
- 支持 `RotatingFileHandler`，行 326-334。
- `TqdmHandler` 可减少与进度条冲突。
- `setup_logger(reset=True)` 可重置 handler，行 304-306。

主要问题：

1. **全局状态非线程安全**

   - `_STATE = {"configured": False}`，行 26。
   - `setup_logger()` 修改 handler 与 `_STATE` 无锁。
   - 多线程同时初始化可能重复 handler 或关闭其他线程正在使用的 handler。
2. **`DuplicateThrottleFilter` 非线程安全**

   - `_last_msg` 与 `_last_times` 行 133-134 无锁。
   - 高并发日志可能丢失非重复消息或产生不一致节流。
3. **日志级别控制不完整**

   - stream handler 固定 `INFO`，行 317。
   - 即使 `setup_logger(level="DEBUG")`，控制台仍不输出 DEBUG。
   - 这可能与用户预期不一致。
4. **导入时输出 banner**

   - `__init__.py` 行 116 调用 `init_easyidp_logger()`。
   - `init_easyidp_logger()` 行 359 输出欢迎信息。
   - 对库使用、测试和文档构建偏吵。
5. **告警机制混用**

   - 代码主要使用 `logger.warning()`，如 `__init__.py` 行 52、`data.py` 行 64、180。
   - 未使用 Python `warnings.warn()` 做可捕获的 API 级告警。
   - 对下游库用户不便于过滤 DeprecationWarning/UserWarning。

---

### 1.4 依赖

文件：`pyproject.toml`

运行依赖：行 25-45。

- 已声明较新版本：
  - `numpy>=2.2.5`
  - `matplotlib>=3.10.3`
  - `rasterio>=1.4.3`
  - `scipy>=1.15.3`
  - `shapely>=2.1.0`
- `gdown>=5.2.0` 已是运行依赖，行 27。
- `oss2` 只在 test group，行 63，但 `__init__.py` 行 175-204 可能运行时动态安装。

主要问题：

1. **运行时动态安装依赖是严重维护风险**

   - 若 Aliyun 是正式功能，应将 `oss2` 放入 optional dependency，如 `[project.optional-dependencies] aliyun = ["oss2>=..."]`。
   - 不应在 import 阶段 `pip install`。
2. **缺少 lint/type/test 配置**

   - `pyproject.toml` 无 `[tool.ruff]`。
   - 无 `[tool.mypy]`。
   - 无 `[tool.pytest.ini_options]`。
   - 虽然 test group 包含 `mypy`，行 62，但 CI 没有执行。
3. **缺少 optional dependency 分层**

   - 文档、测试已有 dependency-groups，但用户可选功能如 `aliyun`, `docs`, `dev`, `mcp` 没有标准 extras。
   - 建议为 PyPI 用户提供：
     - `easyidp[aliyun]`
     - `easyidp[docs]`
     - `easyidp[test]`
     - `easyidp[mcp]`

---

### 1.5 测试组织

测试文件：

- `/home/crest/Documents/Github/EasyIDP/tests/test_init_class_func.py`
- `/home/crest/Documents/Github/EasyIDP/tests/test_data.py`
- `/home/crest/Documents/Github/EasyIDP/tests/test_jsonfile.py`
- `/home/crest/Documents/Github/EasyIDP/tests/test_shp.py`
- `/home/crest/Documents/Github/EasyIDP/tests/test_cvtools.py`
- `/home/crest/Documents/Github/EasyIDP/tests/test_geotools.py`
- `/home/crest/Documents/Github/EasyIDP/tests/test_pix4d.py`
- `/home/crest/Documents/Github/EasyIDP/tests/test_reconstruct.py`
- `/home/crest/Documents/Github/EasyIDP/tests/test_pointcloud.py`
- `/home/crest/Documents/Github/EasyIDP/tests/test_visualize.py`
- `/home/crest/Documents/Github/EasyIDP/tests/test_metashape.py`
- `/home/crest/Documents/Github/EasyIDP/tests/test_roi.py`
- `/home/crest/Documents/Github/EasyIDP/tests/test_geotiff.py`
- `/home/crest/Documents/Github/EasyIDP/tests/test_back2raw_performance.py`

测试初始化：

- `tests/__init__.py` 行 9-14 设置 `IS_TESTING` 并重置 logger。
- 行 17-34 创建 `tests/out` 子目录。
- 行 37-79 提供 `shared_data` fixture，但 scope 为 `module`，会反复构造较重对象。
- 行 100-140 允许直接运行下载测试数据。

CI：

- `.github/workflows/pytest.yml` 行 18 测 Python 3.10-3.13。
- 行 19 `max-parallel: 1`，说明测试/下载存在共享资源冲突。
- 行 34-36 先下载 `TestData`。
- 行 37-39 只执行 `pytest`，未执行 ruff/mypy/docs build。

主要问题：

1. **大量测试依赖真实网络和共享缓存**

   - `tests/test_data.py` 行 24-43 会真实下载 `GDownTest`。
   - 行 49 在模块 import 时直接实例化 `AliYunDownloader()`，真实访问 Aliyun。
   - 行 132-141 真实下载 Aliyun 文件到 `tests/out/data_test/...`。
   - 单元测试和集成测试未分层。
2. **测试文件会修改工作区**

   - `tests/__init__.py` 行 17-34 创建输出目录。
   - 多个测试写入 `tests/out`。
   - 应统一迁移到 `tmp_path` 或 pytest fixture 管理。
3. **并行测试支持不完整**

   - test group 有 `pytest-xdist`，但 CI 未使用。
   - `test_gdown_ali_oss` 行 155-158 跳过 xdist，说明共享缓存有冲突。
   - CI 通过 `max-parallel: 1` 降低风险，但牺牲速度。

---

## 2. 文档与函数/类一致性

### 2.1 API 文档现状

`docs/python_api/index.rst`：

- 行 5-24 描述模块总览。
- 行 27-43 推荐高级 wrapper。
- 行 45-49 提醒 `Container` 相关对象不能 pickle。

`docs/python_api/data.rst`：

- 行 12-17 用户数据集：`Lotus`, `ForestBirds`。
- 行 68-72 开发数据集：`EasyidpDataSet`, `TestData`。
- 行 78-84 函数：`user_data_dir`, `show_data_dir`, `url_checker`, `download_all`。
- 行 37-64 有 Aliyun 下载说明。

### 2.2 autodoc/manualdoc 缺口

明显缺口：

1. `easyidp.logger` 未进入 API 文档

   - `setup_logger()`
   - `init_easyidp_logger()`
   - `BraceStyleAdapter`
   - `DuplicateThrottleFilter`
   - `TqdmHandler`
2. `easyidp.__init__` 包级工具未文档化

   - `get_full_path()`，`src/easyidp/__init__.py` 行 36-42。
   - `parse_relative_path()`，行 45-53。
   - `logged_input()`，行 119-140。
   - `_find_key()` 私有函数可不文档化，但应移到内部模块。
3. `easyidp.data.AliYunDownloader` 未进入 data.rst

   - 类定义行 79-223。
   - 这是当前中国大陆下载逻辑核心，却没有 autodoc 页面。
   - 建议只文档化用户可用配置/错误处理，不暴露凭证获取细节。
4. `easyidp.data.GDownTest` 未进入 data.rst

   - 类定义行 526-534。
   - 如仅测试使用，应改名 `_GDownTest` 或标注 internal。
5. 嵌套数据路径类未文档化

   - `TestData.JsonDataset`, `MetashapeDataset`, `Pix4Dataset`, 等行 669-909。
   - 当前 `TestData.__init__` docstring 用手写列表维护，容易过时。

### 2.3 过时或不一致 API/示例

1. `Lotus` 文档属性名错误

   - `docs/python_api/data.rst` 行 30 示例使用 `lotus.metashape.proj`。
   - 源码中 `Lotus.__init__()` 行 456 是 `self.metashape.project`。
   - `EasyidpDataSet.ReconsProj` 行 372 定义的是 `project`，不是 `proj`。
   - `GDownTest` 行 533 又使用 `self.pix4d.proj`，与 `ReconsProj.project` 不一致。
2. `Lotus.__init__` docstring 拼写/字段不一致

   - `src/easyidp/data.py` 行 413 写 `.pix4d.project`。
   - 但 `GDownTest` 和测试 `test_data.py` 行 35 使用 `pix4d.proj`。
   - 建议统一为 `project`，保留 `proj` deprecated alias。
3. `data.rst` 示例直接触发 3.3GB 下载

   - 行 22-27 示例 `idp.data.Lotus()` 会真实下载大数据。
   - 文档构建或交互式复制风险高。
   - 建议示例使用 `lazy=True` 或 `download=False` 参数。
4. `README.md` 依赖列表过时

   - 行 61 提到 `tifffile`，但 `pyproject.toml` 运行依赖未列出 `tifffile`。
   - 行 64 提到 `laspy/lasrs/lasio`，实际依赖是 `laspy` 与 `lazrs`，无 `lasio`。
   - README 行 25 有拼写错误：`packges`, `dependices`。
5. `download_all()` 文档过弱

   - `data.rst` 行 84 列出，但函数本身只实例化三类且不返回。
   - 容易让用户误解其完整性和下载体积。
6. API Summary 中 `ProjectPool` 未正确链接

   - `docs/python_api/index.rst` 行 47 写 ``ProjectPool``，不是 Sphinx class role。
   - 包级 `ProjectPool` 来源于 `easyidp.reconstruct` 行 161。

---

## 3. 维护性、类型、lint/mypy/ruff、CI/测试策略、MCP/skills 建议

### 3.1 可维护性

主要痛点：

- `src/easyidp/data.py` 909 行，混合下载、交互、路径常量、测试数据描述。
- `TestData` 嵌套类众多，路径硬编码，缺少 schema。
- 包导入阶段网络检测和动态安装依赖，破坏库的可预测性。
- 文档和 API 手工同步，已有属性名不一致。

建议：

- 拆分模块：
  - `easyidp.paths`: `user_data_dir`, path helpers。
  - `easyidp.data.datasets`: `DatasetSpec`, `EasyIDPDataSet`, `Lotus`, `ForestBirds`。
  - `easyidp.data.downloaders`: `GoogleDriveDownloader`, `AliyunDownloader`。
  - `easyidp.data.testing`: `TestData`。
  - `easyidp.logging`: logger setup。
- 引入显式数据集描述：
  - `@dataclass(frozen=True) DatasetSpec`
  - 字段：`name`, `size`, `gdrive_url`, `sha256`, `members`, `mirrors`。
- 默认 lazy：
  - `Lotus(download=True)` 或更安全：`Lotus(download=False)` + `.download()`。

### 3.2 类型标注

现状：

- `logger.py` 有较多类型标注。
- `__init__.py` 和 `data.py` 基本无类型标注。
- `EasyidpDataSet` 属性动态注入，mypy 很难检查。
- nested path dataset 类无类型声明。

建议：

- 为 public API 增加类型：
  - `user_data_dir(file_name: str | os.PathLike[str] = "") -> Path`
  - `url_checker(url: str, timeout: float = 3.0) -> bool`
  - `EasyidpDataSet.load_data(self) -> None`
  - `AliYunDownloader.download(self, dataset_name: str, output: str | Path) -> Path`
- 将 `ReconsProj` 改成 dataclass：
  ```python
  @dataclass
  class ReconsProjectPaths:
      project: Path | None = None
      param: Path | None = None
      dom: Path | None = None
      dsm: Path | None = None
      pcd: Path | None = None
  ```
- 添加 `py.typed`，逐步启用 mypy。

### 3.3 lint/mypy/ruff

现状：

- `pyproject.toml` 行 62 包含 `mypy`，但无配置。
- 未声明 `ruff` 依赖。
- CI 未运行 mypy/ruff。

建议：

- 添加 `ruff` 到 dev/test group。
- 添加基础配置：
  - line length 88 或 100。
  - 启用 `E,F,I,B,UP,SIM,ARG`，先不启用过严规则。
- mypy 分阶段：
  - 初期 `ignore_missing_imports = true`。
  - 对 `easyidp.logger`、新拆分 data 模块开启 strict。
- CI 添加：
  - `ruff check src tests`
  - `ruff format --check src tests`
  - `mypy src/easyidp`

### 3.4 CI/测试策略

建议测试分层：

- `unit`：无网络、无大文件、无共享缓存。
- `integration`：需要 TestData。
- `network`：真实 Google/Aliyun 下载。
- `slow`：性能和大数据测试。
- `docs`：Sphinx build。

建议 pytest markers：

```toml
[tool.pytest.ini_options]
markers = [
  "unit: fast tests without network",
  "integration: tests requiring local TestData",
  "network: tests requiring external network",
  "slow: long running tests",
]
```

CI 建议：

- PR 默认跑：
  - unit + lint + type。
- nightly 或手动跑：
  - integration + network + docs。
- 缓存数据：
  - 用 GitHub Actions cache 缓存 `easyidp.data`。
  - 数据完整性用 sha256 校验，不每次重新下载。

### 3.5 MCP 与 skills 支持建议

面向 MCP/agent 的 API 设计应避免交互和隐式副作用：

1. **导入零副作用**

   - 不网络访问。
   - 不动态安装。
   - 不输出 banner。
   - 不创建目录，除非显式调用。
2. **非交互下载 API**

   - 提供参数：
     - `download(dataset, mirror="auto", assume_yes=False, cache_dir=None)`
     - `dry_run=True`
     - `progress=False`
   - MCP 工具不应调用 `input()`。
3. **结构化结果**

   - 下载返回：
     ```python
     DownloadResult(
         dataset_name: str,
         cache_dir: Path,
         downloaded: bool,
         source: str,
         bytes: int,
         checksum_ok: bool,
     )
     ```
4. **可机器读取的 manifest**

   - `easyidp.data.list_datasets() -> list[DatasetInfo]`
   - `easyidp.data.get_dataset_info(name) -> DatasetInfo`
   - 文档同步从 manifest 生成。
5. **skills 文档建议**

   - 提供 `docs/agent_api/` 或 `docs/mcp/`：
     - 稳定 API 列表。
     - 副作用说明。
     - 数据下载策略。
     - 错误码/异常类型。
     - 线程/进程安全说明。

---

## 4. 九维度评分

| 维度          | 分数 | 理由                                                                                                      |
| ------------- | ---: | --------------------------------------------------------------------------------------------------------- |
| 功能正确性    |  3/5 | 主流程可用，但 `download_all()` 不完整，`proj/project` 不一致，导入时网络检测和动态安装可能导致失败。 |
| 可读性        |  3/5 | logger.py 较清晰；data.py 过长且职责混杂，命名不统一。                                                    |
| 健壮性        |  2/5 | 缺少 timeout、checksum、原子下载、Zip Slip 防护、错误恢复和并发锁。                                       |
| 性能          |  3/5 | 下载/解压本身线性；但导入时网络检查、测试重复初始化和文档示例大下载影响明显。                             |
| 测试          |  2/5 | 覆盖范围广，但真实网络/共享缓存/工作区输出太多，单元与集成未分层。                                        |
| 风格/规范     |  2/5 | 缺少 ruff 配置；中英文混杂；部分函数无类型和规范 docstring。                                              |
| 文档/注释     |  3/5 | API 文档框架完整，但 logger/Aliyun/包级工具缺口明显，示例存在过时属性和大下载风险。                       |
| 可维护性      |  2/5 | data.py 909 行，硬编码路径和动态属性多，导入副作用强。                                                    |
| 并发/线程安全 |  2/5 | 全局 logger 状态、全局 `aliyun_down`、共享缓存目录、下载/解压/删除都无锁。                              |

---

## 5. 命名与可读性专项

### 5.1 不良命名与替代建议

| 当前名称                     | 位置                              | 建议名称                                        | 理由                                              |
| ---------------------------- | --------------------------------- | ----------------------------------------------- | ------------------------------------------------- |
| `EasyidpDataSet`           | `data.py` 行 225                | `EasyIDPDataset`                              | 品牌缩写大小写统一，Dataset 通常不拆成 DataSet。  |
| `ReconsProj`               | `data.py` 行 370                | `ReconsProjectPaths`                          | 当前类实际是路径集合，不是项目对象。              |
| `pix4d.proj`               | `data.py` 行 533；测试行 35     | `pix4d.project`                               | 与 `ReconsProj.project` 行 372 统一。           |
| `AliYunDownloader`         | `data.py` 行 79                 | `AliyunDownloader` 或 `AliyunOSSDownloader` | 专有名词统一；突出 OSS。                          |
| `ali_down`                 | `__init__.py` 行 169；测试行 49 | `aliyun_downloader`                           | 避免缩写，表达对象含义。                          |
| `gd`                       | `data.py` 行 75；测试行 30      | `gdown_test` 或 `google_drive_test_data`    | 缩写不利于阅读。                                  |
| `ctn`                      | `test_init_class_func.py` 行 14 | `container`                                   | 测试代码应清晰。                                  |
| `p4d`, `ms`              | `tests/__init__.py` 行 51、56   | `pix4d_project`, `metashape_project`        | fixture 返回值更清楚。                            |
| `url_checker`              | `data.py` 行 39                 | `is_url_reachable`                            | 返回 bool，谓词命名更准确。                       |
| `_can_access_google_cloud` | `data.py` 行 68                 | `can_access_google_drive`                     | 实际检查 Google Drive 文件，不是泛 Google Cloud。 |
| `download_auth`            | `data.py` 行 144                | `confirm_download_cost`                       | 实际做用户确认，不是认证下载。                    |
| `logged_input`             | `__init__.py` 行 119            | `prompt_user` 或内部 `_logged_input`        | 交互式函数不宜作为普通包级 API。                  |
| `get_full_path`            | `__init__.py` 行 36             | `as_path` 或删除                              | 当前只把 str 转 Path，不保证 full/absolute。      |
| `parse_relative_path`      | `__init__.py` 行 45             | `resolve_metashape_relative_path`             | 注释说明仅用于 metashape frame.zip。              |

### 5.2 超过 50 行或职责不单一的函数/类

#### 1. `src/easyidp/data.py` 整体模块，行 1-909

问题：

- 下载器、数据集、测试数据、路径 schema、用户交互混在一个文件。
- 文件过长，不利于维护。

拆分草案：

```python
# easyidp/data/spec.py
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    size: str
    gdrive_url: str
    sha256: str | None = None

# easyidp/data/base.py
class Dataset:
    def __init__(self, spec: DatasetSpec, cache_dir: Path | None = None) -> None: ...

    def ensure_available(self, downloader: Downloader | None = None) -> Path: ...

# easyidp/data/downloaders.py
class Downloader(Protocol):
    def download(self, spec: DatasetSpec, output: Path) -> DownloadResult: ...

class GoogleDriveDownloader: ...
class AliyunOSSDownloader: ...
```

#### 2. `AliYunDownloader.download_auth()`，`data.py` 行 144-186

问题：

- 同时做成本计算、提示文案、ANSI 格式、输入循环、权限错误。
- 中英文/终端输出与业务逻辑耦合。

拆分建议：

```python
def build_cost_notice(dataset_name: str, cost: float) -> str: ...

def prompt_confirmation(expected: str, max_retries: int = 5) -> bool: ...

def confirm_download_cost(dataset_name: str, dataset_size: str) -> None: ...
```

#### 3. `EasyidpDataSet.__init__()`，`data.py` 行 228-279

问题：

- 初始化对象同时触发下载。
- docstring 很长，属性动态创建。

接口草案：

```python
class Dataset:
    def __init__(
        self,
        spec: DatasetSpec,
        cache_dir: Path | None = None,
        auto_download: bool = False,
    ) -> None:
        self.spec = spec
        self.cache_dir = cache_dir or user_data_dir()
        if auto_download:
            self.ensure_available()
```

#### 4. `EasyidpDataSet._download_data()`，`data.py` 行 311-355

问题：

- 混合 Google 可用性、URL 检查、Aliyun 交互、全局状态。
- 依赖 `easyidp as idp` 形成运行时耦合。

接口草案：

```python
def choose_downloader(
    mirror: Literal["auto", "google", "aliyun"],
    interactive: bool,
) -> Downloader: ...

def download_archive(
    spec: DatasetSpec,
    output: Path,
    downloader: Downloader,
) -> DownloadResult: ...
```

#### 5. `Lotus.__init__()`，`data.py` 行 401-465

问题：

- 手动填充大量硬编码路径。
- docstring 与代码混合维护。

建议：

```python
@dataclass(frozen=True)
class LotusPaths:
    photo: Path
    shp: Path
    pix4d: ReconsProjectPaths
    metashape: ReconsProjectPaths

def build_lotus_paths(data_dir: Path) -> LotusPaths: ...
```

#### 6. `TestData.__init__()`，`data.py` 行 544-668

问题：

- docstring 超长，路径集合职责过重。
- 测试资源 manifest 适合数据驱动生成。

建议：

```python
class TestData(Dataset):
    def paths(self) -> TestDataPaths: ...

@dataclass(frozen=True)
class TestDataPaths:
    json: JsonTestPaths
    shp: ShapefileTestPaths
    pcd: PointCloudTestPaths
    ...
```

#### 7. `tests/__init__.py shared_data()`，行 37-79

问题：

- fixture 太重，创建 ROI/Pix4D/Metashape/back2raw 多个对象。
- 多个测试共享复杂状态，失败定位困难。

拆分建议：

```python
@pytest.fixture(scope="session")
def test_data() -> TestData: ...

@pytest.fixture
def lotus_roi(test_data: TestData) -> ROI: ...

@pytest.fixture
def pix4d_project(test_data: TestData) -> Pix4D: ...

@pytest.fixture
def metashape_project(test_data: TestData) -> Metashape: ...
```

---

## 6. 性能复杂度与瓶颈分析

### 6.1 数据下载

位置：

- `EasyidpDataSet._download_data()`，`data.py` 行 311-355。
- `AliYunDownloader.download()`，行 207-223。

复杂度：

- 网络下载：`O(S)`，S 为 zip 文件大小。
- 磁盘写入：`O(S)`。
- Google URL 预检查：`O(1)` 网络请求，但延迟不可控。
- Aliyun 分片下载：理论 `O(S)`，并发由 `oss2.resumable_download` 内部控制。

瓶颈：

- 大数据集 `Lotus.size = "3.3GB"`，行 398。
- `ForestBirds.size = "1.97GB"`，行 486。
- `TestData.size = "344MB"`，行 541。
- 导入时 Google 检测增加冷启动延迟。

优化建议：

- 取消导入时 URL 检查，改为下载时 lazy check。
- 支持 HEAD 请求优先，避免 GET 读取正文。
- 增加 checksum，避免重复下载。
- 下载到 `.part` 临时文件，成功后原子 rename。
- 支持 `progress=False`，便于 CI/MCP。
- 缓存 manifest：记录版本、大小、sha256。

基准建议：

- 记录：
  - DNS/connect latency。
  - download throughput。
  - unzip time。
  - checksum time。
- 添加 benchmark：
  - 小文件 1KB。
  - 中等 TestData 344MB。
  - 大文件 Lotus 3.3GB，放 nightly。

### 6.2 解压

位置：`_unzip_data()` 行 357-368。

复杂度：

- 解压：`O(U)`，U 为解压后总大小。
- 文件创建：`O(N)`，N 为 zip entry 数量。

瓶颈：

- 大量小文件时 syscall 成本高。
- `extractall()` 无进度、无中断恢复。

优化建议：

- 安全解压逐 entry：
  - 校验目标路径在 `data_dir` 内。
  - 可选进度条。
- 解压到临时目录，成功后 rename。
- 支持文件锁，防止多进程同时解压。

### 6.3 目录扫描/测试数据路径

位置：`TestData` 嵌套类，`data.py` 行 669-909。

复杂度：

- 当前主要是硬编码路径，初始化 `O(1)`。
- 但真实瓶颈是 `TestData()` 触发下载与数据对象构造。

优化建议：

- 路径对象可 lazy property，不必一次全部构造。
- `TestData(download=False)` 可只返回路径和缺失状态。
- 添加 `validate()` 显式检查关键文件，复杂度 `O(N)`。

### 6.4 测试性能

瓶颈：

- `tests/__init__.py shared_data()` 行 37-79 构造多个重对象。
- CI `.github/workflows/pytest.yml` 行 34-39 每个 Python 版本都下载/准备数据。
- `max-parallel: 1` 行 19 降低 CI 并行度。

优化建议：

- 使用 GitHub cache 缓存 `~/.local/share/easyidp.data`。
- 单元测试默认 mock 下载器。
- `network` marker 单独 workflow。
- `shared_data` 拆为 session-scope 轻 fixture + function-scope 可变副本。

### 6.5 文档构建热点

风险：

- autodoc import `easyidp` 会触发 logger banner、Google 检测、可能安装 `oss2`。
- `data.rst` 示例如果未来启用 doctest，会下载 3.3GB。

优化建议：

- 文档构建设置环境变量：
  - `EASYIDP_NO_NETWORK=1`
  - `EASYIDP_QUIET=1`
- 包导入零副作用。
- 示例使用：
  ```python
  lotus = idp.data.Lotus(download=False)
  lotus.ensure_available()
  ```

---

## 7. 并发与线程安全

### 7.1 下载共享状态

问题点：

- `__init__.py` 行 169：全局 `aliyun_down = None`。
- `data.py` 行 342-343：懒创建 `idp.aliyun_down = AliYunDownloader()` 无锁。
- 多线程可能重复创建 downloader。
- 多进程同时下载同一 zip 可能互相覆盖。

建议：

- 使用文件锁：
  - lock path：`<cache_dir>/<dataset>.lock`
  - 下载、解压、删除都持锁。
- 全局 downloader 用 `threading.Lock` 或取消全局单例。
- 下载到唯一临时文件：
  - `<name>.zip.<pid>.<uuid>.part`
  - 校验成功后 `os.replace()`。

### 7.2 logger 全局状态

问题点：

- `_STATE` 行 26 无锁。
- `_reset_handlers()` 行 264-271 关闭 handler 时，其他线程可能正在 emit。
- `DuplicateThrottleFilter` 行 133-168 修改内部状态无锁。

建议：

- 增加模块级 `RLock` 包住 `setup_logger()` 和 `_reset_handlers()`。
- `DuplicateThrottleFilter` 内部使用 `threading.Lock`。
- 避免在库导入时重置/配置 logger；只添加 `NullHandler`，由用户显式启用。

### 7.3 warnings 与用户交互

问题点：

- `parse_relative_path()` 行 52 使用 logger warning，不是 `warnings.warn()`。
- `download_auth()` 行 175 使用 input，MCP/多线程/服务端环境会阻塞。
- `logged_input()` 行 132 调用全局 `input()`。

建议：

- API 行为告警使用 `warnings.warn(..., stacklevel=2)`。
- CLI 交互放入命令行层，不放底层库。
- 下载 API 参数化确认：
  - `assume_yes`
  - `interactive`
  - `confirm_callback`

### 7.4 用户目录共享状态

问题点：

- `user_data_dir()` 行 106-107 使用 `exists` 后 `makedirs`，存在 TOCTOU 竞态。
- `remove_data()` 行 306-309 直接 `shutil.rmtree(self.data_dir)`，可能删除其他进程正在使用的数据。
- `_unzip_data()` 行 363-364 解压后删除 zip，无锁。

建议：

- `Path.mkdir(parents=True, exist_ok=True)`。
- 删除前持锁。
- 对 `remove_data(force=False)` 增加保护：
  - 检查 lock。
  - 默认不删除正在使用的数据。
- 添加 `cache_dir` 参数，允许测试和 MCP 使用隔离目录。

---

## 8. 可执行阶段性重构路线图

### Phase 0：风险止血，低破坏

目标：不大改 API，先减少副作用与测试风险。

1. 移除 import 阶段动态安装 `oss2`

   - 位置：`__init__.py` 行 175-204。
   - 改为下载时提示安装 `easyidp[aliyun]`。
2. 增加环境变量跳过网络检测

   - 如 `EASYIDP_NO_NETWORK=1`。
   - `__init__.py` 行 172 改为 lazy 或可禁用。
3. `user_data_dir()` 使用 `mkdir(exist_ok=True)`

   - 位置：`__init__.py` 行 106-107。
4. `_unzip_data()` 增加 Zip Slip 防护

   - 位置：`data.py` 行 357-368。
5. 修复文档属性错误

   - `docs/python_api/data.rst` 行 30：`lotus.metashape.proj` 改 `lotus.metashape.project`。
   - 同步 GDownTest/test 的 `proj/project` 策略。

### Phase 1：测试分层与 CI

1. 添加 pytest markers：

   - `unit`, `integration`, `network`, `slow`。
2. 修改 `tests/test_data.py`

   - 模块级 `ali_down = idp.data.AliYunDownloader()` 行 49 改 fixture + mock。
   - 真实下载测试标记 `network`。
3. CI 拆分：

   - PR：unit + lint。
   - scheduled/manual：network + integration。
   - docs build 单独 job。
4. 引入 cache：

   - 缓存 `easyidp.data` 用户目录。
   - 加 checksum 校验。

### Phase 2：数据下载 API 重构

1. 新增 `DatasetSpec`。
2. 新增 downloader protocol：
   - `GoogleDriveDownloader`
   - `AliyunOSSDownloader`
3. `EasyidpDataSet` 支持：
   - `download=False`
   - `cache_dir=None`
   - `mirror="auto"`
   - `interactive=False`
4. 保留旧行为但发出 deprecation warning：
   - `Lotus()` 自动下载可逐步迁移到 `Lotus(download=True)`。

### Phase 3：文档与 API 清理

1. 添加 logger API 文档：

   - `docs/python_api/logger.rst`
   - autodoc `setup_logger`, `init_easyidp_logger`。
2. 添加 data downloader 文档：

   - 用户层：如何选择 mirror、如何配置 Aliyun。
   - 开发层：Downloader protocol。
3. 自动生成数据集表格：

   - 从 `DatasetSpec` manifest 生成 `data.rst`。
4. README 同步依赖列表：

   - 修正 `tifffile/lasio` 等过时项。

### Phase 4：类型与 lint

1. 添加 `ruff` 配置。
2. 添加 `mypy` 配置。
3. 从低风险模块开始：
   - `logger.py`
   - 新增 `data/spec.py`
   - 新增 `data/downloaders.py`
4. 动态路径类 dataclass 化。
5. 添加 `py.typed`。

### Phase 5：MCP/skills 友好 API

1. 导入零副作用。
2. 非交互 API 全覆盖。
3. 结构化返回 `DownloadResult`。
4. 机器可读：
   - `list_datasets()`
   - `get_dataset_info()`
   - `validate_dataset()`
5. 文档新增：
   - `docs/mcp/index.rst`
   - `docs/mcp/data_download.rst`
   - `docs/mcp/thread_safety.rst`

---

## 9. 高优先级问题清单

1. `__init__.py` 行 172-204：导入时网络检测 + 动态 pip install，建议最高优先级移除。
2. `data.py` 行 359-360：`extractall()` 无路径校验，建议修复 Zip Slip。
3. `data.py` 行 87：`requests.get()` 无 timeout。
4. `data.py` 行 49 测试模块级真实初始化 Aliyun downloader，建议改 mock fixture。
5. `docs/python_api/data.rst` 行 30：`metashape.proj` 与源码 `metashape.project` 不一致。
6. `data.py` 行 72-77：`download_all()` 不完整且无返回。
7. `logger.py` 行 274-337：logger 全局配置缺少锁。
8. `pyproject.toml`：缺少 ruff/mypy/pytest 配置，CI 未运行 lint/type。
9. `tests/__init__.py` 行 37-79：`shared_data` 过重，应拆分 fixture。

Nya~♡
