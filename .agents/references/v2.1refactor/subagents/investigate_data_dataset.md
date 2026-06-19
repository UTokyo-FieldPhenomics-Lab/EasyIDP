# User Prompt:

只读调查，不要修改任何文件。工作目录 /home/crest/Documents/Github/EasyIDP。

任务：调查 EasyIDP 的 data/dataset 类是否需要纳入 v2.1 新架构重构。

请阅读：

- src/easyidp/data.py
- src/easyidp/__init__.py 中与 user_data_dir、logged_input、Google/Aliyun 下载相关部分
- tests/test_data.py
- tests/__init__.py 中 TestData/shared_data fixture 相关部分
- docs/python_api/data.rst 和相关 autodoc
- .agents/rules/architecture.md 与 .agents/rules/roadmap.md
- 如果有必要，参考 .agents/report/20260608_easyidp_refactor_analysis.md 中 data/logger/docs 部分

请输出中文结构化 Markdown：

1. 当前 data/dataset 的职责、类结构、调用链。
2. 当前问题：导入副作用、下载时机、网络/交互、缓存目录、解压安全、测试耦合、文档一致性。
3. 是否有重构必要，必要性等级（高/中/低）和理由。
4. v2.1 推荐目标架构：DatasetSpec、DatasetRegistry、Downloader、DatasetBundle/TestDataPaths 等。
5. 哪些旧 API 可保留，哪些应 breaking/deprecate。
6. 与 future MCP/skills 的接口要求：JSON-serializable、dry-run、validate-only、无交互。
7. 建议是否写入 .agents/references/v2.1refactor/data_dataset_architecture.md，以及建议内容提纲。

返回结果即可，不写文件。

---

## 1. 当前 data/dataset 的职责、类结构、调用链

### 1.1 模块文件

| 文件                                   | 职责                                                                |
| -------------------------------------- | ------------------------------------------------------------------- |
| `src/easyidp/data.py` (909行)        | 数据集定义 + 下载逻辑 + 解压 + URL检查                              |
| `src/easyidp/__init__.py` (L165-204) | 导入时网络探测 + oss2 自动安装 + 全局状态变量                       |
| `src/easyidp/logger.py`              | 日志系统（与 data 耦合：`user_data_dir()` 共享路径约定）          |
| `tests/__init__.py`                  | `shared_data` fixture + `if __name__ == "__main__"` CI 下载入口 |

### 1.2 类层次

```
EasyidpDataSet (基类, L225)
├── Lotus (L379)           -- 用户 demo 数据集
├── ForestBirds (L467)     -- 用户 demo 数据集
├── GDownTest (L526)       -- Google Drive 连通性测试
└── TestData (L537)        -- 测试数据集 (450+行路径定义)
    ├── .metashape  -> MetashapeDataset (L669)
    ├── .pix4d     -> Pix4Dataset (L706)
    ├── .json      -> JsonDataset (L757)
    ├── .shp       -> ShapefileDataset (L774)
    ├── .pcd       -> PointCloudDataset (L820)
    ├── .roi       -> ROIDataset (L845)
    ├── .tiff      -> TiffDataSet (L853)
    ├── .cv        -> CVDataset (L878)
    ├── .vis       -> VisualDataset (L889)
    └── .b2r       -> Back2rawDataset (L900)
```

### 1.3 各层调用链

```
用户调用 idp.data.Lotus()
  └─ Lotus.__init__()
       ├─ user_data_dir("2017_tanashi_lotus")   -- 生成缓存路径
       ├─ super().load_data()
       │    ├─ _download_data()
       │    │    ├─ [Google可用] gdown.download(url, zip_file)
       │    │    └─ [Google不可用] logged_input("are you locate in China Mainland?")
       │    │         ├─ [Yes] AliYunDownloader.download_auth() → logged_input("我已知悉...")
       │    │         │    └─ AliYunDownloader.download() → oss2.resumable_download()
       │    │         └─ [No] ConnectionError
       │    └─ _unzip_data()
       │         └─ zipfile.ZipFile.extractall(data_dir)  -- 无路径检查
       └─ 设置 self.photo, self.shp, self.pix4d.*, self.metashape.* 路径

导入时:
  import easyidp
    └─ __init__.py L172: data._can_access_google_cloud()
         ├─ [可访问] GOOGLE_AVAILABLE = True
         └─ [不可访问] GOOGLE_AVAILABLE = False
              ├─ 尝试 import oss2
              └─ [失败] pip install oss2 (subprocess.run)
```

### 1.4 测试耦合

```
tests/__init__.py:shared_data fixture (module scope)
  └─ idp.data.TestData()                       -- 下载 344MB 测试数据
       ├─ idp.ROI(test_data.shp.lotus_shp)     -- 读取 shapefile
       ├─ idp.ROI()  + 4个key切片
       ├─ idp.Pix4D(...)                       -- 解析 Pix4D 工程
       ├─ idp.Metashape(...)                   -- 解析 Metashape 工程
       ├─ roi.get_z_from_dsm(...)              -- 提取高度
       └─ p4d.back2raw(roi)                    -- 反投影计算

6个测试文件在模块级别直接创建 TestData():
  test_cvtools.py:9, test_pix4d.py:8, test_shp.py:7,
  test_metashape.py:9, test_pointcloud.py:9, test_back2raw_performance.py:14
```

CI 流程:

```
pytest.yml:
  - pip install -e .
  - python ./tests/__init__.py    # 预下载 TestData
  - pytest
```

---

## 2. 当前问题清单

### 2.1 导入副作用（严重）

**`__init__.py` L172-204**：每次 `import easyidp` 都触发 `requests.get(GDOWN_TEST_URL)` 网络请求（3秒超时）。对国内用户会额外尝试 `pip install oss2`（`subprocess.run`），可能卡住或失败。

违反规则：

> Make every public workflow usable without import-time network access, hidden global state, or interactive prompts. (architecture.md L11)

### 2.2 下载时机问题

`EasyidpDataSet.__init__()` → `load_data()` → `_download_data()`：对象构造即下载，无法 dry-run 或延迟。`TestData.__init__()` 在构造函数中下载 344MB 数据，导致 6 个测试模块 import 时立即触发网络下载。

### 2.3 网络/交互阻塞

3处阻塞式交互：

- `_download_data()` L338: `logged_input("...are you locate in China Mainland? (Y/N)")`
- `download_auth()` L175: `logged_input(">>> ")` 最多 5 次重试
- `AliYunDownloader.__init__()` L87: `requests.get(access_url)` 获取访问密钥

全部不可被 MCP/skills 调用：无可 dry-run、不可 JSON 序列化、不可无头运行。

### 2.4 缓存目录

`user_data_dir()` 在 `__init__.py` 和 `logger.py` 中逻辑接近重复：

- `__init__.py:user_data_dir()` — 面向数据集
- `logger.py:_default_log_file()` — 面向日志文件
  两者都使用 `~/.local/share/easyidp.data` 但实现分散，且 `user_data_dir()` 在调用时自动 `makedirs`，有副作用。

### 2.5 解压安全

`_unzip_data()` L360：

```python
zip_ref.extractall(self.data_dir)
```

无任何路径遍历检查（zip slip 漏洞），恶意 zip 可写文件到 `data_dir` 之外。

### 2.6 测试耦合

- `shared_data` fixture 实例化 `TestData()` → 下载 344MB → 解析多个工程 → 反投影 → 全部作为 fixture 返回。任意数据集问题会导致所有测试失败。
- 6 个测试文件在模块级别创建 `TestData()`（import 即下载），并行测试冲突（`@pytest.mark.skipif` 显式跳过）。
- `if __name__ == "__main__"` 在 `tests/__init__.py` 中作为 CI 下载入口，语义不干净。

### 2.7 文档一致性

`docs/python_api/data.rst` 描述的 Lotus 示例与实际代码一致，但未提及：

- `AliYunDownloader` 类（无 autodoc）
- `GOOGLE_AVAILABLE` / `aliyun_down` 全局状态
- `download_all()` 函数的具体行为
- `TestData` 的子属性（json/shp/pcd等）的文档

autodoc 文件仅覆盖 `EasyidpDataSet`、`Lotus`、`ForestBirds`、`TestData`、`download_all`、`show_data_dir`、`user_data_dir`、`url_checker`，缺少 `AliYunDownloader`、`GDownTest`、`ReconsProj`。

---

## 3. 重构必要性与等级

**等级：高**

理由：

| 维度           | 严重性  | 说明                                                                        |
| -------------- | ------- | --------------------------------------------------------------------------- |
| 导入时网络访问 | 🔴 致命 | `__init__.py` 每次 import 发 HTTP 请求 + subprocess pip install           |
| 交互式阻塞     | 🔴 致命 | `logged_input()` 使模块无法被 MCP/skills/API 调用                         |
| 构造即下载     | 🔴 致命 | 不可 dry-run、不可延迟、不可仅校验                                          |
| zip slip 安全  | 🟠 高   | `extractall` 无路径检查，潜在任意文件写入                                 |
| 全局状态       | 🟠 高   | `GOOGLE_AVAILABLE`、`aliyun_down` 在 `__init__.py` 顶部突变           |
| 测试耦合       | 🟡 中   | 6 个模块级别 TestData() 调用，shared_data 链条过深                          |
| 类职责混杂     | 🟡 中   | `TestData` 900行包含数据集路径 + 下载逻辑 + 8个内部类                     |
| 架构文档已指定 | 🔵 参考 | `architecture.md` 已明确 `DatasetSpec/DatasetRegistry/Downloaders` 目标 |

结论：**当前 data 模块的架构问题直接影响 v3.0 MCP/skills 的可行性**。如果不重构，MCP tools 调用将因导入时副作用和交互式输入而完全失败。必须在 v2.1 中解决。

---

## 4. v2.1 推荐目标架构

根据 `architecture.md` L73-76 的建议并完善：

### 4.1 数据模型

```python
# easyidp/data/spec.py
@dataclass
class DatasetSpec:
    """纯数据描述，不触发下载"""
    name: str
    version: str
    size_bytes: int
    urls: list[str]           # 按优先级排序的下载 URL
    checksum: str | None      # sha256
    manifest: dict[str, str]   # {logical_name: relative_path_in_zip}
    description: str
    license: str

# easyidp/data/registry.py
class DatasetRegistry:
    """数据集注册表，惰性加载"""
    _specs: dict[str, DatasetSpec]  # name -> spec 映射

    @classmethod
    def list(cls) -> list[str]
    @classmethod
    def get(cls, name: str) -> DatasetSpec
    @classmethod
    def validate_all(cls) -> list[str]  # 返回无效 spec 列表
```

### 4.2 下载器

```python
# easyidp/data/downloader.py
class DatasetDownloader:
    """纯数据下载，无交互"""
    def __init__(self, spec: DatasetSpec, cache_root: Path | None = None)

    def dry_run(self) -> DownloadPlan   # 返回 JSON-serializable 计划
    def check_cached(self) -> bool       # 检查本地是否已有完整数据
    def validate(self) -> bool           # 校验 hash，不下载
    def download(self, progress_callback=None) -> Path  # 下载 + 解压，返回缓存路径
    @staticmethod
    def _safe_extract(zip_path: Path, dest_dir: Path) -> None  # 防 zip slip
```

### 4.3 测试数据访问

```python
# easyidp/data/testing.py 或 tests/conftest.py 中
class TestDataPaths:
    """返回 Path 对象，不构造重型对象"""
    def __init__(self, data_root: Path, output_root: Path):
        # 仅设定路径，不做任何 IO
        self.shp = ShapefilePaths(data_root, output_root)
        self.pcd = PointCloudPaths(data_root, output_root)
        ...

# tests/conftest.py
@pytest.fixture(scope="session")
def test_data_root():
    """统一管理测试数据下载：CI 预下载，本地可复用"""
    root = Path(os.environ.get("EASYIDP_TEST_DATA", "~/.cache/easyidp-test")).expanduser()
    # 惰性下载逻辑
    return root
```

### 4.4 目标模块布局

```
easyidp/
  data/
    __init__.py        # 导出公共 API，不触发网络
    spec.py            # DatasetSpec
    registry.py        # DatasetRegistry (内置 spec 清单)
    downloader.py      # DatasetDownloader
    _specs/            # 各数据集规格定义（轻量 JSON/YAML 或 dataclass）
      lotus.py
      forestbirds.py
      testdata.py
tests/
  conftest.py          # test_data_root fixture (替代 shared_data)
```

### 4.5 旧代码处理

`src/easyidp/data.py` 应拆分为：

- 数据定义 → `data/_specs/`
- 下载逻辑 → `data/downloader.py`
- 路径工具 `user_data_dir()` → 移入 `data/__init__.py` 或独立 `data/_paths.py`

`__init__.py` L165-204 的导入时副作用完全移除，改为：

```python
# __init__.py 不再包含任何网络请求
# DOWNLOAD_BACKENDS 在 downloader.py 中惰性发现
```

---

## 5. 旧 API 兼容策略

### 5.1 可直接移除（break）

| API / 行为                                                          | 理由                     |
| ------------------------------------------------------------------- | ------------------------ |
| `__init__.py` L172 `data._can_access_google_cloud()` 导入时调用 | 违反架构核心原则         |
| `__init__.py` L178 `pip install oss2`                           | 导入时安装包是反模式     |
| `__init__.py` 全局 `GOOGLE_AVAILABLE` / `aliyun_down`         | 全局可变状态             |
| `EasyidpDataSet.__init__()` 自动调用 `load_data()`              | 构造即副作用             |
| `EasyidpDataSet.ReconsProj` 内部类                                | 用 dataclass 替代        |
| `download_all()` 函数                                             | 默默下载所有数据集，危险 |
| `AliYunDownloader.calculate_download_cost()` 随机浮动费用         | 不精确，测试不稳定       |

### 5.2 可保留（deprecate）

| API                        | 迁移方式                                            |
| -------------------------- | --------------------------------------------------- |
| `idp.data.Lotus()`       | →`DatasetRegistry.get("lotus").download()`       |
| `idp.data.ForestBirds()` | →`DatasetRegistry.get("forestbirds").download()` |
| `idp.data.TestData()`    | → 测试专用惰性 fixture `test_data_root`          |
| `show_data_dir()`        | →`DatasetDownloader.show_cache_dir()` 或移除     |
| `url_checker()`          | →`DatasetDownloader.check_url()` (内部)          |

### 5.3 兼容 shim（推荐）

```python
# easyidp/data/__init__.py
import warnings

def Lotus():
    warnings.warn("idp.data.Lotus() is deprecated, use DatasetRegistry.get('lotus')", 
                  DeprecationWarning, stacklevel=2)
    from easyidp.data.registry import DatasetRegistry
    return DatasetRegistry.get("lotus").download(unzip=True)
```

`idp.data.TestData()` 应改为仅返回 `TestDataPaths`（不下载），测试需要的数据通过 `conftest.py` 的 fixture 管理。

---

## 6. MCP/skills 接口要求

### 6.1 JSON-serializable

所有公开 API 返回对象必须可 JSON 序列化。具体：

| 要求                            | 对应设计                                                                   |
| ------------------------------- | -------------------------------------------------------------------------- |
| `DatasetDownloader.dry_run()` | 返回 `DownloadPlan`（dict/JSON）包含 url、大小、本地缓存路径、是否需下载 |
| `DatasetRegistry.list()`      | 返回 `list[dict]` 每个包含 name、size、description                       |
| `DatasetRegistry.get(name)`   | 返回 `DatasetSpec` 纯数据（可 JSON 化）                                  |

`ReconsProj` 和子数据集的 path 属性应支持 `as_posix()` 或 `str()` 序列化。

### 6.2 dry-run / validate-only

```python
plan = downloader.dry_run()      # {"action": "download", "size": "344MB", ...}
downloader.validate()            # 校验 hash，不下载
downloader.check_cached()        # 检查本地缓存完整性
```

MCP tools 应先调用 `dry_run()` 获取计划，由用户确认后再调用 `download()`。

### 6.3 无交互

- 移除所有 `input()` / `logged_input()` 调用
- 阿里云费用确认改用回调或环境变量：`EASYIDP_ALIYUN_CONFIRM=true`
- 中国大陆用户检测通过检测超时/连接失败自动回退，不再需要手动确认

### 6.4 进度回调

```python
def download(self, progress_callback: Callable[[int, int], None] | None = None) -> Path
```

MCP 适配器可将回调转为 JSON progress 事件。

---

## 7. 是否建议撰写参考文档

**建议：是。** 应写入 `.agents/references/v2.1refactor/data_dataset_architecture.md`。

该文件在 `architecture.md` L73-76 中已有预留（`easyidp.data: DatasetSpec, DatasetRegistry, Downloaders`），但目前 `.agents/references/v2.1refactor/` 目录下尚无对应文档（glob 结果为空）。

### 建议内容提纲

```markdown
# EasyIDP v2.1 Data/Dataset 架构重构设计

## 1. 现状摘要
- 当前 data.py 的职责和问题列表（引用本调查报告）

## 2. 设计目标
- 导入零网络副作用
- 构造不触发下载
- 无交互 prompt
- 防 zip slip
- JSON-serializable 公开接口
- 测试数据惰性加载

## 3. 目标模块布局
- easyidp/data/{__init__,spec,registry,downloader,_paths,_specs/}

## 4. 核心数据类型
- DatasetSpec (dataclass)
- DownloadPlan (dataclass, JSON-serializable)
- TestDataPaths (dataclass, 纯路径映射)

## 5. 下载器设计
- DatasetDownloader 接口
- 后端插件：gdown / oss2 / requests (按 URL scheme 分发)
- 进度回调抽象
- 解压安全规范

## 6. 测试数据管理
- conftest.py test_data_root fixture
- 环境变量 EASYIDP_TEST_DATA 覆盖缓存路径
- 与 CI 的集成方案

## 7. 迁移计划
- Breaking changes 清单
- Deprecation 兼容层设计
- __init__.py 副作用移除步骤
- 测试迁移要点
```

---

*以上为本次调查的完整结构化结果。* Nya~♡
