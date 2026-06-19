# Data/Dataset v2.1 Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 EasyIDP 的 data/dataset 层重构为无导入网络副作用、显式下载、可配置数据目录、可测试且未来 MCP/skills 友好的 v2.1 架构。

**Architecture:** 保留用户侧短入口 `idp.data.Lotus()`、`idp.data.ForestBirds()`、`idp.data.TestData()`，但构造只做本地轻量状态检查，不下载、不解压、不交互。新增 `idp.config` 作为 JSON 持久化配置入口，`data` 层通过 `DatasetSpec`、`DatasetRegistry`、`Dataset`、`DatasetDownloader` 和安全解压明确组织职责；MCP/skills 直接调用 Python API，不依赖命令行包装。

**Tech Stack:** Python 3.10+, dataclasses, pathlib, json, pytest, gdown, oss2, tqdm, uv

---

## Confirmed Design Decisions

- Public dataset names remain short: `Lotus`, `ForestBirds`, `TestData`; do not require users to type `LotusDataset`.
- Constructor behavior is medium breaking: `idp.data.Lotus()` no longer downloads, but if existing local cache is complete, old downstream code can use the returned paths directly.
- Missing local data should trigger a concise `logger.warning()` with dry-run summary and `.download()` guidance when `notify_missing=True`.
- Dataset downloads, temporary archives, temporary extraction directories, manifests, and final extracted datasets live under the configured EasyIDP data directory.
- Persistent configuration uses JSON through `idp.config`; environment variables are not the primary configuration API.
- Initial config fields are `data_dir`, `log_level`, and `show_banner`.
- Aliyun mirror usage must be explicit and non-interactive: `download(mirror="aliyun", confirm=True)`.
- `DatasetRegistry` is an internal/advanced foundation; ordinary docs should lead with `idp.data.Lotus()` and helper functions such as `idp.data.list_datasets()`.
- No CLI is implemented in this phase; test setup, CI, MCP, and skills should use the Python API directly.

## Target File Structure

- Create: `src/easyidp/config.py`
  - Owns `EasyIDPConfig`, JSON load/save/reset, default data directory, and package settings.
- Replace file with package: `src/easyidp/data.py` -> `src/easyidp/data/`
  - `src/easyidp/data/__init__.py`: public data exports only.
  - `src/easyidp/data/spec.py`: `DatasetSpec`, `DownloadPlan`, `DownloadResult`, `DatasetValidationResult`.
  - `src/easyidp/data/registry.py`: alias resolution and built-in spec/factory registry.
  - `src/easyidp/data/dataset.py`: base `Dataset`, short user-facing dataset classes, path helpers.
  - `src/easyidp/data/_builtin.py`: built-in dataset specs and logical file maps.
  - `src/easyidp/data/paths.py`: data root, archive path, temporary path helpers.
  - `src/easyidp/data/extract.py`: safe zip extraction.
  - `src/easyidp/data/downloader.py`: mirror selection, download, checksum, atomic extraction.
  - `src/easyidp/data/testing.py`: `TestData` path bundle classes.
  - `src/easyidp/data/errors.py`: structured data exceptions.
- Modify: `src/easyidp/__init__.py`
  - Export `config`; remove import-time network checks, `GOOGLE_AVAILABLE`, `aliyun_down`, runtime `oss2` installation, and data-specific `user_data_dir`.
- Modify: `src/easyidp/logger.py`
  - Read log level from `idp.config` after config exists, while avoiding circular import.
- Modify: `tests/test_data.py`, `tests/__init__.py`, data-consuming tests.
  - Replace module-level data downloads with fixtures and explicit skip/error behavior.
- Create: `tests/test_config.py`, `tests/test_data_registry.py`, `tests/test_data_dataset.py`, `tests/test_data_downloader.py`.
- Modify docs: `docs/python_api/data.rst`, `docs/contribute.rst`, relevant autodoc pages.

---

### Task 1: Add JSON-backed `idp.config`

**Files:**

- Create: `src/easyidp/config.py`
- Modify: `src/easyidp/__init__.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Add `tests/test_config.py`:

```python
from pathlib import Path

import easyidp as idp
from easyidp.config import EasyIDPConfig


def test_default_config_uses_default_data_dir(tmp_path):
    config = EasyIDPConfig(config_path=tmp_path / "config.json")

    assert config.data_dir.name == "easyidp.data"
    assert config.log_level == "INFO"
    assert config.show_banner is True


def test_update_changes_session_without_saving(tmp_path):
    config_path = tmp_path / "config.json"
    config = EasyIDPConfig(config_path=config_path)

    config.update(data_dir=tmp_path / "data", log_level="DEBUG", show_banner=False)

    assert config.data_dir == tmp_path / "data"
    assert config.log_level == "DEBUG"
    assert config.show_banner is False
    assert not config_path.exists()


def test_save_and_reload_json_config(tmp_path):
    config_path = tmp_path / "config.json"
    config = EasyIDPConfig(config_path=config_path)
    config.update(data_dir=tmp_path / "data", log_level="WARNING", show_banner=False)
    config.save()

    loaded = EasyIDPConfig(config_path=config_path)

    assert loaded.data_dir == tmp_path / "data"
    assert loaded.log_level == "WARNING"
    assert loaded.show_banner is False


def test_package_exports_config_entrypoint():
    assert hasattr(idp, "config")
    assert hasattr(idp.config, "get")
    assert hasattr(idp.config, "update")
    assert hasattr(idp.config, "save")
    assert hasattr(idp.config, "reset")
```

- [ ] **Step 2: Run failing tests**

Run: `uv run pytest tests/test_config.py -q`

Expected: fail because `easyidp.config` does not exist or lacks the tested API.

- [ ] **Step 3: Implement `EasyIDPConfig`**

Create `src/easyidp/config.py` with this public shape:

```python
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def default_config_path() -> Path:
    """Return the default EasyIDP JSON config path.

    Returns
    -------
    pathlib.Path
        Platform-specific config file path.

    Examples
    --------
    >>> default_config_path().name
    'config.json'

    Notes
    -----
    This function only computes a path and does not create files.
    """
    if sys.platform.startswith("win"):
        root = Path.home() / "AppData" / "Roaming"
    elif sys.platform.startswith("darwin"):
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path.home() / ".config"
    return root / "easyidp" / "config.json"


def default_data_dir() -> Path:
    """Return the default EasyIDP dataset directory.

    Returns
    -------
    pathlib.Path
        Platform-specific data root path.

    Examples
    --------
    >>> default_data_dir().name
    'easyidp.data'
    """
    if sys.platform.startswith("win"):
        root = Path.home() / "AppData" / "Local"
    elif sys.platform.startswith("darwin"):
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path.home() / ".local" / "share"
    return root / "easyidp.data"


@dataclass
class EasyIDPConfig:
    """JSON-backed package configuration.

    Parameters
    ----------
    config_path : pathlib.Path, optional
        JSON config file path. Defaults to the platform user config path.

    Returns
    -------
    EasyIDPConfig
        Mutable session configuration object.

    Examples
    --------
    >>> cfg = EasyIDPConfig()
    >>> cfg.update(log_level="DEBUG")
    >>> cfg.log_level
    'DEBUG'

    Notes
    -----
    Loading may happen at import time, but saving requires an explicit call.
    """
    config_path: Path | None = None
    data_dir: Path = field(default_factory=default_data_dir)
    log_level: str = "INFO"
    show_banner: bool = True

    def __post_init__(self) -> None:
        self.config_path = Path(self.config_path or default_config_path()).expanduser()
        self._load_if_exists()

    def get(self) -> "EasyIDPConfig":
        return self

    def update(self, **kwargs: Any) -> "EasyIDPConfig":
        for key, value in kwargs.items():
            if key == "data_dir":
                self.data_dir = Path(value).expanduser()
                continue
            if key in {"log_level", "show_banner"}:
                setattr(self, key, value)
                continue
            raise KeyError(f"Unknown EasyIDP config key: {key}")
        return self

    def save(self) -> Path:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return self.config_path

    def reset(self, save: bool = False) -> "EasyIDPConfig":
        self.data_dir = default_data_dir()
        self.log_level = "INFO"
        self.show_banner = True
        if save:
            self.save()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "log_level": self.log_level,
            "show_banner": self.show_banner,
        }

    def _load_if_exists(self) -> None:
        if not self.config_path.exists():
            return
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.update(**data)


config = EasyIDPConfig()
get = config.get
update = config.update
save = config.save
reset = config.reset
```

- [ ] **Step 4: Export config in package init without data side effects**

In `src/easyidp/__init__.py`, add near logger imports:

```python
from . import config
```

Do not add network checks or file writes.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_config.py -q`

Expected: all `tests/test_config.py` tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/easyidp/config.py src/easyidp/__init__.py tests/test_config.py
git commit -m "feat(config): add json-backed package configuration"
```

---

### Task 2: Remove import-time dataset network side effects

**Files:**

- Modify: `src/easyidp/__init__.py`
- Test: `tests/test_init_class_func.py` or create `tests/test_import_side_effects.py`

- [ ] **Step 1: Write failing import-side-effect test**

Create `tests/test_import_side_effects.py`:

```python
import importlib
import subprocess

import requests


def test_import_easyidp_does_not_check_network_or_install(monkeypatch):
    def fail_get(*args, **kwargs):
        raise AssertionError("import easyidp must not call requests.get")

    def fail_run(*args, **kwargs):
        raise AssertionError("import easyidp must not run subprocess")

    monkeypatch.setattr(requests, "get", fail_get)
    monkeypatch.setattr(subprocess, "run", fail_run)

    import easyidp

    importlib.reload(easyidp)
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_import_side_effects.py -q`

Expected: fail under old implementation because `__init__.py` calls `data._can_access_google_cloud()`.

- [ ] **Step 3: Delete network and install logic from package init**

Remove from `src/easyidp/__init__.py`:

```python
aliyun_down = None
GOOGLE_AVAILABLE = True
```

Also delete the entire `if not data._can_access_google_cloud():` block, from the Google availability check through the final `ImportError` branch that tries to install or import `oss2`.

Also remove `subprocess` import if it becomes unused. Keep public imports stable:

```python
from . import data
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_import_side_effects.py -q`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/easyidp/__init__.py tests/test_import_side_effects.py
git commit -m "refactor(package): remove data network checks from import"
```

---

### Task 3: Convert `easyidp.data` into a focused package

**Files:**

- Delete after migration: `src/easyidp/data.py`
- Create: `src/easyidp/data/__init__.py`
- Create: `src/easyidp/data/spec.py`
- Create: `src/easyidp/data/errors.py`
- Test: `tests/test_data_registry.py`

- [ ] **Step 1: Write failing import/API shape tests**

Create `tests/test_data_registry.py`:

```python
import easyidp as idp


def test_data_public_api_exports_short_dataset_names():
    assert hasattr(idp.data, "Lotus")
    assert hasattr(idp.data, "ForestBirds")
    assert hasattr(idp.data, "TestData")


def test_data_public_api_exports_models_and_helpers():
    assert hasattr(idp.data, "DatasetSpec")
    assert hasattr(idp.data, "DownloadPlan")
    assert hasattr(idp.data, "DownloadResult")
    assert hasattr(idp.data, "list_datasets")
    assert hasattr(idp.data, "get_dataset")
    assert hasattr(idp.data, "get_spec")
```

- [ ] **Step 2: Run tests and record current behavior**

Run: `uv run pytest tests/test_data_registry.py -q`

Expected: fail until new package exports are implemented.

- [ ] **Step 3: Create data model and error modules**

Create `src/easyidp/data/errors.py`:

```python
class DatasetError(Exception):
    """Base exception for EasyIDP dataset operations."""


class DatasetNotFoundError(DatasetError):
    """Raised when a dataset alias or name is not registered."""


class DatasetDownloadError(DatasetError):
    """Raised when a dataset cannot be downloaded."""


class DatasetExtractError(DatasetError):
    """Raised when archive extraction fails or is unsafe."""


class DatasetChecksumError(DatasetError):
    """Raised when checksum validation fails."""


class DatasetMirrorConfirmationError(DatasetError):
    """Raised when a cost-sensitive mirror needs explicit confirmation."""
```

Create `src/easyidp/data/spec.py`:

```python
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class DatasetSpec:
    """Static dataset metadata.

    Parameters
    ----------
    name : str
        Stable dataset name used for directories and manifests.
    title : str
        Human-readable dataset title.
    version : str
        Dataset manifest version.
    size_bytes : int or None
        Approximate archive size in bytes.
    urls : tuple[str, ...]
        Download URLs or mirror descriptors in priority order.
    files : Mapping[str, str]
        Logical file keys mapped to dataset-relative paths.
    checksum : str, optional
        SHA256 checksum for the archive when available.
    description : str
        Short dataset description.

    Returns
    -------
    DatasetSpec
        Immutable dataset specification.

    Examples
    --------
    >>> spec = DatasetSpec("demo", "Demo", "1", None, (), {"shp": "plots.shp"})
    >>> spec.path_for(Path("/tmp/demo"), "shp")
    PosixPath('/tmp/demo/plots.shp')
    """
    name: str
    title: str
    version: str
    size_bytes: int | None
    urls: tuple[str, ...]
    files: Mapping[str, str]
    checksum: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))

    def path_for(self, root: Path, key: str) -> Path:
        return root / self.files[key]


@dataclass(frozen=True)
class DownloadPlan:
    """JSON-friendly dataset download plan."""
    dataset: str
    root: str
    archive: str
    ready: bool
    needs_download: bool
    size_bytes: int | None
    urls: tuple[str, ...]
    missing_files: tuple[str, ...]


@dataclass(frozen=True)
class DownloadResult:
    """JSON-friendly dataset download result."""
    dataset: str
    root: str
    archive: str
    downloaded: bool
    extracted: bool
    ready: bool
    warnings: tuple[str, ...] = ()
```

- [ ] **Step 4: Add temporary public exports**

Create `src/easyidp/data/__init__.py`:

```python
from .errors import (
    DatasetChecksumError,
    DatasetDownloadError,
    DatasetError,
    DatasetExtractError,
    DatasetMirrorConfirmationError,
    DatasetNotFoundError,
)
from .spec import DatasetSpec, DownloadPlan, DownloadResult


def list_datasets():
    return []


def get_dataset(name):
    raise DatasetNotFoundError(f"Dataset is not registered: {name}")


def get_spec(name):
    raise DatasetNotFoundError(f"Dataset is not registered: {name}")
```

Move `src/easyidp/data.py` out of import resolution by deleting it after all old behavior needed by this task has a package replacement. If deletion is too broad for this task, rename content into a temporary branch-only backup outside `src/` before committing.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_data_registry.py -q`

Expected: helper export test passes; short dataset names still fail until Task 5.

- [ ] **Step 6: Commit package conversion foundation**

```bash
git add src/easyidp/data tests/test_data_registry.py
git rm src/easyidp/data.py
git commit -m "refactor(data): split data module into package foundation"
```

---

### Task 4: Add built-in specs and registry

**Files:**

- Create: `src/easyidp/data/_builtin.py`
- Create: `src/easyidp/data/registry.py`
- Modify: `src/easyidp/data/__init__.py`
- Test: `tests/test_data_registry.py`

- [ ] **Step 1: Extend registry tests**

Append to `tests/test_data_registry.py`:

```python
def test_list_datasets_includes_builtin_aliases():
    names = idp.data.list_datasets()

    assert "lotus" in names
    assert "forestbirds" in names
    assert "test" in names


def test_get_spec_accepts_alias_and_real_name():
    by_alias = idp.data.get_spec("lotus")
    by_name = idp.data.get_spec("2017_tanashi_lotus")

    assert by_alias.name == "2017_tanashi_lotus"
    assert by_name.name == "2017_tanashi_lotus"
    assert by_alias.files["shp"] == "plots.shp"


def test_unknown_dataset_raises_clear_error():
    with pytest.raises(idp.data.DatasetNotFoundError, match="unknown"):
        idp.data.get_spec("unknown")
```

Ensure the file imports `pytest`:

```python
import pytest
```

- [ ] **Step 2: Run failing tests**

Run: `uv run pytest tests/test_data_registry.py -q`

Expected: fail because registry has no built-in specs.

- [ ] **Step 3: Define built-in specs**

Create `src/easyidp/data/_builtin.py` with the logical keys from current `data.py`:

```python
from .spec import DatasetSpec

LOTUS_SPEC = DatasetSpec(
    name="2017_tanashi_lotus",
    title="Tanashi Lotus",
    version="2.1",
    size_bytes=3_300_000_000,
    urls=(
        "gdrive://1SJmp-bG5SZrwdeJL-RnnljM2XmMNMF0j",
        "aliyun://easyidp-data/2017_tanashi_lotus.zip",
    ),
    files={
        "photo": "20170531/photos",
        "shp": "plots.shp",
        "pix4d.project": "20170531",
        "pix4d.param": "20170531/params",
        "pix4d.dom": "20170531/hasu_tanashi_20170531_Ins1RGB_30m_transparent_mosaic_group1.tif",
        "pix4d.dsm": "20170531/hasu_tanashi_20170531_Ins1RGB_30m_dsm.tif",
        "pix4d.pcd": "20170531/hasu_tanashi_20170531_Ins1RGB_30m_group1_densified_point_cloud.ply",
        "metashape.project": "170531.Lotus.psx",
        "metashape.param": "170531.Lotus.files",
        "metashape.dom": "170531.Lotus.outputs/170531.Lotus_dom.tif",
        "metashape.dsm": "170531.Lotus.outputs/170531.Lotus_dsm.tif",
        "metashape.pcd": "170531.Lotus.outputs/170531.Lotus.laz",
    },
    description="Lotus plot UAV reconstruction dataset from Tanashi, Tokyo.",
)

FORESTBIRDS_SPEC = DatasetSpec(
    name="2022_florida_forestbirds",
    title="Florida Forest Birds",
    version="2.1",
    size_bytes=1_970_000_000,
    urls=(
        "gdrive://1mXkzaoSSCAA87cxcMHKL6_VNlykRYxJr",
        "aliyun://easyidp-data/2022_florida_forestbirds.zip",
    ),
    files={
        "photo": "Hidden_Little_03_24_2022",
        "shp": "Hidden_Little_grid.shp",
        "metashape.project": "Hidden_Little_03_24_2022.psx",
        "metashape.param": "Hidden_Little_03_24_2022.files",
        "metashape.dom": "Hidden_Little_03_24_2022.tiff",
        "metashape.dsm": "Hidden_Little_03_24_2022_DEM.tif",
    },
    description="Forest ecology survey dataset provided by the University of Florida.",
)

TEST_DATA_SPEC = DatasetSpec(
    name="data_for_tests",
    title="EasyIDP Test Data",
    version="2.1",
    size_bytes=344_000_000,
    urls=(
        "gdrive://17b_17CofqIuCVOWMnD67_wOnWMtwF8bw",
        "aliyun://easyidp-data/data_for_tests.zip",
    ),
    files={
        "shp.lotus_shp": "shp_test/lotus_plots.shp",
        "shp.lotus_prj": "shp_test/lotus_plots.prj",
        "pix4d.lotus_folder": "pix4d/lotus_tanashi_full",
        "pix4d.lotus_param": "pix4d/lotus_tanashi_full/params",
        "pix4d.lotus_photos": "pix4d/lotus_tanashi_full/photos",
        "pix4d.lotus_dom": "pix4d/lotus_tanashi_full/hasu_tanashi_20170525_Ins1RGB_30m_transparent_mosaic_group1.tif",
        "pix4d.lotus_dsm": "pix4d/lotus_tanashi_full/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif",
        "metashape.lotus_psx": "metashape/Lotus.psx",
        "metashape.lotus_param": "metashape/Lotus.files",
        "metashape.lotus_dsm": "metashape/Lotus.files/170531.Lotus_dsm.tif",
    },
    description="Developer and package test data.",
)

BUILTIN_SPECS = {
    LOTUS_SPEC.name: LOTUS_SPEC,
    FORESTBIRDS_SPEC.name: FORESTBIRDS_SPEC,
    TEST_DATA_SPEC.name: TEST_DATA_SPEC,
}

ALIASES = {
    "lotus": LOTUS_SPEC.name,
    "forestbirds": FORESTBIRDS_SPEC.name,
    "forest_birds": FORESTBIRDS_SPEC.name,
    "test": TEST_DATA_SPEC.name,
    "testdata": TEST_DATA_SPEC.name,
}
```

- [ ] **Step 4: Implement registry**

Create `src/easyidp/data/registry.py`:

```python
from ._builtin import ALIASES, BUILTIN_SPECS
from .errors import DatasetNotFoundError
from .spec import DatasetSpec


class DatasetRegistry:
    """Registry for dataset specs and factories.

    Parameters
    ----------
    specs : dict[str, DatasetSpec], optional
        Initial spec mapping.
    aliases : dict[str, str], optional
        Alias to canonical name mapping.

    Returns
    -------
    DatasetRegistry
        Dataset metadata registry.

    Examples
    --------
    >>> registry = DatasetRegistry()
    >>> registry.get_spec("lotus").name
    '2017_tanashi_lotus'
    """

    def __init__(self, specs=None, aliases=None):
        self._specs = dict(specs or BUILTIN_SPECS)
        self._aliases = dict(aliases or ALIASES)

    def list(self) -> list[str]:
        return sorted(set(self._aliases) | set(self._specs))

    def resolve(self, name: str) -> str:
        key = name.lower()
        if key in self._aliases:
            return self._aliases[key]
        if name in self._specs:
            return name
        raise DatasetNotFoundError(f"Dataset is not registered: {name}")

    def get_spec(self, name: str) -> DatasetSpec:
        return self._specs[self.resolve(name)]


registry = DatasetRegistry()
```

- [ ] **Step 5: Wire public helpers**

Update `src/easyidp/data/__init__.py`:

```python
from .registry import DatasetRegistry, registry


def list_datasets():
    return registry.list()


def get_spec(name):
    return registry.get_spec(name)
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_data_registry.py -q`

Expected: registry tests pass except `get_dataset` and short dataset class exports if not implemented yet.

- [ ] **Step 7: Commit**

```bash
git add src/easyidp/data tests/test_data_registry.py
git commit -m "feat(data): add dataset specs and registry"
```

---

### Task 5: Implement lightweight dataset objects and path bundles

**Files:**

- Create: `src/easyidp/data/dataset.py`
- Create: `src/easyidp/data/testing.py`
- Create: `src/easyidp/data/paths.py`
- Modify: `src/easyidp/data/__init__.py`
- Test: `tests/test_data_dataset.py`

- [ ] **Step 1: Write failing lightweight constructor tests**

Create `tests/test_data_dataset.py`:

```python
from pathlib import Path

import easyidp as idp


def test_lotus_constructor_does_not_download_or_create_root(tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)

    assert lotus.name == "2017_tanashi_lotus"
    assert lotus.root == tmp_path / "2017_tanashi_lotus"
    assert lotus.shp == lotus.root / "plots.shp"
    assert lotus.pix4d.dom.name.endswith("mosaic_group1.tif")
    assert not lotus.root.exists()
    assert lotus.is_ready() is False


def test_lotus_is_ready_when_required_files_exist(tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)
    for path in [lotus.shp, lotus.pix4d.dom, lotus.pix4d.dsm, lotus.metashape.project]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("demo", encoding="utf-8")

    assert lotus.is_ready() is True


def test_missing_dataset_dry_run_reports_missing_files(tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)
    plan = lotus.dry_run()

    assert plan.dataset == "2017_tanashi_lotus"
    assert plan.needs_download is True
    assert "plots.shp" in plan.missing_files


def test_testdata_constructor_builds_nested_path_bundles(tmp_path):
    data = idp.data.TestData(cache_root=tmp_path, test_out=tmp_path / "out", notify_missing=False)

    assert data.shp.lotus_shp == data.root / "shp_test" / "lotus_plots.shp"
    assert data.pix4d.lotus_folder == data.root / "pix4d" / "lotus_tanashi_full"
    assert data.metashape.lotus_psx == data.root / "metashape" / "Lotus.psx"
```

- [ ] **Step 2: Run failing tests**

Run: `uv run pytest tests/test_data_dataset.py -q`

Expected: fail because dataset classes are not implemented.

- [ ] **Step 3: Implement data path helpers**

Create `src/easyidp/data/paths.py`:

```python
from pathlib import Path

from easyidp import config


def user_data_dir(file_name: str | Path = "") -> Path:
    """Return the configured EasyIDP data path.

    Parameters
    ----------
    file_name : str or pathlib.Path, optional
        Optional child path inside the configured data directory.

    Returns
    -------
    pathlib.Path
        Configured data directory or child path.

    Examples
    --------
    >>> user_data_dir().name
    'easyidp.data'

    Notes
    -----
    This helper creates the root directory for compatibility with the old API.
    """
    root = Path(config.get().data_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root / file_name


def resolve_cache_root(cache_root: str | Path | None) -> Path:
    if cache_root is not None:
        return Path(cache_root).expanduser()
    return user_data_dir()
```

- [ ] **Step 4: Implement lightweight dataset base and short classes**

Create `src/easyidp/data/dataset.py`:

```python
from dataclasses import dataclass
from pathlib import Path

from easyidp.logger import logger

from .paths import resolve_cache_root
from .registry import registry
from .spec import DatasetSpec, DownloadPlan


@dataclass(frozen=True)
class ReconstructionPaths:
    project: Path | None = None
    param: Path | None = None
    dom: Path | None = None
    dsm: Path | None = None
    pcd: Path | None = None


class Dataset:
    """Lightweight dataset object with explicit download actions.

    Parameters
    ----------
    name : str
        Dataset alias or canonical name.
    cache_root : str or pathlib.Path, optional
        Per-object data root override.
    notify_missing : bool, default True
        Warn when required files are missing.

    Returns
    -------
    Dataset
        Dataset path and state wrapper.

    Examples
    --------
    >>> lotus = Lotus(notify_missing=False)
    >>> lotus.name
    '2017_tanashi_lotus'
    """

    required_keys: tuple[str, ...] = ()

    def __init__(self, name: str, cache_root=None, notify_missing: bool = True):
        self.spec = registry.get_spec(name)
        self.name = self.spec.name
        self.cache_root = resolve_cache_root(cache_root)
        self.root = self.cache_root / self.name
        self.archive = self.cache_root / ".downloads" / f"{self.name}.zip"
        if notify_missing and not self.is_ready():
            self._warn_missing()

    def path(self, key: str) -> Path:
        return self.spec.path_for(self.root, key)

    def is_ready(self) -> bool:
        return not self._missing_files()

    def dry_run(self, mirror: str = "auto") -> DownloadPlan:
        missing = self._missing_files()
        return DownloadPlan(
            dataset=self.name,
            root=str(self.root),
            archive=str(self.archive),
            ready=not missing,
            needs_download=bool(missing),
            size_bytes=self.spec.size_bytes,
            urls=self.spec.urls,
            missing_files=tuple(missing),
        )

    def validate(self) -> bool:
        return self.is_ready()

    def download(self, mirror="auto", *, confirm=False, force=False, progress=True):
        from .downloader import DatasetDownloader

        downloader = DatasetDownloader(self, mirror=mirror, confirm=confirm, progress=progress)
        return downloader.download(force=force)

    def _missing_files(self) -> list[str]:
        keys = self.required_keys or tuple(self.spec.files)
        return [self.spec.files[key] for key in keys if not self.path(key).exists()]

    def _warn_missing(self) -> None:
        plan = self.dry_run()
        logger.warning(
            "Dataset '%s' is not downloaded. Root: %s. Missing files: %s. "
            "Call `.download()` to download it or `.dry_run()` for details.",
            plan.dataset,
            plan.root,
            ", ".join(plan.missing_files[:5]),
        )


class Lotus(Dataset):
    required_keys = ("shp", "pix4d.dom", "pix4d.dsm", "metashape.project")

    def __init__(self, cache_root=None, notify_missing: bool = True):
        super().__init__("lotus", cache_root=cache_root, notify_missing=notify_missing)
        self.photo = self.path("photo")
        self.shp = self.path("shp")
        self.pix4d = ReconstructionPaths(
            project=self.path("pix4d.project"),
            param=self.path("pix4d.param"),
            dom=self.path("pix4d.dom"),
            dsm=self.path("pix4d.dsm"),
            pcd=self.path("pix4d.pcd"),
        )
        self.metashape = ReconstructionPaths(
            project=self.path("metashape.project"),
            param=self.path("metashape.param"),
            dom=self.path("metashape.dom"),
            dsm=self.path("metashape.dsm"),
            pcd=self.path("metashape.pcd"),
        )


class ForestBirds(Dataset):
    required_keys = ("shp", "metashape.project", "metashape.dom", "metashape.dsm")

    def __init__(self, cache_root=None, notify_missing: bool = True):
        super().__init__("forestbirds", cache_root=cache_root, notify_missing=notify_missing)
        self.photo = self.path("photo")
        self.shp = self.path("shp")
        self.metashape = ReconstructionPaths(
            project=self.path("metashape.project"),
            param=self.path("metashape.param"),
            dom=self.path("metashape.dom"),
            dsm=self.path("metashape.dsm"),
        )
```

- [ ] **Step 5: Implement `TestData` path bundles**

Create `src/easyidp/data/testing.py` with the current nested classes from `data.py`, converted into focused path classes. Include at least the paths already asserted in tests first:

```python
from pathlib import Path

from .dataset import Dataset


class ShapefilePaths:
    def __init__(self, root: Path, test_out: Path):
        self.data_dir = root
        self.lotus_shp = root / "shp_test" / "lotus_plots.shp"
        self.lotus_prj = root / "shp_test" / "lotus_plots.prj"
        self.out = test_out / "shp_test"

    def __truediv__(self, other):
        return self.data_dir / "shp_test" / other


class Pix4DPaths:
    def __init__(self, root: Path):
        self.lotus_folder = root / "pix4d" / "lotus_tanashi_full"
        self.lotus_param = self.lotus_folder / "params"
        self.lotus_photos = self.lotus_folder / "photos"
        self.lotus_dom = self.lotus_folder / "hasu_tanashi_20170525_Ins1RGB_30m_transparent_mosaic_group1.tif"
        self.lotus_dsm = self.lotus_folder / "hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif"


class MetashapePaths:
    def __init__(self, root: Path, test_out: Path):
        self.lotus_psx = root / "metashape" / "Lotus.psx"
        self.lotus_param = root / "metashape" / "Lotus.files"
        self.lotus_dsm = root / "metashape" / "Lotus.files" / "170531.Lotus_dsm.tif"


class TestData(Dataset):
    required_keys = ("shp.lotus_shp", "pix4d.lotus_folder", "metashape.lotus_psx")

    def __init__(self, test_out="./tests/out", cache_root=None, notify_missing: bool = True):
        super().__init__("test", cache_root=cache_root, notify_missing=notify_missing)
        out = Path(test_out)
        self.shp = ShapefilePaths(self.root, out)
        self.pix4d = Pix4DPaths(self.root)
        self.metashape = MetashapePaths(self.root, out)
```

During implementation, copy the full path list from old `TestData` into this module before deleting old `data.py` behavior. Keep each path bundle under 50 executable lines by splitting `JsonPaths`, `PointCloudPaths`, `RoiPaths`, `TiffPaths`, `CvPaths`, `VisualPaths`, and `Back2RawPaths`.

- [ ] **Step 6: Wire exports and factories**

Update `src/easyidp/data/__init__.py`:

```python
from .dataset import Dataset, ForestBirds, Lotus, ReconstructionPaths
from .paths import user_data_dir
from .testing import TestData


def get_dataset(name, cache_root=None, notify_missing=True):
    resolved = registry.resolve(name)
    if resolved == "2017_tanashi_lotus":
        return Lotus(cache_root=cache_root, notify_missing=notify_missing)
    if resolved == "2022_florida_forestbirds":
        return ForestBirds(cache_root=cache_root, notify_missing=notify_missing)
    if resolved == "data_for_tests":
        return TestData(cache_root=cache_root, notify_missing=notify_missing)
    raise DatasetNotFoundError(f"Dataset is not registered: {name}")
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_data_registry.py tests/test_data_dataset.py -q`

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add src/easyidp/data tests/test_data_registry.py tests/test_data_dataset.py
git commit -m "feat(data): add lightweight dataset path objects"
```

---

### Task 6: Implement safe extraction

**Files:**

- Create: `src/easyidp/data/extract.py`
- Test: `tests/test_data_downloader.py`

- [ ] **Step 1: Write safe extraction tests**

Create `tests/test_data_downloader.py`:

```python
import zipfile

import pytest

from easyidp.data.errors import DatasetExtractError
from easyidp.data.extract import safe_extract_zip


def test_safe_extract_zip_extracts_normal_members(tmp_path):
    archive = tmp_path / "demo.zip"
    dest = tmp_path / "dest"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("folder/file.txt", "content")

    safe_extract_zip(archive, dest)

    assert (dest / "folder" / "file.txt").read_text(encoding="utf-8") == "content"


def test_safe_extract_zip_rejects_zip_slip(tmp_path):
    archive = tmp_path / "bad.zip"
    dest = tmp_path / "dest"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "bad")

    with pytest.raises(DatasetExtractError, match="Unsafe archive member"):
        safe_extract_zip(archive, dest)

    assert not (tmp_path / "escape.txt").exists()
```

- [ ] **Step 2: Run failing tests**

Run: `uv run pytest tests/test_data_downloader.py -q`

Expected: fail because `safe_extract_zip` does not exist.

- [ ] **Step 3: Implement safe extraction**

Create `src/easyidp/data/extract.py`:

```python
import zipfile
from pathlib import Path

from .errors import DatasetExtractError


def safe_extract_zip(archive: str | Path, dest: str | Path) -> None:
    """Extract a zip archive while rejecting path traversal.

    Parameters
    ----------
    archive : str or pathlib.Path
        Zip archive path.
    dest : str or pathlib.Path
        Destination directory.

    Returns
    -------
    None
        Files are written under `dest`.

    Examples
    --------
    >>> safe_extract_zip("dataset.zip", "dataset")

    Notes
    -----
    Every member destination is resolved before extraction to prevent zip slip.
    """
    archive = Path(archive)
    dest = Path(dest)
    dest_root = dest.resolve()
    with zipfile.ZipFile(archive, "r") as zip_file:
        for member in zip_file.infolist():
            target = (dest / member.filename).resolve()
            if target != dest_root and dest_root not in target.parents:
                raise DatasetExtractError(f"Unsafe archive member: {member.filename}")
        zip_file.extractall(dest)
```

- [ ] **Step 4: Run extraction tests**

Run: `uv run pytest tests/test_data_downloader.py -q`

Expected: extraction tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/easyidp/data/extract.py tests/test_data_downloader.py
git commit -m "fix(data): add safe dataset archive extraction"
```

---

### Task 7: Implement explicit downloader and mirror confirmation

**Files:**

- Create: `src/easyidp/data/downloader.py`
- Modify: `src/easyidp/data/dataset.py`
- Test: `tests/test_data_downloader.py`

- [ ] **Step 1: Add downloader behavior tests**

Append to `tests/test_data_downloader.py`:

```python
import shutil

import easyidp as idp

from easyidp.data.errors import DatasetMirrorConfirmationError


def test_download_skips_when_dataset_is_ready(tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)
    for path in [lotus.shp, lotus.pix4d.dom, lotus.pix4d.dsm, lotus.metashape.project]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("demo", encoding="utf-8")

    result = lotus.download(progress=False)

    assert result.downloaded is False
    assert result.extracted is False
    assert result.ready is True


def test_aliyun_download_requires_confirm(tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)

    with pytest.raises(DatasetMirrorConfirmationError, match="Aliyun mirror"):
        lotus.download(mirror="aliyun", progress=False)


def test_force_download_uses_downloader_backend(monkeypatch, tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("plots.shp", "demo")
        zf.writestr("20170531/hasu_tanashi_20170531_Ins1RGB_30m_transparent_mosaic_group1.tif", "demo")
        zf.writestr("20170531/hasu_tanashi_20170531_Ins1RGB_30m_dsm.tif", "demo")
        zf.writestr("170531.Lotus.psx", "demo")

    def fake_download(self, url, archive):
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, archive)

    monkeypatch.setattr("easyidp.data.downloader.DatasetDownloader._download_url", fake_download)

    result = lotus.download(force=True, progress=False)

    assert result.downloaded is True
    assert result.extracted is True
    assert lotus.is_ready() is True
```

- [ ] **Step 2: Run failing tests**

Run: `uv run pytest tests/test_data_downloader.py -q`

Expected: fail because downloader behavior is not implemented.

- [ ] **Step 3: Implement downloader skeleton with explicit confirmation**

Create `src/easyidp/data/downloader.py`:

```python
import hashlib
import shutil
from pathlib import Path

from .errors import DatasetDownloadError, DatasetMirrorConfirmationError
from .extract import safe_extract_zip
from .spec import DownloadResult


class DatasetDownloader:
    """Download and extract a dataset through explicit calls.

    Parameters
    ----------
    dataset : easyidp.data.dataset.Dataset
        Dataset object to download.
    mirror : str, default "auto"
        Mirror selector: "auto", "gdrive", or "aliyun".
    confirm : bool, default False
        Required for cost-sensitive mirrors such as Aliyun.
    progress : bool, default True
        Whether backend downloaders may display progress.

    Returns
    -------
    DatasetDownloader
        Downloader bound to one dataset.
    """

    def __init__(self, dataset, mirror="auto", confirm=False, progress=True):
        self.dataset = dataset
        self.mirror = mirror
        self.confirm = confirm
        self.progress = progress

    def download(self, force=False) -> DownloadResult:
        if self.dataset.is_ready() and not force:
            return DownloadResult(
                dataset=self.dataset.name,
                root=str(self.dataset.root),
                archive=str(self.dataset.archive),
                downloaded=False,
                extracted=False,
                ready=True,
            )

        url = self._select_url()
        self._ensure_mirror_allowed(url)
        archive = self.dataset.archive
        part = archive.with_suffix(archive.suffix + ".part")
        extract_tmp = self.dataset.cache_root / ".extracting" / self.dataset.name
        archive.parent.mkdir(parents=True, exist_ok=True)
        extract_tmp.parent.mkdir(parents=True, exist_ok=True)

        self._download_url(url, part)
        part.replace(archive)
        self._validate_checksum(archive)
        if extract_tmp.exists():
            shutil.rmtree(extract_tmp)
        safe_extract_zip(archive, extract_tmp)
        if self.dataset.root.exists():
            shutil.rmtree(self.dataset.root)
        extract_tmp.replace(self.dataset.root)

        return DownloadResult(
            dataset=self.dataset.name,
            root=str(self.dataset.root),
            archive=str(archive),
            downloaded=True,
            extracted=True,
            ready=self.dataset.is_ready(),
        )

    def _select_url(self) -> str:
        urls = self.dataset.spec.urls
        if self.mirror == "auto":
            return urls[0]
        for url in urls:
            if url.startswith(f"{self.mirror}://"):
                return url
        raise DatasetDownloadError(f"Mirror '{self.mirror}' is not available for {self.dataset.name}")

    def _ensure_mirror_allowed(self, url: str) -> None:
        if url.startswith("aliyun://") and not self.confirm:
            raise DatasetMirrorConfirmationError(
                "Aliyun mirror may incur maintainer bandwidth cost. "
                "Call download(mirror='aliyun', confirm=True) to continue."
            )

    def _download_url(self, url: str, archive: Path) -> None:
        if url.startswith("gdrive://"):
            self._download_gdrive(url, archive)
            return
        if url.startswith("aliyun://"):
            self._download_aliyun(url, archive)
            return
        raise DatasetDownloadError(f"Unsupported dataset URL: {url}")

    def _download_gdrive(self, url: str, archive: Path) -> None:
        import gdown

        file_id = url.removeprefix("gdrive://")
        gdown.download(id=file_id, output=str(archive), quiet=not self.progress)

    def _download_aliyun(self, url: str, archive: Path) -> None:
        raise DatasetDownloadError("Aliyun backend is not implemented in this task")

    def _validate_checksum(self, archive: Path) -> None:
        if not self.dataset.spec.checksum:
            return
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != self.dataset.spec.checksum:
            raise DatasetDownloadError(f"Checksum mismatch for {archive}")
```

- [ ] **Step 4: Run downloader tests**

Run: `uv run pytest tests/test_data_downloader.py -q`

Expected: pass for skip-ready, confirmation, and fake backend forced download tests.

- [ ] **Step 5: Commit**

```bash
git add src/easyidp/data/downloader.py src/easyidp/data/dataset.py tests/test_data_downloader.py
git commit -m "feat(data): add explicit dataset downloader"
```

---

### Task 8: Migrate tests away from module-level downloads

**Files:**

- Modify: `tests/__init__.py`
- Modify: `tests/test_data.py`
- Modify: tests that contain module-level `idp.data.TestData()`
- Create or modify: `pytest.ini` or `pyproject.toml` pytest markers if needed

- [ ] **Step 1: Find module-level `TestData()` usage**

Run: `rg "test_data = idp\.data\.TestData\(" tests`

Expected: list files such as `tests/test_cvtools.py`, `tests/test_pix4d.py`, `tests/test_shp.py`, `tests/test_metashape.py`, `tests/test_pointcloud.py`, `tests/test_back2raw_performance.py`, and `tests/test_jsonfile.py`.

- [ ] **Step 2: Add fixture-based explicit data readiness**

Modify `tests/__init__.py`:

```python
@pytest.fixture(scope="session")
def test_data():
    data = idp.data.TestData(notify_missing=False)
    if not data.is_ready():
        pytest.skip("EasyIDP test data is not downloaded. Run `idp.data.TestData().download()` before data-dependent tests.")
    return data
```

Change `shared_data()` to receive the fixture:

```python
@pytest.fixture(scope="module")
def shared_data(test_data):
    roi_all = idp.ROI(test_data.shp.lotus_shp, name_field=0)
    roi_select = idp.ROI()
    for key in ["N1W1", "N1W2", "N2E2", "S1W1"]:
        roi_select[key] = roi_all[key]
        roi_select.crs = roi_all.crs
        roi_select.source = roi_all.source
    return {"test_data": test_data, "roi_all": roi_all, "roi_select": roi_select}
```

- [ ] **Step 3: Replace module-level test data variables**

For each test module with `test_data = idp.data.TestData()`, remove the module-level line and pass `test_data` fixture to tests that need it.

Example replacement in `tests/test_shp.py`:

```python
def test_read_shp(test_data):
    shp_path = test_data.shp.lotus_shp
    assert shp_path.name == "lotus_plots.shp"
    assert shp_path.parent.name == "shp_test"
```

Preserve test behavior; only move data acquisition into fixtures.

- [ ] **Step 4: Update `tests/test_data.py` to new behavior**

Replace tests that expect automatic download with tests that assert no automatic download and explicit dry-run/download behavior.

Example:

```python
def test_lotus_constructor_is_lightweight(tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)

    assert lotus.name == "2017_tanashi_lotus"
    assert not lotus.root.exists()
    assert lotus.dry_run().needs_download is True
```

- [ ] **Step 5: Run data unit tests without downloading**

Run: `uv run pytest tests/test_config.py tests/test_import_side_effects.py tests/test_data_registry.py tests/test_data_dataset.py tests/test_data_downloader.py tests/test_data.py -q`

Expected: pass without network access.

- [ ] **Step 6: Commit**

```bash
git add tests src/easyidp/data
git commit -m "test(data): make dataset fixtures explicit and non-networked"
```

---

### Task 9: Wire logger/banner behavior to `idp.config`

**Files:**

- Modify: `src/easyidp/logger.py`
- Modify: `src/easyidp/__init__.py`
- Test: `tests/test_config.py` or logger tests

- [ ] **Step 1: Add tests for config-controlled logger defaults**

Append to `tests/test_config.py`:

```python
def test_config_can_store_log_level_and_banner(tmp_path):
    config = EasyIDPConfig(config_path=tmp_path / "config.json")
    config.update(log_level="ERROR", show_banner=False)
    config.save()

    loaded = EasyIDPConfig(config_path=tmp_path / "config.json")

    assert loaded.log_level == "ERROR"
    assert loaded.show_banner is False
```

- [ ] **Step 2: Keep logger import safe**

Ensure `logger.py` does not import `easyidp` package root. If it needs config values, import `easyidp.config` directly inside functions to avoid circular import:

```python
def _configured_log_level(default="INFO"):
    try:
        from . import config
    except ImportError:
        return default
    return config.get().log_level
```

- [ ] **Step 3: Gate banner output on config**

Where startup diagnostics/banner are emitted, add:

```python
if not config.get().show_banner:
    return
```

Keep import-time behavior local and fast. Do not add network checks.

- [ ] **Step 4: Run config and logger tests**

Run: `uv run pytest tests/test_config.py tests/test_init_class_func.py -q`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/easyidp/logger.py src/easyidp/__init__.py tests/test_config.py
git commit -m "refactor(config): connect logger defaults to package config"
```

---

### Task 10: Update docs and migration guide

**Files:**

- Modify: `docs/python_api/data.rst`
- Modify: `docs/contribute.rst`
- Add or modify autodoc pages under `docs/python_api/autodoc/`
- Optional: Add `.agents/references/v2.1refactor/data_dataset_architecture.md` details if implementation diverges from current notes.

- [ ] **Step 1: Update data API docs to explicit download style**

Replace old constructor-download example in `docs/python_api/data.rst` with:

```rst
Use example:

.. code-block:: python

    >>> import easyidp as idp
    >>> lotus = idp.data.Lotus()
    >>> lotus.is_ready()
    False
    >>> lotus.dry_run()
    DownloadPlan(dataset='2017_tanashi_lotus', root='/home/user/.local/share/easyidp.data/2017_tanashi_lotus', archive='/home/user/.local/share/easyidp.data/.downloads/2017_tanashi_lotus.zip', ready=False, needs_download=True, size_bytes=3300000000, urls=('gdrive://1SJmp-bG5SZrwdeJL-RnnljM2XmMNMF0j', 'aliyun://easyidp-data/2017_tanashi_lotus.zip'), missing_files=('plots.shp',))
    >>> lotus.download()
    DownloadResult(dataset='2017_tanashi_lotus', root='/home/user/.local/share/easyidp.data/2017_tanashi_lotus', archive='/home/user/.local/share/easyidp.data/.downloads/2017_tanashi_lotus.zip', downloaded=True, extracted=True, ready=True, warnings=())
    >>> lotus.shp
    PosixPath('/home/user/.local/share/easyidp.data/2017_tanashi_lotus/plots.shp')
```

Add a note:

```rst
Dataset constructors do not download data in EasyIDP v2.1. They only create
lightweight path objects and check local readiness. Use ``.download()`` for
explicit downloads.
```

- [ ] **Step 2: Document config data directory**

Add:

```rst
Changing the data directory:

.. code-block:: python

    >>> import easyidp as idp
    >>> idp.config.update(data_dir="D:/EasyIDPData")
    >>> idp.config.save()
```

- [ ] **Step 3: Update contribution data setup**

In `docs/contribute.rst`, replace typo and old hidden download flow:

```rst
Then download the test dataset explicitly:

.. code-block:: bash

    uv run python -c "import easyidp as idp; idp.data.TestData().download()"

Run tests:

.. code-block:: bash

    uv run pytest
```

- [ ] **Step 4: Run doc-related smoke checks**

Run: `uv run python -m compileall src/easyidp`

Expected: compileall exits with code 0.

Run if docs dependencies are installed: `uv run sphinx-build -b html docs docs/_build/html`

Expected: Sphinx build exits with code 0.

- [ ] **Step 5: Commit**

```bash
git add docs src/easyidp
git commit -m "docs(data): document explicit dataset downloads and config"
```

---

### Task 11: Full verification and cleanup

**Files:**

- Review all files touched in previous tasks.
- Update tests or docs only if verification exposes concrete failures.

- [ ] **Step 1: Run targeted data/config tests**

Run: `uv run pytest tests/test_config.py tests/test_import_side_effects.py tests/test_data_registry.py tests/test_data_dataset.py tests/test_data_downloader.py tests/test_data.py -q`

Expected: all selected tests pass, except tests marked to skip because large data is not downloaded.

- [ ] **Step 2: Run full unit suite**

Run: `uv run pytest -q`

Expected: pass, with data-dependent tests skipped only when `TestData` is not downloaded.

- [ ] **Step 3: Run static checks required by project rules**

Run: `uv run ruff check src tests`

Expected: no lint errors.

Run: `uv run mypy src/easyidp`

Expected: no type errors, or record existing unrelated type debt explicitly before fixing in a separate task.

- [ ] **Step 4: Inspect import behavior manually**

Run: `uv run python -c "import easyidp as idp; print(idp.config.get().to_dict()); print(idp.data.list_datasets())"`

Expected: prints config dictionary and dataset names without network access, prompts, or package installation.

- [ ] **Step 5: Inspect git diff**

Run: `git status --short`

Run: `git diff --stat`

Run: `git diff -- src/easyidp tests docs pyproject.toml`

Expected: only intended files changed.

- [ ] **Step 6: Final commit**

```bash
git add src/easyidp tests docs pyproject.toml
git commit -m "refactor(data): complete explicit dataset workflow"
```

---

## Execution Notes

- Do not restore automatic download-on-construction. That is the main breaking change.
- Do not add environment-variable data path overrides in v2.1; use JSON config through `idp.config`.
- Do not add a CLI in this phase; use Python API calls for CI, MCP, skills, and documentation examples.
- Keep user-facing logs, warnings, and errors in English.
- Keep functions focused and under 50 executable lines where practical; split path bundles if they grow large.
- Preserve existing local caches by keeping default data directory and dataset root names unchanged.
- If `TestData` path coverage is incomplete during migration, copy the missing logical path from old `src/easyidp/data.py` into `src/easyidp/data/testing.py` in the same task that exposes the failing test.

## Self-Review Checklist

- Spec coverage: configuration, explicit dataset constructors, missing-cache warnings, registry, downloader, Aliyun confirmation, safe extraction, tests, CI, MCP/skills API, and docs are each covered by tasks.
- Placeholder scan: no task contains open-ended placeholder instructions.
- Type consistency: public names are `EasyIDPConfig`, `DatasetSpec`, `DownloadPlan`, `DownloadResult`, `Dataset`, `Lotus`, `ForestBirds`, `TestData`, `DatasetRegistry`, and `DatasetDownloader` throughout.
- Migration consistency: default root remains `easyidp.data`; old short dataset class names remain public; automatic download is removed.
