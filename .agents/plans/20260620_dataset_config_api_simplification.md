# Dataset And Config API Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify EasyIDP's user-facing dataset and configuration APIs so dataset objects print useful status and configuration changes persist immediately.

**Architecture:** `easyidp.config` becomes a small persistent key-value API with `get`, `set`, and `reset`; `set` and `reset` always write JSON atomically. `easyidp.data.Dataset` keeps path-shortcut behavior but hides manifest/download internals behind private fields and user-friendly `repr()` output.

**Tech Stack:** Python 3.10+, `pathlib`, JSON config files, pytest, uv.

---

## File Structure

- Modify `src/easyidp/config.py`: replace public `update/save` workflow with `get/set/reset`, reload JSON on `get`, and write atomically on `set/reset`.
- Modify `src/easyidp/data/dataset.py`: remove public dataset metadata properties, switch `required` to internal `ready_check`, add compact `__repr__`, and use `idp.config.get("data_dir")`.
- Modify `src/easyidp/data/downloader.py`: stop depending on removed public dataset properties such as `mirrors` and `archive`.
- Modify JSON manifests in `src/easyidp/data/datasets/*.json`: remove `spec.title` and rename top-level `required` to `ready_check`.
- Modify `src/easyidp/logger.py`: read config through the new `get(key)` API.
- Modify tests in `tests/test_config.py`, `tests/test_data.py`, and `tests/test_metashape.py`: update expected config behavior, dataset repr, and removed aliases.
- Modify docs in `docs/python_api/data.rst` and generated autodoc stubs under `docs/python_api/autodoc/`: document only the reduced public API.

## Public API Decisions

- `idp.config.get("data_dir")` returns the current persisted value, reloading JSON first.
- `idp.config.get()` returns a plain dict snapshot with `data_dir`, `log_level`, and `show_banner`.
- `idp.config.set(data_dir="/path/to/easyidp.data")` updates memory and writes JSON immediately.
- `idp.config.reset()` restores defaults and writes JSON immediately.
- `idp.config.update()` and `idp.config.save()` are removed from public documentation and package exports in v2.1.
- `data_dir` changes do not migrate old data. They only affect dataset objects created after the change.
- Manual JSON edits are picked up on the next `idp.config.get(...)` call. No config lock is added; EasyIDP writes atomically and concurrent writers use last-write-wins behavior.
- `Dataset` public attributes are reduced to `name`, `root`, dynamic path shortcuts, `path()`, `is_ready()`, `dry_run()`, and `download()`.
- `Dataset` no longer exposes `title`, `description`, `size_bytes`, `mirrors`, `required`, `files`, `cache_root`, `archive`, `zip_file`, or `data_dir` as public API.

---

### Task 1: Persisted Config API

**Files:**
- Modify: `src/easyidp/config.py`
- Modify: `src/easyidp/logger.py`
- Test: `tests/test_config.py`
- Test: `tests/test_init_class_func.py`

- [ ] **Step 1: Replace config tests with get/set/reset semantics**

Update `tests/test_config.py` to assert immediate persistence and JSON reload behavior:

```python
import json

import pytest

import easyidp as idp
from easyidp.config import EasyIDPConfig


def test_default_config_uses_default_data_dir(tmp_path):
    config = EasyIDPConfig(config_path=tmp_path / "config.json")

    assert config.get("data_dir").name == "easyidp.data"
    assert config.get("log_level") == "INFO"
    assert config.get("show_banner") is True


def test_set_writes_json_immediately(tmp_path):
    config_path = tmp_path / "config.json"
    config = EasyIDPConfig(config_path=config_path)

    config.set(data_dir=tmp_path / "data", log_level="DEBUG", show_banner=False)

    assert config_path.exists()
    loaded = EasyIDPConfig(config_path=config_path)
    assert loaded.get("data_dir") == tmp_path / "data"
    assert loaded.get("log_level") == "DEBUG"
    assert loaded.get("show_banner") is False


def test_get_reloads_manual_json_edits(tmp_path):
    config_path = tmp_path / "config.json"
    config = EasyIDPConfig(config_path=config_path)
    config.set(data_dir=tmp_path / "old")

    config_path.write_text(
        json.dumps({
            "data_dir": str(tmp_path / "manual"),
            "log_level": "WARNING",
            "show_banner": False,
        }),
        encoding="utf-8",
    )

    assert config.get("data_dir") == tmp_path / "manual"
    assert config.get("log_level") == "WARNING"
    assert config.get("show_banner") is False


def test_get_without_key_returns_plain_snapshot(tmp_path):
    config = EasyIDPConfig(config_path=tmp_path / "config.json")
    config.set(data_dir=tmp_path / "data", show_banner=False)

    snapshot = config.get()

    assert snapshot == {
        "data_dir": str(tmp_path / "data"),
        "log_level": "INFO",
        "show_banner": False,
    }


def test_package_exports_small_config_entrypoint():
    assert hasattr(idp, "config")
    assert hasattr(idp.config, "get")
    assert hasattr(idp.config, "set")
    assert hasattr(idp.config, "reset")
    assert not hasattr(idp.config, "save")


def test_set_unknown_key_raises_keyerror(tmp_path):
    config = EasyIDPConfig(config_path=tmp_path / "config.json")

    with pytest.raises(KeyError, match="Unknown EasyIDP config key: bad_key"):
        config.set(bad_key="anything")


def test_reset_writes_defaults_immediately(tmp_path):
    config_path = tmp_path / "config.json"
    config = EasyIDPConfig(config_path=config_path)
    config.set(data_dir=tmp_path / "data", log_level="DEBUG", show_banner=False)

    config.reset()

    loaded = EasyIDPConfig(config_path=config_path)
    assert loaded.get("data_dir").name == "easyidp.data"
    assert loaded.get("log_level") == "INFO"
    assert loaded.get("show_banner") is True
```

- [ ] **Step 2: Run config tests and verify failure**

Run:

```bash
uv run pytest tests/test_config.py -v
```

Expected before implementation: failures for missing `set`, old `get()` return type, and old `save` export.

- [ ] **Step 3: Implement `get/set/reset` with atomic writes**

In `src/easyidp/config.py`, replace the old public methods with:

```python
def get(self, key: str | None = None) -> Any:
    """Return current configuration values after reloading JSON.

    Parameters
    ----------
    key : str, optional
        Configuration key to read. When omitted, return a plain dict snapshot.

    Returns
    -------
    Any
        The requested value or a JSON-serializable configuration snapshot.

    Examples
    --------
    >>> cfg = EasyIDPConfig()
    >>> cfg.get("log_level")
    'INFO'
    """
    self._load_if_exists()
    if key is None:
        return self.to_dict()
    if key not in {"data_dir", "log_level", "show_banner"}:
        raise KeyError(f"Unknown EasyIDP config key: {key}")
    return getattr(self, key)


def set(self, **kwargs: Any) -> "EasyIDPConfig":
    """Update configuration and persist it immediately.

    Parameters
    ----------
    **kwargs : Any
        Supported keys are ``data_dir``, ``log_level``, and ``show_banner``.

    Returns
    -------
    EasyIDPConfig
        This configuration object.

    Examples
    --------
    >>> cfg = EasyIDPConfig()
    >>> cfg.set(log_level="DEBUG")
    EasyIDPConfig(...)
    """
    self._apply(kwargs)
    self._save()
    return self


def reset(self) -> "EasyIDPConfig":
    """Restore defaults and persist them immediately.

    Returns
    -------
    EasyIDPConfig
        This configuration object.

    Examples
    --------
    >>> cfg = EasyIDPConfig()
    >>> cfg.reset().get("log_level")
    'INFO'
    """
    self.data_dir = default_data_dir()
    self.log_level = "INFO"
    self.show_banner = True
    self._save()
    return self
```

Add private helpers:

```python
def _apply(self, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if key == "data_dir":
            self.data_dir = Path(value).expanduser()
            continue
        if key == "log_level":
            self.log_level = str(value)
            continue
        if key == "show_banner":
            self.show_banner = bool(value)
            continue
        raise KeyError(f"Unknown EasyIDP config key: {key}")


def _save(self) -> Path:
    self.config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = self.config_path.with_suffix(self.config_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
    tmp_path.replace(self.config_path)
    return self.config_path.resolve()
```

Change module exports at the bottom of `src/easyidp/config.py` to:

```python
config = EasyIDPConfig()
get = config.get
set = config.set
reset = config.reset
```

- [ ] **Step 4: Update logger config reads**

In `src/easyidp/logger.py`, replace object-style config reads with key reads:

```python
log_level = _cfg.get("log_level") if cfg is not None else "INFO"
show_banner = _cfg.get("show_banner") if cfg is not None else True
```

If the existing function already passes a local `cfg` object, keep the branch readable by loading values once at function start.

- [ ] **Step 5: Update tests that call `idp.config.update`**

In `tests/test_init_class_func.py`, replace:

```python
idp.config.update(log_level="ERROR", show_banner=False)
```

with:

```python
idp.config.set(log_level="ERROR", show_banner=False)
```

Replace any reset helper calls with `idp.config.reset()`.

- [ ] **Step 6: Run config tests**

Run:

```bash
uv run pytest tests/test_config.py tests/test_init_class_func.py -v
```

Expected: all selected tests pass.

---

### Task 2: Dataset Manifest And Public API Simplification

**Files:**
- Modify: `src/easyidp/data/dataset.py`
- Modify: `src/easyidp/data/downloader.py`
- Modify: `src/easyidp/data/datasets/forestbirds.json`
- Modify: `src/easyidp/data/datasets/lotus.json`
- Modify: `src/easyidp/data/datasets/testdata.json`
- Test: `tests/test_data.py`
- Test: `tests/test_metashape.py`

- [ ] **Step 1: Update dataset tests for the reduced API and repr output**

In `tests/test_data.py`, remove assertions for `title`, `archive`, and `required` as public attributes. Add tests for repr output:

```python
def test_dataset_repr_shows_missing_status(tmp_path):
    birds = idp.data.ForestBirds(cache_root=tmp_path, notify_missing=False)

    text = repr(birds)

    assert object.__repr__(birds) in text
    assert "Official EasyIDP forest birds demo dataset from Florida." in text
    assert "Size: 1.97 GB" in text
    assert "Status: not downloaded. call .download() to save at" in text
    assert str(tmp_path / "2022_florida_forestbirds") in text
    assert "idp.config.set(data_dir=\"/path/to/easyidp.data\")" in text


def test_dataset_repr_shows_available_status(tmp_path):
    birds = idp.data.ForestBirds(cache_root=tmp_path, notify_missing=False)
    for key in ("shp", "metashape.project", "metashape.dom", "metashape.dsm"):
        path = birds.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    text = repr(birds)

    assert object.__repr__(birds) in text
    assert "Size: 1.97 GB" in text
    assert "Status: available at" in text
    assert str(birds.root) in text
    assert "not downloaded" not in text
```

Update ready checks to use explicit file keys:

```python
for key in ("shp", "metashape.project", "pix4d.dom", "pix4d.dsm"):
    path = lotus.path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
```

Update archive tests to use an internal helper after it is added:

```python
assert lotus._archive_path() == tmp_path / ".downloads" / "2017_tanashi_lotus.zip"
```

- [ ] **Step 2: Run dataset tests and verify failure**

Run:

```bash
uv run pytest tests/test_data.py -v
```

Expected before implementation: failures for old JSON field names, missing repr output, and old public attributes.

- [ ] **Step 3: Rename JSON readiness fields and remove titles**

In each dataset manifest under `src/easyidp/data/datasets/`:

- Delete `spec.title`.
- Rename top-level `required` to `ready_check`.
- Keep `spec.name`, `spec.folder`, `spec.archive`, `spec.size_bytes`, `spec.mirrors`, and `spec.description`.

Example target shape:

```json
{
  "spec": {
    "name": "forestbirds",
    "folder": "2022_florida_forestbirds",
    "archive": "2022_florida_forestbirds.zip",
    "size_bytes": 1970000000,
    "mirrors": {
      "gdrive": {
        "file_id": "1mXkzaoSSCAA87cxcMHKL6_VNlykRYxJr"
      },
      "openxlab": {
        "dataset_repo": "HowcanoeWang/easyidp-demo-dataset",
        "source_path": "/2022_florida_forestbirds.zip"
      }
    },
    "description": "Official EasyIDP forest birds demo dataset from Florida."
  },
  "ready_check": ["shp", "metashape.project", "metashape.dom", "metashape.dsm"],
  "files": {
    "photo": "Hidden_Little_03_24_2022",
    "shp": "Hidden_Little_grid.shp",
    "metashape.project": "Hidden_Little_03_24_2022.psx",
    "metashape.param": "Hidden_Little_03_24_2022.files",
    "metashape.dom": "Hidden_Little_03_24_2022.tiff",
    "metashape.dsm": "Hidden_Little_03_24_2022_DEM.tif"
  }
}
```

- [ ] **Step 4: Implement reduced Dataset internals**

In `src/easyidp/data/dataset.py`, update validation:

```python
for field in ("name", "folder", "size_bytes"):
    if field not in spec:
        raise ValueError(f"manifest spec missing required field: {field!r}")

ready_check = data.get("ready_check", [])
if not isinstance(ready_check, list):
    raise ValueError("manifest 'ready_check' must be a list")
for key in ready_check:
    if key not in files:
        raise ValueError(f"ready_check key {key!r} not found in files")
```

In `Dataset.__init__`, remove `_title` and replace `_required` with `_ready_check`:

```python
self._name = spec["name"]
self._description = spec.get("description", "")
self._size_bytes = spec["size_bytes"]
self._mirrors = types.MappingProxyType(spec.get("mirrors", {}))
self._folder = spec["folder"]
self._archive_name = spec.get("archive", f"{self._folder}.zip")
self._ready_check = tuple(data.get("ready_check", ()))
self._files = types.MappingProxyType(data.get("files", {}))
```

Use the new config API:

```python
if cache_root is None:
    cache_root = _cfg.get("data_dir")
```

Remove the constructor warning block that logs missing datasets.

- [ ] **Step 5: Keep only minimal public Dataset properties**

Keep:

```python
@property
def name(self):
    """Dataset manifest name."""
    return self._name


@property
def root(self):
    """Extracted dataset directory."""
    return self._ns_root
```

Remove public properties for `title`, `description`, `size_bytes`, `mirrors`, `required`, `files`, `cache_root`, `data_dir`, `archive`, and `zip_file`.

Add private archive helper:

```python
def _archive_path(self):
    """Return the temporary archive path for downloader internals.

    Returns
    -------
    pathlib.Path
        Temporary archive path under the cache root.

    Examples
    --------
    >>> ds = Dataset("lotus")
    >>> ds._archive_path().name
    '2017_tanashi_lotus.zip'
    """
    return self._cache_root / ".downloads" / self._archive_name
```

- [ ] **Step 6: Add size formatting and Dataset repr**

Add helper:

```python
def _format_size(size_bytes):
    """Format bytes as a compact decimal size string.

    Parameters
    ----------
    size_bytes : int
        Size in bytes.

    Returns
    -------
    str
        Human-readable size such as ``"1.97 GB"``.

    Examples
    --------
    >>> _format_size(1970000000)
    '1.97 GB'
    """
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1000
```

Add `Dataset.__repr__`:

```python
def __repr__(self):
    lines = [object.__repr__(self)]
    if self._description:
        lines.append(self._description)
    lines.append(f"Size: {_format_size(self._size_bytes)}")

    if self.is_ready():
        lines.extend([
            "Status: available at",
            f"    {self.root}",
        ])
        return "\n".join(lines)

    lines.extend([
        "Status: not downloaded. call .download() to save at",
        f"    {self.root}",
        "    You can change the EasyIDP data cache path with",
        '    idp.config.set(data_dir="/path/to/easyidp.data")',
    ])
    return "\n".join(lines)
```

- [ ] **Step 7: Update readiness and dry-run internals**

Change `is_ready()`:

```python
check_keys = self._ready_check if self._ready_check else self._files
return all(self.path(k).exists() for k in check_keys)
```

Change `dry_run()`:

```python
ready = self.is_ready()
check_keys = self._ready_check if self._ready_check else list(self._files)
missing = [
    self._files[k] for k in check_keys if not self.path(k).exists()
]
return {
    "name": self._name,
    "description": self._description,
    "root": str(self._ns_root),
    "ready": ready,
    "needs_download": not ready,
    "size_bytes": self._size_bytes,
    "missing": missing,
}
```

- [ ] **Step 8: Update downloader internals**

In `src/easyidp/data/downloader.py`, replace public attribute usage:

```python
mirror_key = _select_mirror(dataset._mirrors, mirror)
mirror_config = dataset._mirrors[mirror_key]
archive = dataset._archive_path()
```

Then use `archive` for download, extraction, unlink, and `_result`:

```python
safe_extract_zip(archive, dataset.root)
archive.unlink(missing_ok=True)
```

Update `_result()` to call `dataset._archive_path()` if it still reports archive details.

- [ ] **Step 9: Update old dataset aliases in tests**

In `tests/test_metashape.py`, replace `test_data.data_dir` with `test_data.root`.

Run:

```bash
uv run pytest tests/test_data.py tests/test_metashape.py -v
```

Expected: selected tests pass or data-dependent tests skip if local test data is absent.

---

### Task 3: Documentation And Public Surface Cleanup

**Files:**
- Modify: `docs/python_api/data.rst`
- Modify: `docs/python_api/autodoc/easyidp.data.Lotus.rst`
- Modify: `docs/python_api/autodoc/easyidp.data.ForestBirds.rst`
- Modify: `docs/python_api/autodoc/easyidp.data.TestData.rst`
- Modify: `docs/python_api/advanced.rst`
- Modify: doctest examples in `src/easyidp/**/*.py` that mention `idp.config.update`

- [ ] **Step 1: Update data API docs for the new repr and config API**

In `docs/python_api/data.rst`, replace config examples:

```python
idp.config.set(data_dir="/path/to/easyidp.data")
lotus = idp.data.Lotus()
```

Add a short repr example:

```text
<easyidp.data.dataset.ForestBirds object at 0x...>
Official EasyIDP forest birds demo dataset from Florida.
Size: 1.97 GB
Status: not downloaded. call .download() to save at
    /home/user/.local/share/easyidp.data/2022_florida_forestbirds
    You can change the EasyIDP data cache path with
    idp.config.set(data_dir="/path/to/easyidp.data")
```

- [ ] **Step 2: Update autodoc stubs**

For each dataset autodoc file, keep methods:

```rst
~Lotus.download
~Lotus.dry_run
~Lotus.is_ready
~Lotus.path
```

Keep attributes:

```rst
~Lotus.name
~Lotus.root
```

Remove attributes for `archive`, `cache_root`, `data_dir`, `description`, `files`, `mirrors`, `required`, `size_bytes`, `title`, and `zip_file`.

- [ ] **Step 3: Search and replace stale config calls**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path
for path in Path('.').rglob('*.py'):
    if '.venv' in path.parts:
        continue
    text = path.read_text(encoding='utf-8')
    if 'config.update' in text or 'config.save' in text:
        print(path)
PY
```

Update remaining examples to `idp.config.set(...)` or `idp.config.reset()`.

- [ ] **Step 4: Run focused tests and docs smoke checks**

Run:

```bash
uv run pytest tests/test_config.py tests/test_data.py -v
uv run pytest tests/test_metashape.py -v
```

Expected: config and data tests pass; metashape tests pass or skip data-dependent cases when test data is not downloaded.

- [ ] **Step 5: Run broad verification**

Run:

```bash
uv run pytest
```

Expected: test suite passes or data-dependent tests skip with the documented message when local demo data is absent.

---

## Self-Review

- Spec coverage: the plan covers config API simplification, immediate persistence, manual JSON reload behavior, no config lock, no automatic data migration, dataset JSON field cleanup, reduced dataset public API, and repr output.
- Placeholder scan: no `TBD`, unresolved `TODO`, or ambiguous implementation steps are intentionally left.
- Type consistency: config uses `get(key: str | None)`, `set(**kwargs)`, and `reset()` throughout; dataset readiness uses `ready_check` in JSON and `_ready_check` internally.
