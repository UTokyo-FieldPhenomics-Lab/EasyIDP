# Data And Dataset Architecture Notes

## Current Model

- `src/easyidp/data.py` mixes dataset definitions, download logic, extraction, path mapping, URL checks, and test dataset path declarations.
- `EasyidpDataSet.__init__()` calls `load_data()`, so constructing `Lotus`, `ForestBirds`, or `TestData` can trigger network and disk writes.
- `src/easyidp/__init__.py` currently owns `user_data_dir`, `logged_input`, Google availability checks, Aliyun fallback state, and runtime `oss2` installation behavior.
- Tests rely on `TestData()` and `shared_data` fixtures that can download large data and construct heavy ROI/reconstruction objects.

## Problem

- Importing EasyIDP must not perform network requests, install packages, or configure hidden global download state.
- Dataset object construction must not download or extract data by default.
- Interactive prompts block MCP, skills, CI, services, and notebooks running in non-interactive mode.
- `ZipFile.extractall()` without path validation has zip slip risk.
- Test data path mapping, demo dataset metadata, and download mechanics are tightly coupled.

## Recommended Direction

- Split dataset metadata from download execution.
- Use explicit data objects: `DatasetSpec`, `DatasetRegistry`, `DownloadPlan`, `DownloadResult`, and optional path bundles.
- Make all download workflows explicit, non-interactive by default, and safe for automation.
- Move test-data path declarations out of heavy runtime constructors.

## Target Module Shape

```text
easyidp.data
  __init__.py      # public exports only, no network side effects
  spec.py          # DatasetSpec, DatasetFile, DownloadPlan, DownloadResult
  registry.py      # DatasetRegistry and built-in dataset specs
  downloader.py    # Downloader protocol and implementations
  paths.py         # user cache/data directory helpers
  extract.py       # safe archive extraction
  testing.py       # TestDataPaths or test-only path bundles
```

## Core Models

```python
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    version: str
    size_bytes: int | None
    urls: tuple[str, ...]
    checksum: str | None
    files: dict[str, str]
    description: str = ""


@dataclass(frozen=True)
class DownloadPlan:
    dataset: str
    cache_dir: Path
    needs_download: bool
    urls: tuple[str, ...]
    size_bytes: int | None


@dataclass(frozen=True)
class DownloadResult:
    dataset: str
    root: Path
    downloaded: bool
    extracted: bool
    warnings: tuple[str, ...] = ()
```

## Download Policy

- No import-time network checks.
- No runtime package installation.
- No `input()` in library code.
- Provide `dry_run()`, `validate()`, and explicit `download()` flows.
- Use temporary files and atomic rename for archives.
- Validate archive extraction paths before writing files.
- Mirror selection should be explicit or automatic without prompting.

## Test Data Policy

- Tests should use fixtures in `tests/conftest.py` or equivalent, not module-level `TestData()` construction.
- Test path bundles should be pure path maps and should not download during construction.
- CI should run data download/setup explicitly before integration tests.
- Unit tests should not require network access.

## MCP And Skills Requirements

- Public data APIs should return JSON-serializable plans/results.
- Tool-facing calls should support `dry_run`, `validate_only`, and `progress=False`.
- Large files should be returned as paths and metadata, not embedded contents.
- Errors should be structured enough for agents to recover or ask for confirmation.

## Migration Notes

- v2.1 can break automatic download-on-construction behavior.
- Keep thin compatibility helpers only for common demo workflows if they do not hide network or prompt behavior.
- Replace `download_all()` with explicit registry-driven selection.
- Move `user_data_dir` into the data/path layer or a shared path utility.

## References:

Detailed subagent investigation can refer to: `.agents/references/v2.1refactor/subagents/investigate_structures_container.md`
