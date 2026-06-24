# ModelScope Dataset Mirror For Mainland China

## Decision

Use ModelScope, not OpenXLab, as the mainland China dataset mirror for EasyIDP demo datasets.

The current EasyIDP v2.1 data downloader already has a mirror abstraction in `src/easyidp/data/downloader.py`, with manifests under `src/easyidp/data/datasets/*.json`. The existing implementation supports:

- `gdrive`: Google Drive download through `gdown`
- `openxlab`: anonymous OpenXLab CDN resolve download

Replace the `openxlab` mirror path with a `modelscope` mirror path.

## Why Replace OpenXLab

The previous OpenXLab investigation showed that anonymous downloads are possible through an undocumented CDN resolve path, but the official SDK path is not suitable for EasyIDP's public no-login workflow.

OpenXLab issues observed:

- `openxlab.dataset.download()` requires local `openxlab login` even for public datasets.
- The usable anonymous path relies on CDN resolve behavior instead of the official Python download API.
- The SDK/CLI had rough edges in testing, including `openxlab dataset ls` failing with `NameError: name 'rprint' is not defined`.
- Production code must manually resolve metadata, parse CDN URLs, stream files, and verify hashes.

ModelScope result:

- The public EasyIDP demo dataset file downloaded anonymously with the official Python SDK.
- The single-file SDK API is simpler than the OpenXLab anonymous CDN workaround.
- No login or token was needed for the tested public dataset.

## Verified ModelScope SDK Test

Dataset page:

```text
https://modelscope.cn/datasets/HowcanoeWang/EasyIDP-Demo-Dataset/files
```

SDK version tested:

```text
modelscope 1.37.1
```

Single-file API:

```python
from modelscope.hub.file_download import dataset_file_download

path = dataset_file_download(
    dataset_id="HowcanoeWang/EasyIDP-Demo-Dataset",
    file_path="gdown_test.zip",
    local_dir="/tmp/opencode/modelscope_easyidp_demo_single/local_dir",
    cache_dir="/tmp/opencode/modelscope_easyidp_demo_single/cache",
)
```

Verified output:

```text
DOWNLOADED_PATH /tmp/opencode/modelscope_easyidp_demo_single/local_dir/gdown_test.zip
ZIP_EXISTS True
SIZE 280
SHA256 b353aee3743d29d968b09222c5a3b208268cce09100ab73b4048cb7cf42a844c
ZIP_NAMES ['file1.txt', 'folder1/']
ZIP_TEST None
```

Filtered snapshot API also worked:

```python
from modelscope.hub.snapshot_download import snapshot_download

path = snapshot_download(
    repo_id="HowcanoeWang/EasyIDP-Demo-Dataset",
    repo_type="dataset",
    allow_file_pattern="gdown_test.zip",
    local_dir="/tmp/opencode/modelscope_easyidp_demo/local_dir",
    cache_dir="/tmp/opencode/modelscope_easyidp_demo/cache",
)
```

For EasyIDP, prefer `dataset_file_download()` because each dataset manifest points to a single archive file. `snapshot_download()` lists the repository and is unnecessary for this workflow.

## Current EasyIDP Download Shape

Current downloader file:

```text
src/easyidp/data/downloader.py
```

Current dispatch:

```python
if mirror_key == "gdrive":
    _download_gdrive(mirror_config["file_id"], archive, progress)
elif mirror_key == "openxlab":
    _download_openxlab(mirror_config, archive, progress)
else:
    raise ValueError(f"Unknown mirror type: {mirror_key!r}")
```

Current OpenXLab manifest shape:

```json
"mirrors": {
  "gdrive": {
    "file_id": "1yWvIOYJ1ML-UGleh3gT5b7dxXzBuSPgQ"
  },
  "openxlab": {
    "dataset_repo": "HowcanoeWang/easyidp-demo-dataset",
    "source_path": "/gdown_test.zip"
  }
}
```

The current `mirror="auto"` behavior returns the first key in the manifest mirror dict. If `gdrive` remains first, `auto` keeps Google Drive as the default. If `modelscope` is moved before `gdrive`, `auto` will use ModelScope by default.

## Recommended Manifest Shape

Use `modelscope` as the mirror key.

Recommended replacement for `download_smoke.json`:

```json
"mirrors": {
  "gdrive": {
    "file_id": "1yWvIOYJ1ML-UGleh3gT5b7dxXzBuSPgQ"
  },
  "modelscope": {
    "dataset_id": "HowcanoeWang/EasyIDP-Demo-Dataset",
    "file_path": "gdown_test.zip",
    "sha256": "b353aee3743d29d968b09222c5a3b208268cce09100ab73b4048cb7cf42a844c"
  }
}
```

For larger EasyIDP archives, keep the same pattern:

```json
"modelscope": {
  "dataset_id": "HowcanoeWang/EasyIDP-Demo-Dataset",
  "file_path": "2017_tanashi_lotus.zip",
  "sha256": "<archive_sha256>"
}
```

Notes:

- `file_path` should not start with `/`; the ModelScope SDK test used `gdown_test.zip`.
- Store `sha256` in EasyIDP's manifest because ModelScope's simple SDK call returns a local path, not a validated EasyIDP-owned integrity contract.
- Keep `spec.size_bytes` as the human-facing dataset size, but use actual archive byte size or SHA256 for download verification when available.

## Recommended Downloader Implementation

Add `modelscope` dispatch:

```python
if mirror_key == "gdrive":
    _download_gdrive(mirror_config["file_id"], archive, progress)
elif mirror_key == "modelscope":
    _download_modelscope(mirror_config, archive, progress)
else:
    raise ValueError(f"Unknown mirror type: {mirror_key!r}")
```

Recommended helper shape:

```python
def _download_modelscope(mirror_config, archive, progress):
    """Download a dataset archive from ModelScope."""
    try:
        from modelscope.hub.file_download import dataset_file_download
    except ImportError:
        raise RuntimeError(
            "modelscope is required for ModelScope dataset downloads. "
            "Install it with: pip install 'easyidp[data]'."
        )

    archive.parent.mkdir(parents=True, exist_ok=True)
    part = archive.with_suffix(archive.suffix + ".part")

    downloaded = dataset_file_download(
        dataset_id=mirror_config["dataset_id"],
        file_path=mirror_config["file_path"].lstrip("/"),
        local_dir=str(part.parent),
        cache_dir=str(archive.parent / ".modelscope_cache"),
    )

    os.replace(downloaded, part)
    _verify_downloaded_file(part, mirror_config)
    os.replace(part, archive)
```

Implementation notes:

- `progress` is currently passed through the EasyIDP public API, but `dataset_file_download()` controls its own tqdm output. First implementation can ignore `progress`; later check whether ModelScope exposes a quiet flag.
- Use a small `_verify_downloaded_file(path, mirror_config)` helper to check optional `size` and `sha256` fields.
- Keep verification after SDK download. Do not rely only on SDK success.
- Use `.part` and `os.replace()` to avoid leaving a half archive at the final path.
- Delete or leave `.modelscope_cache` based on future cache policy; do not let it become the extracted dataset root.

## Dependency Change

Current optional data dependencies in `pyproject.toml` are:

```toml
[project.optional-dependencies]
data = [
    "gdown>=5.2.0",
    "requests>=2.32.3",
]
```

Recommended change:

```toml
[project.optional-dependencies]
data = [
    "gdown>=5.2.0",
    "modelscope>=1.37.1",
    "requests>=2.32.3",
]
```

If `requests` becomes unused after removing OpenXLab direct HTTP streaming, keep it only if other data utilities still need it.

## Migration Steps

1. Add `modelscope>=1.37.1` to the `data` optional dependency group.
2. Replace manifest `openxlab` entries with `modelscope` entries.
3. Add SHA256 fields for each ModelScope archive when available.
4. Replace `_download_openxlab()` dispatch with `_download_modelscope()`.
5. Remove OpenXLab-only helpers after no manifest references `openxlab`.
6. Update tests that expect `openxlab` mirror names to use `modelscope`.
7. Keep `download_smoke` as the first verification target because it is tiny and already has a verified SHA256.

## Test Commands

Inspect SDK version and functions:

```bash
uv run --with modelscope python - <<'PY'
import inspect
import modelscope
from modelscope.hub.file_download import dataset_file_download
from modelscope.hub.snapshot_download import snapshot_download

print('modelscope', modelscope.__version__)
print('dataset_file_download', inspect.signature(dataset_file_download))
print('snapshot_download', inspect.signature(snapshot_download))
PY
```

Verify the smoke archive:

```bash
uv run --with modelscope python - <<'PY'
from pathlib import Path
import hashlib
import zipfile
from modelscope.hub.file_download import dataset_file_download

path = dataset_file_download(
    dataset_id='HowcanoeWang/EasyIDP-Demo-Dataset',
    file_path='gdown_test.zip',
    local_dir='/tmp/opencode/modelscope_easyidp_demo_single/local_dir',
    cache_dir='/tmp/opencode/modelscope_easyidp_demo_single/cache',
)

zip_path = Path(path)
data = zip_path.read_bytes()
print('DOWNLOADED_PATH', path)
print('SIZE', len(data))
print('SHA256', hashlib.sha256(data).hexdigest())
with zipfile.ZipFile(zip_path) as zf:
    print('ZIP_NAMES', zf.namelist())
    print('ZIP_TEST', zf.testzip())
PY
```

Expected result:

```text
SIZE 280
SHA256 b353aee3743d29d968b09222c5a3b208268cce09100ab73b4048cb7cf42a844c
ZIP_NAMES ['file1.txt', 'folder1/']
ZIP_TEST None
```

## Open Questions Before Code Changes

- Should `mirror="auto"` continue to prefer Google Drive, or should ModelScope be first in each manifest for mainland China users?
- Should EasyIDP keep OpenXLab as a third fallback mirror for one release, or remove it immediately from v2.1 because the architecture prioritizes clarity over backward compatibility?
- Should archive SHA256 become a required manifest field for all non-Google mirrors?

## Current Recommendation

For v2.1, keep the implementation minimal:

- `gdrive` remains available.
- `modelscope` replaces `openxlab` as the China-accessible public mirror.
- `download_smoke` verifies the new path first.
- Manifest-level SHA256 is used for EasyIDP-owned integrity checks.
- OpenXLab code and docs become historical references only after the ModelScope implementation passes smoke tests.
