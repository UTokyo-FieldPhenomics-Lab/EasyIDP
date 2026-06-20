# OpenXLab Anonymous CDN Download Notes

## Context

EasyIDP currently uses Google Drive through `gdown` as the first dataset download path and falls back to AliYun OSS for mainland China users. The goal of this investigation was to test whether OpenDataLab/OpenXLab can replace the private AliYun OSS fallback with a public, lower-maintenance download path.

Two OpenXLab/OpenDataLab dataset cases were tested:

- EasyIDP demo dataset: `HowcanoeWang/easyidp-demo-dataset`
- OmniObject3D dataset: `omniobject3d/OmniObject3D-New`

## SDK Package Findings

Use `openxlab`, not `opendatalab`, for new work.

- `opendatalab` is deprecated and its CLI warns that users should migrate to `openxlab`.
- `openxlab==0.1.3` was used in the tests.
- The official Python API is:

```python
from openxlab.dataset import download

download(
    dataset_repo="username/repo_name",
    source_path="/path/to/file",
    target_path="/path/to/local/folder",
)
```

However, this official `download()` API is not suitable for anonymous public downloads in its current form.

## Official Download API Limitation

Calling the official API without `openxlab login` fails before downloading:

```text
Please login openxlab and config ak/sk, try "openxlab login"
```

The root cause is in the SDK implementation:

- `get_dataset_files(..., auth=False)` can list public dataset files anonymously.
- `download_check()` always calls `http_authorization_header()`.
- `get_dataset_download_urls()` also always calls `http_authorization_header()`.
- `http_authorization_header()` calls `get_jwt(auth=True)` and exits if no local login/config exists.

Therefore, `openxlab.dataset.download()` requires local OpenXLab auth even for public datasets.

## Anonymous CDN Resolve Method

Although the SDK requires login, the public dataset resolve endpoint can return a temporary CDN URL anonymously.

The tested flow is:

1. Query dataset metadata with the public info API.
2. Query file metadata with `get_dataset_files(..., auth=False)`.
3. Build the resolve URL:

```text
https://openxlab.org.cn/datasets/resolve/{dataset_id}/main/{file_path_without_leading_slash}
```

4. Request the resolve URL with redirects disabled.
5. Expect HTTP `302` and read the `Location` header.
6. Stream-download from the `Location` URL.
7. Verify byte size and SHA256 against file metadata.

The returned CDN host in tests was:

```text
cdn-xlab-data.openxlab.org.cn
```

## Minimal Prototype

```python
from pathlib import Path
from urllib.parse import quote
import hashlib

import requests
from openxlab.dataset.handler.download_dataset_repository import ContextInfoNoLogin


def download_public_openxlab_file(dataset_repo, source_path, output_path):
    ctx = ContextInfoNoLogin()
    api = ctx.get_client().get_api()
    dataset_name = dataset_repo.replace("/", ",")

    files = api.get_dataset_files(
        dataset_name,
        payload={"prefix": source_path},
        needContent=True,
        auth=False,
    )["list"]
    info = files[0]
    file_name = info["path"].lstrip("/")
    resolve_url = (
        f"{api.host}/datasets/resolve/{info['dataset_id']}/main/"
        f"{quote(file_name, safe='/')}"
    )

    response = requests.get(resolve_url, allow_redirects=False, timeout=60)
    response.raise_for_status() if response.status_code != 302 else None
    download_url = response.headers["Location"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0

    with requests.get(download_url, stream=True, timeout=180) as stream:
        stream.raise_for_status()
        with output_path.open("wb") as file:
            for chunk in stream.iter_content(4 * 1024 * 1024):
                if not chunk:
                    continue
                file.write(chunk)
                digest.update(chunk)
                size += len(chunk)

    if size != info["size"] or digest.hexdigest() != info["sha256"]:
        raise RuntimeError("Downloaded file does not match OpenXLab metadata.")

    return output_path
```

Notes for production code:

- Do not depend on `openxlab.dataset.handler...` internals if avoidable; they are not stable public APIs.
- A small direct `requests` client against the OpenXLab HTTP APIs may be easier to maintain than importing SDK internals.
- Keep SHA256 verification mandatory for downloaded archives.
- The resolved CDN URL includes temporary signed query parameters, so it should be generated at download time rather than persisted.

## Verified EasyIDP Demo Dataset

Dataset repo:

```text
HowcanoeWang/easyidp-demo-dataset
```

File:

```text
/gdown_test.zip
```

Result:

```text
SIZE 280
SHA256 b353aee3743d29d968b09222c5a3b208268cce09100ab73b4048cb7cf42a844c
ZIP NAMES ['file1.txt', 'folder1/']
ZIP TEST None
```

Conclusion: anonymous CDN resolve download works for the EasyIDP test archive.

## Verified OmniObject3D Files

Dataset repo:

```text
omniobject3d/OmniObject3D-New
```

Source directory:

```text
/raw/blender_renders
```

### battery.tar.gz

```text
PATH /raw/blender_renders/battery.tar.gz
SIZE 68,863,951 bytes
SHA256 6c136adaf456b2ef1a0ce2e8afa22235854b28327f31146fb5d5178726c55d71
MATCH True
```

### anise.tar.gz

```text
PATH /raw/blender_renders/anise.tar.gz
SIZE 1,100,170,958 bytes
SHA256 ad66724c4fb5b780c86efac0565204fe25c526964bfe91ddee6c417c0a752ee3
MATCH True
ELAPSED_SECONDS 46.5
```

Conclusion: the anonymous CDN resolve method works for both small archives and 1 GB-level archives.

## Implications For EasyIDP

Recommended replacement direction:

1. Keep Google Drive or other sources only if they are still useful as mirrors.
2. Replace AliYun OSS fallback with an OpenXLab public downloader if the EasyIDP datasets are hosted there publicly.
3. Avoid requiring `openxlab login` for normal users; use anonymous resolve instead.
4. Store each dataset's OpenXLab repo and source archive path as metadata.
5. Verify downloaded files with SHA256 before unzip.
6. Keep clear error messages for three distinct failures:
   - dataset/file metadata not found
   - CDN resolve does not return `302`
   - downloaded file fails size or SHA256 validation

Potential risks:

- The anonymous resolve endpoint is not documented as the primary Python SDK path, so API compatibility is less guaranteed than the official `download()` function.
- The `openxlab` SDK currently contains CLI issues, such as `dataset ls` failing with `NameError: name 'rprint' is not defined` in one tested path.
- Public datasets may still require application forms or login if the server returns special access statuses for particular datasets.
- Very large files should support resume or retry in production code.

## Useful Test Commands

Check `openxlab` availability:

```bash
uv run --with openxlab openxlab version
```

Official API test, expected to fail without login:

```bash
uv run --with openxlab python - <<'PY'
from openxlab.dataset import download

download(
    dataset_repo="HowcanoeWang/easyidp-demo-dataset",
    source_path="/gdown_test.zip",
    target_path="/tmp/opencode/openxlab-test",
)
PY
```

Expected failure text:

```text
Please login openxlab and config ak/sk, try "openxlab login"
```
