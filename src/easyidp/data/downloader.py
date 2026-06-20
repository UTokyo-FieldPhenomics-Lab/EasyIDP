"""Explicit dataset downloaders with gdrive and anonymous OpenXLab mirrors."""

import hashlib
import os
import re
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]


def download_dataset(dataset, mirror="auto", force=False, progress=True):
    """Download and extract a dataset.

    Parameters
    ----------
    dataset : Dataset
        Dataset instance with mirrors and archive metadata.
    mirror : str, optional
        Mirror name or ``"auto"`` to pick the first available mirror.
    force : bool, optional
        Re-download even when the dataset is already ready.
    progress : bool, optional
        Show a progress bar during download.

    Returns
    -------
    dict
        JSON-friendly result with keys ``name``, ``root``, ``archive``,
        ``downloaded``, ``extracted``, ``ready``.
    """
    ready = dataset.is_ready()

    if ready and not force:
        return _result(dataset, downloaded=False, extracted=False, ready=True)

    mirror_key = _select_mirror(dataset.mirrors, mirror)
    mirror_config = dataset.mirrors[mirror_key]

    if mirror_key == "gdrive":
        _download_gdrive(mirror_config["file_id"], dataset.archive, progress)
    elif mirror_key == "openxlab":
        _download_openxlab(mirror_config, dataset.archive, progress)
    else:
        raise ValueError(f"Unknown mirror type: {mirror_key!r}")

    safe_extract_zip(dataset.archive, dataset.root)
    dataset.archive.unlink(missing_ok=True)
    return _result(dataset, downloaded=True, extracted=True, ready=dataset.is_ready())


def safe_extract_zip(archive, dest):
    """Extract a zip archive, rejecting path-traversal members.

    Parameters
    ----------
    archive : Path
        Path to the zip file.
    dest : Path
        Directory to extract into.

    Raises
    ------
    RuntimeError
        If any member resolves outside *dest*.
    """
    dest = dest.resolve()
    with zipfile.ZipFile(archive, "r") as zf:
        for member in zf.infolist():
            member_path = (dest / member.filename).resolve()
            if os.path.commonpath([dest, member_path]) != str(dest):
                raise RuntimeError(
                    f"Zip slip rejected: {member.filename!r} resolves outside "
                    f"destination {dest}"
                )
        zf.extractall(dest)


def _select_mirror(mirrors, mirror):
    """Select a mirror key.

    Parameters
    ----------
    mirrors : Mapping
        Available mirrors dict from the manifest.
    mirror : str
        Requested mirror name or ``"auto"``.

    Returns
    -------
    str
        Selected mirror key.

    Raises
    ------
    ValueError
        If *mirror* is ``"auto"`` and no mirrors exist, or if the named
        mirror is not found.
    """
    if mirror == "auto":
        if not mirrors:
            raise ValueError("No mirrors configured for this dataset")
        return next(iter(mirrors))
    if mirror not in mirrors:
        available = ", ".join(sorted(mirrors))
        raise ValueError(
            f"Mirror {mirror!r} not found. Available: {available}"
        )
    return mirror


def _download_gdrive(file_id, archive, progress):
    """Download a file from Google Drive via gdown.

    Parameters
    ----------
    file_id : str
        Google Drive file ID.
    archive : Path
        Target file path for the downloaded archive.
    progress : bool
        Show a progress bar.

    Raises
    ------
    RuntimeError
        If gdown is not installed (with install hint).
    """
    try:
        import gdown  # type: ignore[import-not-found, import-untyped]
    except ImportError:
        raise RuntimeError(
            "gdown is required for Google Drive downloads. "
            "Install it with: pip install 'easyidp[data]'. "
            "or with : uv sync --extras data"
            "For EasyIDP development case, run: "
            "uv sync --all-groups --all-extras"
            "To keep groups of docs and tests dependencies."
        )

    archive.parent.mkdir(parents=True, exist_ok=True)
    part = archive.with_suffix(archive.suffix + ".part")

    gdown.download(id=file_id, output=str(part), quiet=not progress)

    if not part.exists() or part.stat().st_size == 0:
        raise RuntimeError(f"gdown download produced empty file: {part}")

    os.replace(part, archive)


def _download_openxlab(mirror_config, archive, progress):
    """Download a file from OpenXLab anonymously via the v3 API.

    Parameters
    ----------
    mirror_config : Mapping
        Mirror config with ``dataset_repo`` and ``source_path`` keys.
    archive : Path
        Target file path for the downloaded archive.
    progress : bool
        Show a progress bar.

    Raises
    ------
    RuntimeError
        On API error, missing metadata, or download verification failure.
    """
    info = _fetch_openxlab_file_info(mirror_config)

    archive.parent.mkdir(parents=True, exist_ok=True)
    part = archive.with_suffix(archive.suffix + ".part")

    _stream_download(
        url=info["url"],
        output=part,
        expected_size=info["size"],
        expected_sha256=info["sha256"],
        progress=progress,
    )

    os.replace(part, archive)


def _fetch_openxlab_file_info(mirror_config):
    """Resolve CDN download URL and metadata for an OpenXLab file.

    Parameters
    ----------
    mirror_config : Mapping
        Mirror config with ``dataset_repo`` and ``source_path`` keys.

    Returns
    -------
    dict
        Keys ``url``, ``size``, ``sha256``.

    Raises
    ------
    RuntimeError
        If the API response is missing required fields or has a non-zero
        code.
    """
    dataset = mirror_config["dataset_repo"].replace("/", ",")
    source_path = mirror_config["source_path"].lstrip("/")

    api_url = (
        f"https://openxlab.org.cn/datasets/api/v3/datasets/{dataset}/r/main"
    )

    resp = requests.post(
        api_url,
        json={"path": source_path, "preview": False},
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()

    if body.get("code") != 0:
        raise RuntimeError(
            f"OpenXLab API error: code={body.get('code')} "
            f"msg={body.get('msg', '')}"
        )

    data = body.get("data", {})
    url = data.get("url")
    meta = data.get("meta", {})
    size = meta.get("size")

    if not url or size is None:
        raise RuntimeError(
            "Incomplete OpenXLab file metadata: missing URL or size"
        )

    sha256_hex = _extract_sha256_from_cdn_url(url)
    if not sha256_hex:
        raise RuntimeError(
            f"Could not extract SHA256 from CDN URL: {url}"
        )

    return {"url": url, "size": size, "sha256": sha256_hex}


def _extract_sha256_from_cdn_url(url):
    """Extract a 64-hex SHA256 from an OpenXLab CDN objects URL path.

    Parameters
    ----------
    url : str
        Full CDN URL.

    Returns
    -------
    str or None
        Lowercase hex digest if found, else ``None``.
    """
    parsed = urlparse(url)
    match = re.search(r"/objects/([a-f0-9]{64})", parsed.path)
    return match.group(1) if match else None


def _stream_download(url, output, expected_size, expected_sha256, progress):
    """Stream a file from *url* to *output* with verification.

    Parameters
    ----------
    url : str
        Download URL.
    output : Path
        Output file path.
    expected_size : int
        Expected file size in bytes.
    expected_sha256 : str
        Expected SHA256 hex digest.
    progress : bool
        Show a tqdm progress bar.

    Raises
    ------
    RuntimeError
        If the downloaded size or SHA256 does not match.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    sha256 = hashlib.sha256()
    total = 0

    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with (
            tqdm(
                total=expected_size,
                unit="B",
                unit_scale=True,
                disable=not progress,
            ) as pbar,
            output.open("wb") as fh,
        ):
            for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                sha256.update(chunk)
                total += len(chunk)
                pbar.update(len(chunk))

    if total != expected_size:
        raise RuntimeError(
            f"Download size mismatch: got {total} bytes, "
            f"expected {expected_size}"
        )

    actual_hex = sha256.hexdigest()
    if actual_hex != expected_sha256:
        raise RuntimeError(
            f"SHA256 mismatch: got {actual_hex}, "
            f"expected {expected_sha256}"
        )


def _result(dataset, *, downloaded, extracted, ready):
    """Build a JSON-friendly download result dict.

    Parameters
    ----------
    dataset : Dataset
        Dataset instance.
    downloaded : bool
        Whether the archive was freshly downloaded.
    extracted : bool
        Whether the archive was freshly extracted.
    ready : bool
        Whether all required files are now present.

    Returns
    -------
    dict
        Result with ``name``, ``root``, ``archive``, ``downloaded``,
        ``extracted``, ``ready`` as JSON-friendly values.
    """
    return {
        "name": dataset.name,
        "root": str(dataset.root),
        "archive": str(dataset.archive),
        "downloaded": downloaded,
        "extracted": extracted,
        "ready": ready,
    }
