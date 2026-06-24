"""Explicit dataset downloaders with gdrive and ModelScope mirrors."""

import hashlib
import os
import zipfile


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

    mirror_key = _select_mirror(dataset._mirrors, mirror)
    mirror_config = dataset._mirrors[mirror_key]
    archive = dataset._archive_path()

    if mirror_key == "gdrive":
        _download_gdrive(mirror_config["file_id"], archive, progress)
    elif mirror_key == "modelscope":
        _download_modelscope(mirror_config, archive, progress)
    else:
        raise ValueError(f"Unknown mirror type: {mirror_key!r}")

    safe_extract_zip(archive, dataset.root)
    archive.unlink(missing_ok=True)
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


def _download_modelscope(mirror_config, archive, progress):
    """Download a dataset archive from ModelScope.

    Parameters
    ----------
    mirror_config : Mapping
        Mirror config with ``dataset_repo`` and ``file_path`` keys.
    archive : Path
        Target file path for the downloaded archive.
    progress : bool
        Print EasyIDP-level source and target paths. ModelScope controls its
        own progress output internally.

    Examples
    --------
    >>> cfg = {"dataset_repo": "owner/repo", "file_path": "archive.zip"}
    >>> _download_modelscope(cfg, Path("archive.zip"), progress=False)  # doctest: +SKIP

    Raises
    ------
    RuntimeError
        If ModelScope is missing or the downloaded file fails verification.
    """
    try:
        from modelscope.hub.file_download import dataset_file_download  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "modelscope is required for ModelScope dataset downloads. "
            "Install it with: pip install 'easyidp[data]'. "
            "For EasyIDP development, run: uv sync --all-groups --all-extras"
        ) from exc

    archive.parent.mkdir(parents=True, exist_ok=True)
    part = archive.with_suffix(archive.suffix + ".part")
    local_dir = archive.parent / ".modelscope_download"
    cache_dir = archive.parent / ".modelscope_cache"
    file_path = mirror_config["file_path"].lstrip("/")

    if progress:
        original_url = f"https://modelscope.cn/datasets/{mirror_config['dataset_repo']}/files"
        print("Downloading...")
        print(f"From (dataset): {original_url}")
        print(f"From (file): {file_path}")
        print(f"To:  {part}")

    downloaded = dataset_file_download(
        dataset_id=mirror_config["dataset_repo"],
        file_path=file_path,
        local_dir=str(local_dir),
        cache_dir=str(cache_dir),
    )

    os.replace(downloaded, part)
    info = _fetch_modelscope_file_info(mirror_config)
    _verify_downloaded_file(part, info)
    os.replace(part, archive)


def _fetch_modelscope_file_info(mirror_config):
    """Fetch ModelScope file size and SHA256 metadata.

    Parameters
    ----------
    mirror_config : Mapping
        Mirror config with ``dataset_repo`` and ``file_path`` keys.

    Returns
    -------
    dict
        Download verification metadata with ``size_bytes`` and ``sha256``.

    Raises
    ------
    RuntimeError
        If the target file is absent from ModelScope metadata.

    Examples
    --------
    >>> cfg = {"dataset_repo": "owner/repo", "file_path": "archive.zip"}
    >>> _fetch_modelscope_file_info(cfg)  # doctest: +SKIP
    {'size_bytes': 1024, 'sha256': '...'}
    """
    from modelscope.hub.api import HubApi  # type: ignore[import-untyped]

    file_path = mirror_config["file_path"].lstrip("/")
    api = HubApi()
    files = api.get_dataset_files(
        repo_id=mirror_config["dataset_repo"],
        recursive=True,
        page_size=100,
    )

    for item in files:
        if item.get("Path") != file_path:
            continue
        return {"size_bytes": item.get("Size"), "sha256": item.get("Sha256")}

    raise RuntimeError(f"ModelScope file metadata not found: {file_path}")


def _verify_downloaded_file(path, mirror_config):
    """Verify downloaded archive size and SHA256 metadata.

    Parameters
    ----------
    path : Path
        Downloaded file path.
    mirror_config : Mapping
        Mirror config with optional ``size_bytes`` and ``sha256`` values.

    Raises
    ------
    RuntimeError
        If the downloaded file size or SHA256 does not match the manifest.

    Examples
    --------
    >>> _verify_downloaded_file(Path("archive.zip"), {})  # doctest: +SKIP
    """
    expected_size = mirror_config.get("size_bytes")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise RuntimeError(
            f"Download size mismatch: got {path.stat().st_size} bytes, "
            f"expected {expected_size}"
        )

    expected_sha256 = mirror_config.get("sha256")
    if expected_sha256 is None:
        return

    actual_hex = _file_sha256(path)
    if actual_hex != expected_sha256.lower():
        raise RuntimeError(
            f"SHA256 mismatch: got {actual_hex}, "
            f"expected {expected_sha256.lower()}"
        )


def _file_sha256(path):
    """Return the SHA256 hex digest for a file.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Lowercase SHA256 hex digest.

    Examples
    --------
    >>> _file_sha256(Path("archive.zip"))  # doctest: +SKIP
    '...'
    """
    sha256 = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


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
        "archive": str(dataset._archive_path()),
        "downloaded": downloaded,
        "extracted": extracted,
        "ready": ready,
    }
