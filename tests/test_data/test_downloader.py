import easyidp as idp


def test_download_skips_ready_dataset(tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)
    for key in lotus._ready_check:
        p = lotus.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()

    result = lotus.download()

    assert result["name"] == "lotus"
    assert result["downloaded"] is False
    assert result["extracted"] is False
    assert result["ready"] is True


def test_safe_extract_rejects_zip_slip(tmp_path):
    import zipfile as zf

    from easyidp.data.downloader import safe_extract_zip

    bad_zip = tmp_path / "bad.zip"
    with zf.ZipFile(bad_zip, "w") as z:
        z.writestr(zf.ZipInfo("../escape.txt"), "malicious")

    dest = tmp_path / "dest"
    dest.mkdir()

    try:
        safe_extract_zip(bad_zip, dest)
    except RuntimeError as exc:
        assert "escape.txt" in str(exc)
    else:
        assert False, "expected RuntimeError for zip slip"


def test_download_extracts_mocked_gdrive_archive(tmp_path):
    import sys
    import zipfile as zf
    from unittest import mock

    from easyidp.data.downloader import _download_gdrive

    archive = tmp_path / ".downloads" / "test.zip"
    dest = tmp_path / "extracted"
    dest.mkdir()

    test_zip = tmp_path / "real.zip"
    with zf.ZipFile(test_zip, "w") as z:
        z.writestr("file1.txt", "hello")

    def _fake_download(*, id, output, quiet):
        import shutil
        shutil.copy(test_zip, output)

    mock_gdown = mock.MagicMock()
    mock_gdown.download = _fake_download
    sys.modules["gdown"] = mock_gdown

    try:
        _download_gdrive("fake-id", archive, progress=False)
    finally:
        sys.modules.pop("gdown", None)

    assert archive.exists()
    assert archive.stat().st_size > 0


def test_gdrive_missing_dependency_mentions_data_extra(tmp_path, monkeypatch):
    import builtins

    from easyidp.data.downloader import _download_gdrive

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gdown":
            raise ImportError("missing gdown")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        _download_gdrive("fake-id", tmp_path / "test.zip", progress=False)
    except RuntimeError as exc:
        message = str(exc)
    else:
        assert False, "expected RuntimeError for missing gdown"

    assert "pip install 'easyidp[data]'" in message
    assert "uv sync --all-groups --all-extras" in message
    assert "uv add" not in message
    assert "easyidp[gdrive]" not in message


def test_download_removes_archive_after_extracting(tmp_path, monkeypatch):
    import zipfile as zf

    import easyidp.data.downloader as downloader

    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)

    def fake_download_modelscope(mirror_config, archive, progress):
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zf.ZipFile(archive, "w") as z:
            for key in ("shp", "metashape.project", "pix4d.dom", "pix4d.dsm"):
                z.writestr(str(lotus.path(key).relative_to(lotus.root)), "data")

    monkeypatch.setattr(
        downloader,
        "_download_modelscope",
        fake_download_modelscope,
    )

    result = lotus.download(mirror="modelscope", progress=False)

    assert result["downloaded"] is True
    assert result["extracted"] is True
    assert result["ready"] is True
    assert not lotus._archive_path().exists()


def test_modelscope_downloader_uses_sdk_and_verifies_archive(tmp_path, monkeypatch):
    import hashlib
    import sys
    import types

    from easyidp.data.downloader import _download_modelscope

    content = b"x" * 280
    archive = tmp_path / ".downloads" / "gdown_test.zip"
    calls = []

    def fake_dataset_file_download(**kwargs):
        calls.append(kwargs)
        output = tmp_path / "sdk" / kwargs["file_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
        return str(output)

    class FakeHubApi:
        def get_dataset_files(self, **kwargs):
            calls.append(kwargs)
            return [{
                "Path": "gdown_test.zip",
                "Size": len(content),
                "Sha256": hashlib.sha256(content).hexdigest(),
            }]

    download_module = types.SimpleNamespace(dataset_file_download=fake_dataset_file_download)
    api_module = types.SimpleNamespace(HubApi=FakeHubApi)
    monkeypatch.setitem(sys.modules, "modelscope", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "modelscope.hub", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "modelscope.hub.file_download", download_module)
    monkeypatch.setitem(sys.modules, "modelscope.hub.api", api_module)

    _download_modelscope(
        {
            "dataset_repo": "HowcanoeWang/EasyIDP-Demo-Dataset",
            "file_path": "gdown_test.zip",
        },
        archive,
        progress=False,
    )

    assert archive.read_bytes() == content
    assert calls == [{
        "dataset_id": "HowcanoeWang/EasyIDP-Demo-Dataset",
        "file_path": "gdown_test.zip",
        "local_dir": str(archive.parent / ".modelscope_download"),
        "cache_dir": str(archive.parent / ".modelscope_cache"),
    }, {
        "repo_id": "HowcanoeWang/EasyIDP-Demo-Dataset",
        "recursive": True,
        "page_size": 100,
    }]


def test_modelscope_missing_dependency_mentions_data_extra(tmp_path, monkeypatch):
    import builtins

    from easyidp.data.downloader import _download_modelscope

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "modelscope":
            raise ImportError("missing modelscope")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        _download_modelscope(
            {"dataset_repo": "repo", "file_path": "file.zip"},
            tmp_path / "file.zip",
            progress=False,
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        assert False, "expected RuntimeError for missing modelscope"

    assert "modelscope is required" in message
    assert "pip install 'easyidp[data]'" in message


def test_modelscope_downloader_rejects_sha256_mismatch(tmp_path, monkeypatch):
    import hashlib
    import sys
    import types

    from easyidp.data.downloader import _download_modelscope

    def fake_dataset_file_download(**kwargs):
        output = tmp_path / "sdk" / kwargs["file_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"wrong")
        return str(output)

    class FakeHubApi:
        def get_dataset_files(self, **kwargs):
            return [{
                "Path": "gdown_test.zip",
                "Size": 5,
                "Sha256": hashlib.sha256(b"right").hexdigest(),
            }]

    download_module = types.SimpleNamespace(dataset_file_download=fake_dataset_file_download)
    api_module = types.SimpleNamespace(HubApi=FakeHubApi)
    monkeypatch.setitem(sys.modules, "modelscope", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "modelscope.hub", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "modelscope.hub.file_download", download_module)
    monkeypatch.setitem(sys.modules, "modelscope.hub.api", api_module)

    try:
        _download_modelscope(
            {
                "dataset_repo": "HowcanoeWang/EasyIDP-Demo-Dataset",
                "file_path": "gdown_test.zip",
            },
            tmp_path / ".downloads" / "gdown_test.zip",
            progress=False,
        )
    except RuntimeError as exc:
        assert "SHA256 mismatch" in str(exc)
    else:
        assert False, "expected RuntimeError for SHA256 mismatch"
