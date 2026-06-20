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

    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)

    def fake_download_openxlab(mirror_config, archive, progress):
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zf.ZipFile(archive, "w") as z:
            for key in ("shp", "metashape.project", "pix4d.dom", "pix4d.dsm"):
                z.writestr(str(lotus.path(key).relative_to(lotus.root)), "data")

    monkeypatch.setattr(
        idp.data.downloader,
        "_download_openxlab",
        fake_download_openxlab,
    )

    result = lotus.download(mirror="openxlab", progress=False)

    assert result["downloaded"] is True
    assert result["extracted"] is True
    assert result["ready"] is True
    assert not lotus._archive_path().exists()


class FakeResponse:
    def __init__(self, content=b"fake-zip-data", *, json_data=None,
                 status_code=200, headers=None):
        self._content = content
        self._json_data = json_data
        self.status_code = status_code
        self.headers = headers or {}
        self.chunk_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data

    def iter_content(self, chunk_size=None):
        self.chunk_sizes.append(chunk_size)
        yield self._content


_OPENXLAB_CDN_URL = (
    "https://cdn-xlab-data.openxlab.org.cn/objects/"
    "b353aee3743d29d968b09222c5a3b208268cce09100ab73b4048cb7cf42a844c"
)
_EXPECTED_SHA256 = "b353aee3743d29d968b09222c5a3b208268cce09100ab73b4048cb7cf42a844c"
_OPENXLAB_API_JSON = {
    "code": 0,
    "data": {
        "url": _OPENXLAB_CDN_URL,
        "meta": {"size": 280},
    },
}


def test_openxlab_anonymous_downloader_uses_cdn_url(tmp_path):
    from unittest.mock import patch

    from easyidp.data.downloader import _download_openxlab

    mirror_config = {
        "dataset_repo": "HowcanoeWang/easyidp-demo-dataset",
        "source_path": "/gdown_test.zip",
    }
    archive = tmp_path / ".downloads" / "gdown_test.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)

    post_mock = FakeResponse(b"", json_data=_OPENXLAB_API_JSON)
    get_mock = FakeResponse(b"fake-zip-content" * 18)

    post_called = []
    get_called = []

    def fake_post(url, **kwargs):
        post_called.append((url, kwargs))
        return post_mock

    def fake_get(url, **kwargs):
        get_called.append((url, kwargs))
        return get_mock

    with patch("requests.post", side_effect=fake_post), \
            patch("requests.get", side_effect=fake_get):
        try:
            _download_openxlab(mirror_config, archive, progress=False)
        except RuntimeError:
            pass

    assert len(post_called) == 1
    url, kwargs = post_called[0]
    assert "openxlab.org.cn/datasets/api/v3/datasets/" in url
    assert "HowcanoeWang,easyidp-demo-dataset/r/main" in url
    assert kwargs["json"] == {"path": "gdown_test.zip", "preview": False}
    assert kwargs["timeout"] == 60

    assert len(get_called) == 1
    url, kwargs = get_called[0]
    assert url == _OPENXLAB_CDN_URL
    assert kwargs["stream"] is True
    assert kwargs["timeout"] == 180


def test_openxlab_downloader_rejects_sha256_mismatch(tmp_path):
    from unittest.mock import patch

    from easyidp.data.downloader import _download_openxlab

    mirror_config = {
        "dataset_repo": "HowcanoeWang/easyidp-demo-dataset",
        "source_path": "/gdown_test.zip",
    }
    archive = tmp_path / ".downloads" / "gdown_test.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)

    post_mock = FakeResponse(b"", json_data=_OPENXLAB_API_JSON)
    wrong_content = b"x" * 280
    get_mock = FakeResponse(wrong_content)

    with patch("requests.post", return_value=post_mock), \
            patch("requests.get", return_value=get_mock):
        try:
            _download_openxlab(mirror_config, archive, progress=False)
        except RuntimeError as exc:
            assert "SHA256 mismatch" in str(exc)
        else:
            assert False, "expected RuntimeError for SHA256 mismatch"


def test_openxlab_downloader_uses_tqdm_when_progress_enabled(tmp_path):
    import hashlib

    from unittest.mock import patch

    from easyidp.data.downloader import _download_openxlab

    content = b"x" * 280
    valid_sha = hashlib.sha256(content).hexdigest()
    cdn_url = _OPENXLAB_CDN_URL.replace(_EXPECTED_SHA256, valid_sha)
    api_json = {
        "code": 0,
        "data": {
            "url": cdn_url,
            "meta": {"size": 280},
        },
    }

    mirror_config = {
        "dataset_repo": "HowcanoeWang/easyidp-demo-dataset",
        "source_path": "/gdown_test.zip",
    }
    archive = tmp_path / ".downloads" / "gdown_test.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)

    post_mock = FakeResponse(b"", json_data=api_json)
    get_mock = FakeResponse(content)

    with patch("requests.post", return_value=post_mock), \
            patch("requests.get", return_value=get_mock), \
            patch("easyidp.data.downloader.tqdm") as mock_tqdm:
        mock_tqdm.return_value.__enter__ = lambda s: s
        mock_tqdm.return_value.__exit__ = lambda *a: None
        mock_tqdm.return_value.update = lambda n: None

        _download_openxlab(mirror_config, archive, progress=True)

        assert mock_tqdm.called, "tqdm should be used when progress=True"


def test_openxlab_downloader_prints_source_and_target(tmp_path, capsys):
    import hashlib

    from unittest.mock import patch

    from easyidp.data.downloader import _download_openxlab

    content = b"x" * 280
    valid_sha = hashlib.sha256(content).hexdigest()
    cdn_url = _OPENXLAB_CDN_URL.replace(_EXPECTED_SHA256, valid_sha)
    api_json = {
        "code": 0,
        "data": {
            "url": cdn_url,
            "meta": {"size": 280},
        },
    }
    mirror_config = {
        "dataset_repo": "HowcanoeWang/easyidp-demo-dataset",
        "source_path": "/gdown_test.zip",
    }
    archive = tmp_path / ".downloads" / "gdown_test.zip"

    with patch("requests.post", return_value=FakeResponse(b"", json_data=api_json)), \
            patch("requests.get", return_value=FakeResponse(content)), \
            patch("easyidp.data.downloader.tqdm") as mock_tqdm:
        mock_tqdm.return_value.__enter__ = lambda s: s
        mock_tqdm.return_value.__exit__ = lambda *a: None
        mock_tqdm.return_value.update = lambda n: None

        _download_openxlab(mirror_config, archive, progress=True)

    output = capsys.readouterr().out
    assert "Downloading...\n" in output
    assert (
        "From (original): https://openxlab.org.cn/datasets/"
        "HowcanoeWang/easyidp-demo-dataset/gdown_test.zip\n"
    ) in output
    assert f"From (resolved): {cdn_url}\n" in output
    assert f"To:  {archive.with_suffix(archive.suffix + '.part')}\n" in output


def test_openxlab_downloader_stays_quiet_without_progress(tmp_path, capsys):
    import hashlib

    from unittest.mock import patch

    from easyidp.data.downloader import _download_openxlab

    content = b"x" * 280
    valid_sha = hashlib.sha256(content).hexdigest()
    cdn_url = _OPENXLAB_CDN_URL.replace(_EXPECTED_SHA256, valid_sha)
    api_json = {
        "code": 0,
        "data": {
            "url": cdn_url,
            "meta": {"size": 280},
        },
    }
    mirror_config = {
        "dataset_repo": "HowcanoeWang/easyidp-demo-dataset",
        "source_path": "/gdown_test.zip",
    }
    archive = tmp_path / ".downloads" / "gdown_test.zip"

    with patch("requests.post", return_value=FakeResponse(b"", json_data=api_json)), \
            patch("requests.get", return_value=FakeResponse(content)):
        _download_openxlab(mirror_config, archive, progress=False)

    assert capsys.readouterr().out == ""


def test_stream_download_uses_small_chunks_for_smooth_progress(tmp_path):
    import hashlib

    from unittest.mock import patch

    from easyidp.data.downloader import _stream_download

    content = b"x" * 1024
    response = FakeResponse(content)
    expected_sha = hashlib.sha256(content).hexdigest()

    with patch("requests.get", return_value=response):
        _stream_download(
            "https://example.com/archive.zip",
            tmp_path / "archive.zip",
            expected_size=len(content),
            expected_sha256=expected_sha,
            progress=False,
        )

    assert response.chunk_sizes == [512 * 1024]
