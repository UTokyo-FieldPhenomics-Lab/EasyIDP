import easyidp as idp


def test_lotus_paths_are_short_namespaces(tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)

    assert lotus.name == "lotus"
    assert lotus.title == "Tanashi Lotus 2017"
    assert lotus.root == tmp_path / "2017_tanashi_lotus"
    assert lotus.archive == tmp_path / ".downloads" / "2017_tanashi_lotus.zip"
    assert lotus.shp == lotus.root / "plots.shp"
    assert lotus.photo == lotus.root / "20170531" / "photos"
    assert lotus.metashape.project == lotus.root / "170531.Lotus.psx"
    assert lotus.metashape.dom == lotus.root / "170531.Lotus.outputs" / "170531.Lotus_dom.tif"
    assert lotus.pix4d.project == lotus.root / "20170531"
    assert lotus.pix4d.param == lotus.root / "20170531" / "params"
    assert not hasattr(lotus, "ms")
    assert not hasattr(lotus, "p4d")


def test_forestbirds_paths_are_short_namespaces(tmp_path):
    birds = idp.data.ForestBirds(cache_root=tmp_path, notify_missing=False)

    assert birds.name == "forestbirds"
    assert birds.root == tmp_path / "2022_florida_forestbirds"
    assert birds.shp == birds.root / "Hidden_Little_grid.shp"
    assert birds.metashape.project == birds.root / "Hidden_Little_03_24_2022.psx"
    assert not hasattr(birds, "pix4d")
    assert not hasattr(birds, "ms")
    assert not hasattr(birds, "p4d")


def test_path_namespace_supports_nested_paths(tmp_path):
    from easyidp.data.dataset import _PathNamespace

    namespace = _PathNamespace(
        tmp_path,
        {"metashape": {"outputs": {"dom": "dom.tif"}}},
    )

    assert namespace.metashape.outputs.dom == tmp_path / "dom.tif"


def test_testdata_uses_same_manifest_paths_as_runtime(tmp_path):
    data = idp.data.TestData(cache_root=tmp_path, test_out=tmp_path / "out", notify_missing=False)

    assert data.name == "testdata"
    assert data.root == tmp_path / "data_for_tests"
    assert data.metashape.lotus_psx == data.root / "metashape" / "Lotus.psx"
    assert data.pix4d.lotus_folder == data.root / "pix4d" / "lotus_tanashi_full"
    assert not hasattr(data, "ms")
    assert not hasattr(data, "p4d")
    assert data.shp.lotus_shp == data.root / "shp_test" / "lotus_plots.shp"
    assert data.tiff.soyweed_part == data.root / "tiff_test" / "2_12.tif"
    assert data.test_out == tmp_path / "out"
    assert data.shp.out == tmp_path / "out" / "shp_test"
    assert data.cv.out == tmp_path / "out" / "cv_test"
    assert data.vis.out == tmp_path / "out" / "visual_test"
    assert data.b2r.out == tmp_path / "out" / "back2raw_test"


def test_constructor_does_not_create_cache_dirs(tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)

    assert not lotus.root.exists()
    assert not lotus.archive.parent.exists()


def test_is_ready_uses_required_keys_only(tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)
    for key in lotus.required:
        path = lotus.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    assert lotus.is_ready()


def test_dry_run_is_json_friendly(tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)
    plan = lotus.dry_run()

    assert plan["name"] == "lotus"
    assert plan["ready"] is False
    assert plan["needs_download"] is True
    assert isinstance(plan["root"], str)
    assert isinstance(plan["archive"], str)
    assert "plots.shp" in plan["missing"]


def test_data_root_comes_from_config(tmp_path, monkeypatch):
    from types import SimpleNamespace

    fake = SimpleNamespace(data_dir=tmp_path / "configured")
    monkeypatch.setattr(idp.config, "get", lambda: fake)
    lotus = idp.data.Lotus(notify_missing=False)
    assert lotus.root == tmp_path / "configured" / "2017_tanashi_lotus"


def test_public_api_is_small():
    expected = {"Lotus", "ForestBirds", "TestData", "list_datasets"}
    forbidden = {"DatasetRegistry", "registry", "user_data_dir", "PathNamespace"}

    missing = [n for n in expected if not hasattr(idp.data, n)]
    present = [n for n in forbidden if hasattr(idp.data, n)]

    assert not missing, f"Expected exports missing: {missing}"
    assert not present, f"Forbidden exports found: {present}"


# --- downloader unit tests ------------------------------------------------


def test_download_skips_ready_dataset(tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)
    for key in lotus.required:
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


class FakeResponse:
    def __init__(self, content=b"fake-zip-data", *, json_data=None,
                 status_code=200, headers=None):
        self._content = content
        self._json_data = json_data
        self.status_code = status_code
        self.headers = headers or {}

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
