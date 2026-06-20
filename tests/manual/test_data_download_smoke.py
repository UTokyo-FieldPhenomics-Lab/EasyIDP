"""Smoke tests for real download flows. Run with EASYIDP_RUN_DOWNLOAD_SMOKE=1."""

import os

import easyidp as idp
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("EASYIDP_RUN_DOWNLOAD_SMOKE") != "1",
    reason="Set EASYIDP_RUN_DOWNLOAD_SMOKE=1 to run download smoke tests",
)


def test_gdrive_tiny_download_smoke(tmp_path):
    ds = idp.data.Dataset("download_smoke", cache_root=tmp_path, notify_missing=False)

    result = ds.download(mirror="gdrive", force=True, progress=False)

    assert result["name"] == "download_smoke"
    assert result["downloaded"] is True
    assert result["extracted"] is True
    assert result["ready"] is True
    assert ds.is_ready()


def test_openxlab_tiny_download_smoke(tmp_path):
    ds = idp.data.Dataset("download_smoke", cache_root=tmp_path, notify_missing=False)

    result = ds.download(mirror="openxlab", force=True, progress=True)

    assert result["name"] == "download_smoke"
    assert result["downloaded"] is True
    assert result["extracted"] is True
    assert result["ready"] is True
    assert ds.is_ready()
