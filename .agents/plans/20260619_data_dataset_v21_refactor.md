# Data/Dataset v2.1 Simplified Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old download-heavy `easyidp.data` module with a small official demo-data shortcut layer backed by JSON dataset manifests.

**Architecture:** `easyidp.data` is not a general dataset framework. It only exposes official demo datasets through short path attributes such as `idp.data.Lotus().metashape.project` so docs and examples do not need long file strings. Dataset metadata and file mappings live in `src/easyidp/data/datasets/*.json`; `dataset.py` parses those manifests and builds path namespaces dynamically.

**Tech Stack:** Python 3.10+, `json`, `dataclasses`, `pathlib`, `zipfile`, `pytest`, optional `gdown`, optional `modelscope>=1.37.1`, `uv`.

---

## Starting Point

- Base commit: `bac752b2ac337b2948fc56a37d2a94743fab9bbf`.
- Work branch: create from the base commit, for example `data-json-plan`.
- Current baseline note: `uv run pytest tests/test_config.py -q` may fail at collection with `ModuleNotFoundError: No module named 'easyidp'` in this old worktree. Resolve packaging/test collection before relying on full-suite results.

## Design Decisions

- Keep only short class entry points: `idp.data.Lotus()`, `idp.data.ForestBirds()`, and `idp.data.TestData()`.
- Remove `ALIASES`, `DatasetRegistry`, `registry`, `_builtin.py`, `paths.py`, `errors.py`, `testing.py`, `get_spec()`, `get_dataset()`, `download_all()`, `show_data_dir()`, `url_checker()`, and `user_data_dir()`.
- Do not download, extract, prompt, import `oss2`/`openxlab`/`modelscope`, or perform network checks during import or dataset construction.
- Replace Aliyun OSS downloads with ModelScope public dataset downloads for mainland China. EasyIDP no longer ships or fetches maintainer-owned OSS credentials, and normal public dataset downloads must not require OpenXLab Access Key, OpenXLab Secret Key, ModelScope token, or interactive login.
- ModelScope dataset page: `https://modelscope.cn/datasets/HowcanoeWang/EasyIDP-Demo-Dataset/files`; use `dataset_repo: "HowcanoeWang/EasyIDP-Demo-Dataset"` in manifests and downloader tests.
- Keep `gdown` and `modelscope` out of core dependencies as optional data extras. Use ModelScope's official SDK single-file API `modelscope.hub.file_download.dataset_file_download()` for China mirror downloads instead of OpenXLab's anonymous CDN workaround.
- Read the default data root only from `idp.config.get()`. `EasyIDPConfig` is a pure Python dataclass/json config object, not Pydantic.
- Use built-in exceptions where possible: `ValueError` for bad manifests or mirror names, `RuntimeError` for failed downloads, ModelScope SDK errors, size/SHA256 mismatches, and `zipfile.BadZipFile` for invalid archives.
- ModelScope behavior is based on `.agents/references/20260622_modelscope_dataset_mirror.md`. The verified SDK path is `dataset_file_download(dataset_id="HowcanoeWang/EasyIDP-Demo-Dataset", file_path="gdown_test.zip", ...)` with no login or token. EasyIDP manifests store only `dataset_repo` and `file_path`; fetch `size_bytes` and `sha256` dynamically through `HubApi.get_dataset_files()` before verification.
- Do not keep compatibility with the current temporary v2.1 `DatasetSpec`/registry API because it has not shipped.
- Documentation must describe `data` as optional demo data convenience, not as a required data ingestion path. Core EasyIDP APIs continue accepting normal file paths directly.

## 2026-06-22 ModelScope Mirror Reference

This section supersedes the earlier OpenXLab mirror plan. Keep the OpenXLab investigation only as historical context in `.agents/references/20260620_openxlab_anonymous_cdn_download.md`.

The latest ModelScope project metadata was verified with `modelscope==1.37.1` through `HubApi.dataset_info()`, `HubApi.get_dataset_files()`, and `HubApi.get_dataset_file_url()`. The dataset is public and did not require login or a token during testing.

```json
{
  "dataset": {
    "dataset_id": "HowcanoeWang/EasyIDP-Demo-Dataset",
    "hub_id": 201707,
    "name": "EasyIDP-Demo-Dataset",
    "namespace": "HowcanoeWang",
    "license": "MIT License",
    "private": false,
    "gated": false,
    "login_required": false,
    "last_commit": "e87035f1eb7687f69e0596b7b2730a6acceb3a25",
    "last_modified": "2026-06-22 11:30:18+08:00",
    "url": "https://modelscope.cn/datasets/HowcanoeWang/EasyIDP-Demo-Dataset/files"
  },
  "mirrors": {
    "lotus": {
      "modelscope": {
        "dataset_repo": "HowcanoeWang/EasyIDP-Demo-Dataset",
        "file_path": "2017_tanashi_lotus.zip",
        "download_url": "https://www.modelscope.cn/api/v1/datasets/HowcanoeWang/EasyIDP-Demo-Dataset/repo?Source=SDK&Revision=master&FilePath=2017_tanashi_lotus.zip&View=False"
      }
    },
    "forestbirds": {
      "modelscope": {
        "dataset_repo": "HowcanoeWang/EasyIDP-Demo-Dataset",
        "file_path": "2022_florida_forestbirds.zip",
        "download_url": "https://www.modelscope.cn/api/v1/datasets/HowcanoeWang/EasyIDP-Demo-Dataset/repo?Source=SDK&Revision=master&FilePath=2022_florida_forestbirds.zip&View=False"
      }
    },
    "testdata": {
      "modelscope": {
        "dataset_repo": "HowcanoeWang/EasyIDP-Demo-Dataset",
        "file_path": "data_for_tests.zip",
        "download_url": "https://www.modelscope.cn/api/v1/datasets/HowcanoeWang/EasyIDP-Demo-Dataset/repo?Source=SDK&Revision=master&FilePath=data_for_tests.zip&View=False"
      }
    },
    "download_smoke": {
      "modelscope": {
        "dataset_repo": "HowcanoeWang/EasyIDP-Demo-Dataset",
        "file_path": "gdown_test.zip",
        "download_url": "https://www.modelscope.cn/api/v1/datasets/HowcanoeWang/EasyIDP-Demo-Dataset/repo?Source=SDK&Revision=master&FilePath=gdown_test.zip&View=False"
      }
    }
  }
}
```

Use `dataset_repo` and `file_path` in EasyIDP manifests. Treat `download_url` as reference/debug output only; production code should let the ModelScope SDK generate the URL and should fetch verification metadata dynamically to avoid coupling EasyIDP manifests to mirrored file metadata.

## Target File Structure

```text
src/easyidp/
  config.py                         # Existing JSON-backed package config.
  __init__.py                        # Remove user_data_dir; keep config export.
  data/
    __init__.py                      # Public exports only.
    dataset.py                       # Dataset, internal _PathNamespace, Lotus, ForestBirds, TestData.
    downloader.py                    # Explicit archive download and safe extraction.
    datasets/
      lotus.json                     # Lotus manifest.
      forestbirds.json               # ForestBirds manifest.
      testdata.json                  # TestData manifest.
      download_smoke.json            # Tiny manual-only mirror smoke-test manifest.
```

## Manifest Shape

Each JSON manifest must use this shape:

```json
{
  "spec": {
    "name": "lotus",
    "title": "Tanashi Lotus 2017",
    "archive": "2017_tanashi_lotus.zip",
    "folder": "2017_tanashi_lotus",
    "size_bytes": 3577087822,
    "mirrors": {
      "gdrive": {
        "file_id": "1SJmp-bG5SZrwdeJL-RnnljM2XmMNMF0j"
      },
      "modelscope": {
        "dataset_repo": "HowcanoeWang/EasyIDP-Demo-Dataset",
        "file_path": "2017_tanashi_lotus.zip"
      }
    },
    "description": "Official EasyIDP demo dataset."
  },
  "required": ["shp", "ms.project", "p4d.dom"],
  "files": {
    "photo": "20170531/photos",
    "shp": "plots.shp",
    "ms.project": "170531.Lotus.psx",
    "ms.param": "170531.Lotus.files",
    "ms.dom": "170531.Lotus.outputs/170531.Lotus_dom.tif",
    "ms.dsm": "170531.Lotus.outputs/170531.Lotus_dsm.tif",
    "ms.pcd": "170531.Lotus.outputs/170531.Lotus.laz",
    "p4d.project": "20170531",
    "p4d.param": "20170531/params"
  }
}
```

---

### Task 1: Replace Data Tests With Simplified API Tests

**Files:**

- Modify: `tests/test_data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Replace old network-heavy tests**

Replace `tests/test_data.py` with tests for JSON manifests, short paths, no import-time download, and explicit config-driven roots:

```python
from pathlib import Path

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


def test_forestbirds_paths_are_short_namespaces(tmp_path):
    birds = idp.data.ForestBirds(cache_root=tmp_path, notify_missing=False)

    assert birds.name == "forestbirds"
    assert birds.root == tmp_path / "2022_florida_forestbirds"
    assert birds.shp == birds.root / "Hidden_Little_grid.shp"
    assert birds.metashape.project == birds.root / "Hidden_Little_03_24_2022.psx"
    assert not hasattr(birds, "pix4d")


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
    assert data.shp.lotus_shp == data.root / "shp_test" / "lotus_plots.shp"
    assert data.tiff.soyweed_part == data.root / "tiff_test" / "2_12.tif"
    assert data.test_out == tmp_path / "out"
    assert data.shp.out == tmp_path / "out" / "shp_test"


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


def test_data_root_comes_from_config(tmp_path):
    old_dir = idp.config.get().data_dir
    try:
        idp.config.update(data_dir=tmp_path / "configured")
        lotus = idp.data.Lotus(notify_missing=False)
        assert lotus.root == tmp_path / "configured" / "2017_tanashi_lotus"
    finally:
        idp.config.update(data_dir=old_dir)


def test_public_api_is_small():
    assert hasattr(idp.data, "Lotus")
    assert hasattr(idp.data, "ForestBirds")
    assert hasattr(idp.data, "TestData")
    assert hasattr(idp.data, "list_datasets")
    assert not hasattr(idp.data, "DatasetRegistry")
    assert not hasattr(idp.data, "registry")
    assert not hasattr(idp.data, "user_data_dir")
    assert not hasattr(idp.data, "PathNamespace")
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/test_data.py -q`

Expected: fail because the old `easyidp.data` is still a single module with network-heavy classes and no JSON-backed namespaces.

---

### Task 2: Add JSON Manifests

**Files:**

- Create: `src/easyidp/data/datasets/lotus.json`
- Create: `src/easyidp/data/datasets/forestbirds.json`
- Create: `src/easyidp/data/datasets/testdata.json`
- Create: `src/easyidp/data/datasets/download_smoke.json`
- Test: `tests/test_data.py`

- [ ] **Step 1: Create `lotus.json`**

```json
{
  "spec": {
    "name": "lotus",
    "title": "Tanashi Lotus 2017",
    "folder": "2017_tanashi_lotus",
    "archive": "2017_tanashi_lotus.zip",
    "size_bytes": 3577087822,
    "mirrors": {
      "gdrive": {
        "file_id": "1SJmp-bG5SZrwdeJL-RnnljM2XmMNMF0j"
      },
      "modelscope": {
        "dataset_repo": "HowcanoeWang/EasyIDP-Demo-Dataset",
        "file_path": "2017_tanashi_lotus.zip"
      }
    },
    "description": "Official EasyIDP lotus demo dataset from Tanashi, Tokyo."
  },
  "required": ["shp", "ms.project", "p4d.dom", "p4d.dsm"],
  "files": {
    "photo": "20170531/photos",
    "shp": "plots.shp",
    "ms.project": "170531.Lotus.psx",
    "ms.param": "170531.Lotus.files",
    "ms.dom": "170531.Lotus.outputs/170531.Lotus_dom.tif",
    "ms.dsm": "170531.Lotus.outputs/170531.Lotus_dsm.tif",
    "ms.pcd": "170531.Lotus.outputs/170531.Lotus.laz",
    "p4d.project": "20170531",
    "p4d.param": "20170531/params",
    "p4d.dom": "20170531/hasu_tanashi_20170531_Ins1RGB_30m_transparent_mosaic_group1.tif",
    "p4d.dsm": "20170531/hasu_tanashi_20170531_Ins1RGB_30m_dsm.tif",
    "p4d.pcd": "20170531/hasu_tanashi_20170531_Ins1RGB_30m_group1_densified_point_cloud.ply"
  }
}
```

- [ ] **Step 2: Create `forestbirds.json`**

```json
{
  "spec": {
    "name": "forestbirds",
    "title": "Florida Forest Birds 2022",
    "folder": "2022_florida_forestbirds",
    "archive": "2022_florida_forestbirds.zip",
    "size_bytes": 2116376783,
    "mirrors": {
      "gdrive": {
        "file_id": "1mXkzaoSSCAA87cxcMHKL6_VNlykRYxJr"
      },
      "modelscope": {
        "dataset_repo": "HowcanoeWang/EasyIDP-Demo-Dataset",
        "file_path": "2022_florida_forestbirds.zip"
      }
    },
    "description": "Official EasyIDP forest birds demo dataset from Florida."
  },
  "required": ["shp", "ms.project", "ms.dom", "ms.dsm"],
  "files": {
    "photo": "Hidden_Little_03_24_2022",
    "shp": "Hidden_Little_grid.shp",
    "ms.project": "Hidden_Little_03_24_2022.psx",
    "ms.param": "Hidden_Little_03_24_2022.files",
    "ms.dom": "Hidden_Little_03_24_2022.tiff",
    "ms.dsm": "Hidden_Little_03_24_2022_DEM.tif"
  }
}
```

- [ ] **Step 3: Create `testdata.json`**

Use the paths from old `TestData` and keep one manifest as the single source of truth:

```json
{
  "spec": {
    "name": "testdata",
    "title": "EasyIDP Test Data",
    "folder": "data_for_tests",
    "archive": "data_for_tests.zip",
    "size_bytes": 581849811,
    "mirrors": {
      "gdrive": {
        "file_id": "17b_17CofqIuCVOWMnD67_wOnWMtwF8bw"
      },
      "modelscope": {
        "dataset_repo": "HowcanoeWang/EasyIDP-Demo-Dataset",
        "file_path": "data_for_tests.zip"
      }
    },
    "description": "Official EasyIDP developer test data."
  },
  "required": ["shp.lotus_shp", "p4d.lotus_folder", "ms.lotus_psx"],
  "files": {
    "json.for_read_json": "json_test/for_read_json.json",
    "json.labelme_demo": "json_test/labelme_demo_img.json",
    "json.labelme_warn": "json_test/labelme_warn_img.json",
    "json.labelme_err": "json_test/for_read_json.json",
    "json.geojson_soy": "json_test/2023_soybean_field.geojson",
    "ms.goya_psx": "metashape/goya_test.psx",
    "ms.goya_param": "metashape/goya_test.files",
    "ms.lotus_psx": "metashape/Lotus.psx",
    "ms.lotus_param": "metashape/Lotus.files",
    "ms.lotus_dsm": "metashape/Lotus.files/170531.Lotus_dsm.tif",
    "ms.wheat_psx": "metashape/wheat_tanashi.psx",
    "ms.wheat_param": "metashape/wheat_tanashi.files",
    "ms.multichunk_psx": "metashape/multichunk.psx",
    "ms.multichunk_param": "metashape/multichunk.files",
    "ms.multifolder_psx": "metashape/multifolder.psx",
    "ms.multifolder_param": "metashape/multifolder.files",
    "ms.nestedfolder_psx": "metashape/nestedfolders.psx",
    "ms.nestedfolder_param": "metashape/nestedfolders.files",
    "ms.camera_disorder_psx": "metashape/camera_disorder.psx",
    "ms.camera_disorder_param": "metashape/camera_disorder.files",
    "ms.two_calib_psx": "metashape/two_calib.psx",
    "ms.two_calib_param": "metashape/two_calib.files",
    "ms.multi_spectral_psx": "metashape/multi_spectral.psx",
    "ms.multi_spectral_param": "metashape/multi_spectral.files",
    "p4d.lotus_folder": "pix4d/lotus_tanashi_full",
    "p4d.lotus_param": "pix4d/lotus_tanashi_full/params",
    "p4d.lotus_photos": "pix4d/lotus_tanashi_full/photos",
    "p4d.lotus_dom": "pix4d/lotus_tanashi_full/hasu_tanashi_20170525_Ins1RGB_30m_transparent_mosaic_group1.tif",
    "p4d.lotus_dsm": "pix4d/lotus_tanashi_full/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif",
    "p4d.lotus_pcd": "pix4d/lotus_tanashi_full/hasu_tanashi_20170525_Ins1RGB_30m_group1_densified_point_cloud.ply",
    "p4d.lotus_dom_part": "pix4d/lotus_tanashi_full/plot_dom.tif",
    "p4d.lotus_dsm_part": "pix4d/lotus_tanashi_full/plot_dsm.tif",
    "p4d.lotus_pcd_part": "pix4d/lotus_tanashi_full/plot_pcd.ply",
    "p4d.maize_folder": "pix4d/maize_tanashi/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d",
    "p4d.maize_dom": "pix4d/maize_tanashi/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d/3_dsm_ortho/2_mosaic/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d_transparent_mosaic_group1.tif",
    "p4d.maize_dsm": "pix4d/maize_tanashi/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d/3_dsm_ortho/1_dsm/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d_dsm.tif",
    "p4d.maize_noparam": "pix4d/maize_tanashi/maize_tanashi_no_param",
    "p4d.maize_empty": "pix4d/maize_tanashi/maize_tanashi_raname_empty_test",
    "p4d.maize_noout": "pix4d/maize_tanashi/maize_tanashi_raname_no_outputs",
    "shp.lotus_shp": "shp_test/lotus_plots.shp",
    "shp.lotus_prj": "shp_test/lotus_plots.prj",
    "shp.complex_shp": "shp_test/complex_shp_review.shp",
    "shp.complex_prj": "shp_test/complex_shp_review.prj",
    "shp.lonlat_shp": "shp_test/lon_lat.shp",
    "shp.utm53n_shp": "shp_test/lon_lat_utm53n.shp",
    "shp.utm53n_prj": "shp_test/lon_lat_utm53n.prj",
    "shp.rice_shp": "shp_test/rice_ind_duplicate.shp",
    "shp.rice_prj": "shp_test/rice_ind_duplicate.prj",
    "shp.roi_shp": "shp_test/roi.shp",
    "shp.roi_prj": "shp_test/roi.prj",
    "shp.testutm_shp": "shp_test/test_utm.shp",
    "shp.testutm_prj": "shp_test/test_utm.prj",
    "shp.jp_crs_shp": "shp_test/jp_crs.shp",
    "shp.jp_crs_prj": "shp_test/jp_crs.prj",
    "shp.mlayer_shp": "shp_test/mlayer_roi.shp",
    "shp.mask_rice_roi": "shp_test/mask_rice_grid_32.shp",
    "shp.mask_rice_prj": "shp_test/mask_rice_grid_32.prj",
    "shp.mask_rice_gt_shp": "shp_test/mask_rice_train_true_value.shp",
    "shp.mask_rice_gt_prj": "shp_test/mask_rice_train_true_value.prj",
    "pcd.lotus_las": "pcd_test/hasu_tanashi.las",
    "pcd.lotus_laz": "pcd_test/hasu_tanashi.laz",
    "pcd.lotus_pcd": "pcd_test/hasu_tanashi.pcd",
    "pcd.lotus_las13": "pcd_test/hasu_tanashi_1.3.las",
    "pcd.lotus_laz13": "pcd_test/hasu_tanashi_1.3.laz",
    "pcd.lotus_ply_asc": "pcd_test/hasu_tanashi_ascii.ply",
    "pcd.lotus_ply_bin": "pcd_test/hasu_tanashi_binary.ply",
    "pcd.maize_las": "pcd_test/maize3na_20210614_15m_utm.las",
    "pcd.maize_laz": "pcd_test/maize3na_20210614_15m_utm.laz",
    "pcd.maize_ply": "pcd_test/maize3na_20210614_15m_utm.ply",
    "roi.dxf": "roi_test/hasu_tanashi_ccroi.dxf",
    "roi.lxyz_txt": "roi_test/hasu_tanashi_lxyz.txt",
    "roi.xyz_txt": "roi_test/hasu_tanashi_xyz.txt",
    "tiff.soyweed_part": "tiff_test/2_12.tif",
    "tiff.mlayer_ndvi": "tiff_test/mlayer_yamato_ndvi.tif",
    "tiff.mlayer_multi": "tiff_test/mlayer_yamato_multi.tif",
    "tiff.mask_rice_geotiff_empty_polygon": "tiff_test/mask_rice_grid_48.tif",
    "tiff.mask_rice_geotiff_with_polygon": "tiff_test/mask_rice_grid_77.tif"
  }
}
```

- [ ] **Step 4: Create `download_smoke.json`**

This manifest is only for explicit manual mirror checks. Do not expose it in `list_datasets()` or user-facing docs.

```json
{
  "spec": {
    "name": "download_smoke",
    "title": "EasyIDP Download Smoke Test",
    "folder": "download_smoke",
    "archive": "gdown_test.zip",
    "size_bytes": 280,
    "mirrors": {
      "gdrive": {
        "file_id": "1yWvIOYJ1ML-UGleh3gT5b7dxXzBuSPgQ"
      },
      "modelscope": {
        "dataset_repo": "HowcanoeWang/EasyIDP-Demo-Dataset",
        "file_path": "gdown_test.zip"
      }
    },
    "description": "Tiny archive for manual gdown and ModelScope download smoke tests."
  },
  "required": ["file1"],
  "files": {
    "file1": "file1.txt",
    "folder1": "folder1"
  }
}
```

- [ ] **Step 5: Run tests and confirm they still fail on missing implementation**

Run: `uv run pytest tests/test_data.py -q`

Expected: fail because `dataset.py` and the package layout are not implemented yet.

---

### Task 3: Replace `data.py` With a Small Data Package

**Files:**

- Delete: `src/easyidp/data.py`
- Create: `src/easyidp/data/__init__.py`
- Create: `src/easyidp/data/dataset.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Create `src/easyidp/data/dataset.py`**

```python
"""Official EasyIDP demo dataset shortcuts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from easyidp import config
from easyidp.logger import logger


_MANIFEST_DIR = Path(__file__).parent / "datasets"


class _PathNamespace:
    """Recursive internal namespace for dataset paths.

    Parameters
    ----------
    root : pathlib.Path
        Dataset root directory.
    tree : Mapping[str, str or Mapping]
        Nested relative path mapping for this namespace.

    Returns
    -------
    _PathNamespace
        Object whose attributes are absolute paths or nested namespaces.

    Examples
    --------
    >>> ns = _PathNamespace(Path('/data'), {'metashape': {'project': 'demo.psx'}})
    >>> ns.metashape.project
    PosixPath('/data/demo.psx')

    Notes
    -----
    This class is the object side of dotted manifest keys. For example,
    ``ms.outputs.dom`` becomes ``dataset.metashape.outputs.dom``. It is not
    exported from ``easyidp.data`` and is not part of the basic public API.
    """

    def __init__(self, root: Path, tree: Mapping[str, Any]) -> None:
        self._root = root
        for key, value in tree.items():
            if isinstance(value, Mapping):
                setattr(self, key, _PathNamespace(root, value))
                continue
            setattr(self, key, root / value)

    def __truediv__(self, other: str) -> Path:
        """Join a relative path below this namespace root."""
        return self._root / other


class Dataset:
    """Official demo dataset path bundle.

    Parameters
    ----------
    manifest_name : str
        Manifest file stem under ``data/datasets``.
    cache_root : pathlib.Path or str or None, optional
        Override for ``idp.config.get().data_dir``.
    notify_missing : bool, optional
        Log a warning when required files are missing.

    Returns
    -------
    Dataset
        Dataset path bundle with short attributes.

    Examples
    --------
    >>> lotus = Dataset('lotus', notify_missing=False)
    >>> lotus.name
    'lotus'
    """

    def __init__(
        self,
        manifest_name: str,
        cache_root: Path | str | None = None,
        notify_missing: bool = True,
    ) -> None:
        manifest = _load_manifest(manifest_name)
        spec = manifest["spec"]
        self.name = spec["name"]
        self.title = spec["title"]
        self.description = spec.get("description", "")
        self.size_bytes = spec.get("size_bytes")
        self.mirrors = MappingProxyType(dict(spec.get("mirrors", {})))
        self.required = tuple(manifest.get("required", ()))
        self.files = MappingProxyType(dict(manifest["files"]))

        root = Path(cache_root).expanduser() if cache_root else config.get().data_dir
        self.cache_root = root
        self.root = root / spec["folder"]
        self.archive = root / ".downloads" / spec["archive"]
        self._build_path_attributes()

        if notify_missing and not self.is_ready():
            logger.warning(
                "Dataset '{}' is not ready. Call .download() to fetch it.",
                self.name,
            )

    @property
    def data_dir(self) -> Path:
        """Legacy alias for the dataset root."""
        return self.root

    @property
    def zip_file(self) -> Path:
        """Legacy alias for the downloaded archive path."""
        return self.archive

    def path(self, key: str) -> Path:
        """Return the absolute path for a manifest file key."""
        return self.root / self.files[key]

    def is_ready(self) -> bool:
        """Return whether required local files already exist."""
        keys = self.required or tuple(self.files.keys())
        return all(self.path(key).exists() for key in keys)

    def dry_run(self) -> dict[str, Any]:
        """Return a JSON-friendly local readiness summary."""
        keys = self.required or tuple(self.files.keys())
        missing = [self.files[key] for key in keys if not self.path(key).exists()]
        return {
            "name": self.name,
            "root": str(self.root),
            "archive": str(self.archive),
            "ready": not missing,
            "needs_download": bool(missing),
            "size_bytes": self.size_bytes,
            "mirrors": dict(self.mirrors),
            "missing": missing,
        }

    def download(
        self,
        mirror: str = "auto",
        force: bool = False,
        progress: bool = True,
    ) -> dict[str, Any]:
        """Download and extract this dataset explicitly."""
        from .downloader import download_dataset

        return download_dataset(self, mirror, force, progress)

    def _build_path_attributes(self) -> None:
        tree: dict[str, Any] = {}
        for key, rel_path in self.files.items():
            _insert_path(tree, key.split("."), rel_path)
        for key, value in tree.items():
            if isinstance(value, Mapping):
                setattr(self, key, _PathNamespace(self.root, value))
                continue
            setattr(self, key, self.root / value)


class Lotus(Dataset):
    """Tanashi lotus official demo dataset."""

    def __init__(
        self,
        cache_root: Path | str | None = None,
        notify_missing: bool = True,
    ) -> None:
        super().__init__("lotus", cache_root, notify_missing)


class ForestBirds(Dataset):
    """Florida forest birds official demo dataset."""

    def __init__(
        self,
        cache_root: Path | str | None = None,
        notify_missing: bool = True,
    ) -> None:
        super().__init__("forestbirds", cache_root, notify_missing)


class TestData(Dataset):
    """Developer test-data path bundle."""

    __test__ = False

    def __init__(
        self,
        test_out: str | Path = "./tests/out",
        cache_root: Path | str | None = None,
        notify_missing: bool = True,
    ) -> None:
        super().__init__("testdata", cache_root, notify_missing)
        self.test_out = Path(test_out)
        self._attach_test_out()

    def _attach_test_out(self) -> None:
        """Attach runtime-only test output directories.

        JSON manifests only describe files shipped inside the downloaded
        dataset. Test output paths such as ``test_data.shp.out`` depend on
        the runtime ``test_out`` argument, so they are constructed here
        instead of encoded in ``testdata.json``.
        """
        groups = ("json", "shp", "pcd", "tiff", "cv", "vis", "b2r")
        names = {
            "json": "json_test",
            "shp": "shp_test",
            "pcd": "pcd_test",
            "tiff": "tiff_test",
            "cv": "cv_test",
            "vis": "visual_test",
            "b2r": "back2raw_test",
        }
        for group in groups:
            current = getattr(self, group, _PathNamespace(self.root, {}))
            setattr(current, "out", self.test_out / names[group])
            setattr(self, group, current)


def list_datasets() -> list[str]:
    """Return official demo dataset names."""
    return ["lotus", "forestbirds", "testdata"]


def _load_manifest(name: str) -> dict[str, Any]:
    path = _MANIFEST_DIR / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Unknown EasyIDP demo dataset: {name}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _validate_manifest(path, manifest)
    return manifest


def _insert_path(tree: dict[str, Any], parts: list[str], rel_path: str) -> None:
    """Insert one dotted manifest key into the namespace tree.

    Parameters
    ----------
    tree : dict[str, Any]
        Mutable nested namespace tree.
    parts : list[str]
        Dotted manifest key split into Python attribute names.
    rel_path : str
        Dataset-relative file path from the manifest.

    Returns
    -------
    None

    Notes
    -----
    Raises ``ValueError`` if a key would be both a path and namespace,
    for example ``metashape`` and ``ms.project`` in the same manifest.
    """
    current = tree
    for part in parts[:-1]:
        _validate_attr_name(part)
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Dataset path key conflict at: {'.'.join(parts)}")
        current = child

    leaf = parts[-1]
    _validate_attr_name(leaf)
    if leaf in current:
        raise ValueError(f"Duplicate or conflicting dataset path key: {'.'.join(parts)}")
    current[leaf] = rel_path


def _validate_attr_name(name: str) -> None:
    """Validate one manifest key part as a public Python attribute.

    Parameters
    ----------
    name : str
        Attribute name from a dotted manifest key.

    Returns
    -------
    None
    """
    if not name.isidentifier() or name.startswith("_"):
        raise ValueError(f"Invalid dataset path attribute name: {name}")


def _validate_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    for key in ("spec", "files"):
        if key not in manifest:
            raise ValueError(f"Dataset manifest {path} missing '{key}'")
    spec = manifest["spec"]
    for key in ("name", "title", "folder", "archive"):
        if key not in spec:
            raise ValueError(f"Dataset manifest {path} missing spec.{key}")
```

- [ ] **Step 2: Create `src/easyidp/data/__init__.py`**

```python
"""Official EasyIDP demo dataset shortcuts."""

from .dataset import Dataset, ForestBirds, Lotus, TestData, list_datasets

__all__ = [
    "Dataset",
    "ForestBirds",
    "Lotus",
    "TestData",
    "list_datasets",
]
```

- [ ] **Step 3: Delete `src/easyidp/data.py`**

Run: `rm src/easyidp/data.py`

Expected: the package directory `src/easyidp/data/` replaces the old module.

- [ ] **Step 4: Run data tests**

Run: `uv run pytest tests/test_data.py -q`

Expected: pass all tests in `tests/test_data.py`.

---

### Task 4: Add Explicit Downloader With ModelScope Mirror

**2026-06-22 update:** The original OpenXLab snippets in this task are superseded by the ModelScope mirror decision and the metadata in `2026-06-22 ModelScope Mirror Reference` above. When executing this task, implement `_download_modelscope()` with `modelscope.hub.file_download.dataset_file_download()`, update tests from `openxlab` to `modelscope`, and remove OpenXLab-specific API/URL parsing code instead of copying the older OpenXLab examples below.

**Files:**

- Modify: `pyproject.toml`
- Create: `src/easyidp/data/downloader.py`
- Modify: `tests/test_data.py`
- Create: `tests/manual/test_data_download_smoke.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Move Google Drive backend to optional dependencies**

Keep Google Drive and ModelScope download support out of `[project].dependencies`. If `gdown` is currently listed there, move it to optional extras. Do not add `openxlab` as a dependency. Add `modelscope>=1.37.1` to the optional `data` extra.

Add this block to `pyproject.toml`:

```toml
[project.optional-dependencies]
gdrive = [
    "gdown>=5.2.0",
]
data = [
    "gdown>=5.2.0",
    "modelscope>=1.37.1",
]
```

Users who need Google Drive downloads can install `easyidp[gdrive]`. Users in mainland China can install `easyidp[data]` and use the ModelScope mirror without configuring credentials.

- [ ] **Step 2: Add downloader unit tests**

Append these tests to `tests/test_data.py`:

```python
import hashlib
import zipfile
from unittest.mock import Mock, patch

import pytest


def test_download_skips_ready_dataset(tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)
    for key in lotus.required:
        path = lotus.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    result = lotus.download(progress=False)

    assert result["ready"] is True
    assert result["downloaded"] is False
    assert result["extracted"] is False


def test_safe_extract_rejects_zip_slip(tmp_path):
    from easyidp.data.downloader import safe_extract_zip

    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "bad")

    with pytest.raises(RuntimeError, match="Unsafe archive member"):
        safe_extract_zip(archive, tmp_path / "out")


def test_download_extracts_mocked_gdrive_archive(tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("plots.shp", "shape")
        zf.writestr("170531.Lotus.psx", "project")
        zf.writestr(
            "20170531/hasu_tanashi_20170531_Ins1RGB_30m_transparent_mosaic_group1.tif",
            "dom",
        )
        zf.writestr(
            "20170531/hasu_tanashi_20170531_Ins1RGB_30m_dsm.tif",
            "dsm",
        )

    def fake_download(file_id, dest, progress):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(archive.read_bytes())

    with patch("easyidp.data.downloader._download_gdrive", side_effect=fake_download):
        result = lotus.download(force=True, progress=False)

    assert result["downloaded"] is True
    assert result["extracted"] is True
    assert result["ready"] is True


class FakeResponse:
    def __init__(self, json_data=None, content=b"", status_code=200, headers=None):
        self._json_data = json_data
        self._content = content
        self.status_code = status_code
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data

    def iter_content(self, chunk_size):
        yield self._content


def test_openxlab_anonymous_downloader_uses_cdn_url(tmp_path):
    from easyidp.data.downloader import _download_openxlab

    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)
    content = b"zip-content"
    sha256 = hashlib.sha256(content).hexdigest()
    cdn_url = f"https://cdn-xlab-data.openxlab.org.cn/objects/{sha256}?sig=1"
    post = Mock(return_value=FakeResponse({
        "code": 0,
        "data": {"url": cdn_url, "meta": {"size": len(content)}},
    }))
    get = Mock(return_value=FakeResponse(
        content=content,
        headers={"content-length": str(len(content))},
    ))

    with patch("easyidp.data.downloader.requests.post", post), patch(
        "easyidp.data.downloader.requests.get", get
    ):
        mirror = lotus.mirrors["openxlab"]
        _download_openxlab(mirror, lotus.archive, progress=False)

    assert lotus.archive.read_bytes() == content
    post.assert_called_once_with(
        "https://openxlab.org.cn/datasets/api/v3/datasets/HowcanoeWang,easyidp-demo-dataset/r/main",
        json={"path": "2017_tanashi_lotus.zip", "preview": False},
        timeout=60,
    )
    get.assert_called_once_with(cdn_url, stream=True, timeout=180)


def test_openxlab_downloader_rejects_sha256_mismatch(tmp_path):
    from easyidp.data.downloader import _download_openxlab

    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)
    wrong_sha = "0" * 64
    post = Mock(return_value=FakeResponse({
        "code": 0,
        "data": {
            "url": f"https://cdn-xlab-data.openxlab.org.cn/objects/{wrong_sha}",
            "meta": {"size": 3},
        },
    }))
    get = Mock(return_value=FakeResponse(content=b"zip"))

    with patch("easyidp.data.downloader.requests.post", post), patch(
        "easyidp.data.downloader.requests.get", get
    ), pytest.raises(RuntimeError, match="SHA256"):
        _download_openxlab(lotus.mirrors["openxlab"], lotus.archive, progress=False)


def test_openxlab_downloader_uses_tqdm_when_progress_enabled(tmp_path):
    from easyidp.data.downloader import _download_openxlab

    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)
    content = b"zip"
    sha256 = hashlib.sha256(content).hexdigest()
    post = Mock(return_value=FakeResponse({
        "code": 0,
        "data": {
            "url": f"https://cdn-xlab-data.openxlab.org.cn/objects/{sha256}",
            "meta": {"size": len(content)},
        },
    }))
    get = Mock(return_value=FakeResponse(content=content))
    progress_bar = Mock()
    progress_bar.__enter__ = Mock(return_value=progress_bar)
    progress_bar.__exit__ = Mock(return_value=False)

    with patch("easyidp.data.downloader.requests.post", post), patch(
        "easyidp.data.downloader.requests.get", get
    ), patch("easyidp.data.downloader.tqdm", return_value=progress_bar):
        _download_openxlab(lotus.mirrors["openxlab"], lotus.archive, progress=True)

    progress_bar.update.assert_called_once_with(len(content))
```

- [ ] **Step 3: Create `src/easyidp/data/downloader.py`**

```python
"""Explicit downloader for official EasyIDP demo datasets."""

from __future__ import annotations

import hashlib
import os
import re
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm.auto import tqdm


_OPENXLAB_DATASET_API = "https://openxlab.org.cn/datasets/api/v3/datasets"
_OPENXLAB_OBJECT_RE = re.compile(r"/objects/([0-9a-fA-F]{64})(?:[/?#]|$)")


def download_dataset(dataset, mirror, force, progress):
    """Download and extract a dataset archive."""
    if not force and dataset.is_ready():
        return _result(dataset, False, False, True)

    mirror_name, mirror_config = _select_mirror(dataset.mirrors, mirror)
    if mirror_name == "gdrive":
        _download_gdrive(mirror_config["file_id"], dataset.archive, progress)
    elif mirror_name == "openxlab":
        _download_openxlab(mirror_config, dataset.archive, progress)
    else:
        raise ValueError(f"Unknown dataset mirror: {mirror_name}")

    safe_extract_zip(dataset.archive, dataset.root)
    return _result(dataset, True, True, dataset.is_ready())


def safe_extract_zip(archive: Path, dest: Path) -> None:
    """Extract a zip archive while rejecting path traversal."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if os.path.commonpath([str(dest), str(target)]) != str(dest):
                raise RuntimeError(f"Unsafe archive member: {member}")
        zf.extractall(dest)


def _select_mirror(mirrors: dict, mirror: str) -> tuple[str, dict]:
    if not mirrors:
        raise RuntimeError("Dataset has no download mirrors")
    if mirror == "auto":
        name = next(iter(mirrors))
        return name, mirrors[name]
    if mirror in mirrors:
        return mirror, mirrors[mirror]
    raise ValueError(f"Unknown or unavailable mirror: {mirror}")


def _download_gdrive(file_id: str, archive: Path, progress: bool) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive downloads require the optional dependency. "
            "Install with `pip install 'easyidp[gdrive]'` or use mirror='openxlab'."
        ) from exc

    part = Path(str(archive) + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(part), quiet=not progress)
    if not part.exists() or part.stat().st_size == 0:
        raise RuntimeError(f"Download failed: {file_id}")
    os.replace(part, archive)


def _download_openxlab(mirror_config: dict, archive: Path, progress: bool) -> None:
    info = _fetch_openxlab_file_info(mirror_config)
    part = Path(str(archive) + ".part")
    _stream_download(info["url"], part, info["size"], info["sha256"], progress)
    os.replace(part, archive)


def _fetch_openxlab_file_info(mirror_config: dict) -> dict:
    dataset = mirror_config["dataset_repo"].replace("/", ",")
    source_path = mirror_config["source_path"].lstrip("/")
    url = f"{_OPENXLAB_DATASET_API}/{dataset}/r/main"
    response = requests.post(
        url,
        json={"path": source_path, "preview": False},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"OpenXLab metadata request failed: {payload}")

    data = payload.get("data") or {}
    download_url = data.get("url")
    size = (data.get("meta") or {}).get("size")
    sha256 = _sha256_from_openxlab_url(download_url)
    if not download_url or not size or not sha256:
        raise RuntimeError(f"OpenXLab metadata is incomplete: {payload}")
    return {"url": download_url, "size": int(size), "sha256": sha256}


def _sha256_from_openxlab_url(download_url: str | None) -> str:
    if not download_url:
        return ""
    match = _OPENXLAB_OBJECT_RE.search(urlparse(download_url).path)
    return match.group(1).lower() if match else ""


def _stream_download(
    url: str,
    output: Path,
    expected_size: int,
    expected_sha256: str,
    progress: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    bar = tqdm(total=expected_size, unit="B", unit_scale=True, disable=not progress)
    with requests.get(url, stream=True, timeout=180) as stream, bar:
        stream.raise_for_status()
        with output.open("wb") as file:
            for chunk in stream.iter_content(4 * 1024 * 1024):
                if not chunk:
                    continue
                file.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                bar.update(len(chunk))

    if size != expected_size:
        raise RuntimeError(f"Downloaded size mismatch: {size} != {expected_size}")
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError("Downloaded file SHA256 does not match OpenXLab metadata")


def _result(dataset, downloaded: bool, extracted: bool, ready: bool) -> dict:
    return {
        "name": dataset.name,
        "root": str(dataset.root),
        "archive": str(dataset.archive),
        "downloaded": downloaded,
        "extracted": extracted,
        "ready": ready,
    }
```

- [ ] **Step 4: Add manual network smoke tests**

Create `tests/manual/test_data_download_smoke.py`. These tests are skipped unless the developer explicitly opts in. No OpenXLab credentials are required for public datasets:

```python
import os

import pytest

import easyidp as idp


pytestmark = pytest.mark.skipif(
    os.environ.get("EASYIDP_RUN_DOWNLOAD_SMOKE") != "1",
    reason="Manual network smoke test. Set EASYIDP_RUN_DOWNLOAD_SMOKE=1 to run.",
)


def test_gdrive_tiny_download_smoke(tmp_path):
    data = idp.data.Dataset("download_smoke", cache_root=tmp_path, notify_missing=False)

    result = data.download(mirror="gdrive", force=True, progress=False)

    assert result["ready"] is True
    assert (data.root / "file1.txt").exists()


def test_openxlab_tiny_download_smoke(tmp_path):
    data = idp.data.Dataset("download_smoke", cache_root=tmp_path, notify_missing=False)

    result = data.download(mirror="openxlab", force=True, progress=True)

    assert result["ready"] is True
    assert (data.root / "file1.txt").exists()
```

Run manually with:

```bash
EASYIDP_RUN_DOWNLOAD_SMOKE=1 \
uv run pytest tests/manual/test_data_download_smoke.py -q
```

- [ ] **Step 5: Run normal tests**

Run: `uv run pytest tests/test_data.py -q`

Expected: all normal tests pass without network access.

---

### Task 5: Remove Root `user_data_dir` and Old Data Side Effects

**Files:**

- Modify: `src/easyidp/__init__.py`
- Modify: `tests/test_config.py`
- Test: `tests/test_config.py tests/test_data.py`

- [ ] **Step 1: Add config ownership assertions**

Append this test to `tests/test_config.py`:

```python
def test_user_data_dir_removed_from_package_root():
    assert not hasattr(idp, "user_data_dir")
    assert hasattr(idp, "config")
    assert idp.config.get().data_dir.name == "easyidp.data"
```

- [ ] **Step 2: Remove `user_data_dir` from `src/easyidp/__init__.py`**

Delete the function spanning the old `def user_data_dir(file_name=""):` block. Keep `get_full_path()` and `parse_relative_path()` unchanged.

- [ ] **Step 3: Remove unused root imports if possible**

If `os` is only used by `parse_relative_path()`, keep it. If `Path` is still used by `get_full_path()`, keep it.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_config.py tests/test_data.py -q`

Expected: config and data tests pass.

---

### Task 6: Update Test Fixtures That Consume TestData

**Files:**

- Modify: `tests/__init__.py`
- Modify: `tests/conftest.py`
- Search/modify: tests that reference `.metashape` or `.pix4d` on `TestData`
- Test: `tests/test_data.py` plus one data-consuming test module if local test data exists

- [ ] **Step 1: Replace old group names in test fixtures**

Update test fixtures to use the new short group names:

```python
data = idp.data.TestData(notify_missing=False)

# Old
# data.pix4d.lotus_folder
# data.metashape.lotus_psx

# New
data.pix4d.lotus_folder
data.metashape.lotus_psx
```

- [ ] **Step 2: Keep missing test data as skip behavior**

Use this fixture shape in `tests/conftest.py`:

```python
@pytest.fixture(scope="module")
def test_data():
    data = idp.data.TestData(notify_missing=False)
    if not data.is_ready():
        pytest.skip(
            "EasyIDP test data is not downloaded. "
            "Run `idp.data.TestData().download()` before data-dependent tests."
        )
    return data
```

- [ ] **Step 3: Update manual test-data downloader script**

In `tests/__init__.py`, use the explicit download flow:

```python
if __name__ == "__main__":
    import easyidp as idp

    print("Downloading test data...")
    data = idp.data.TestData(notify_missing=False)
    if not data.is_ready():
        data.download()
    print(f"Test data root: {data.root}")
```

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_data.py -q`

Expected: pass.

If local official test data exists, run one data-consuming module, for example: `uv run pytest tests/test_shp.py -q`.

Expected: pass or skip because data is not downloaded.

---

### Task 7: Update Existing Documentation

**Files:**

- Modify: `docs/python_api/data.rst`
- Modify: `docs/python_api/index.rst`
- Create: `docs/python_api/advanced.rst`
- Delete or stop referencing: `docs/python_api/autodoc/easyidp.data.download_all.rst`
- Delete or stop referencing: `docs/python_api/autodoc/easyidp.data.user_data_dir.rst`
- Delete or stop referencing: `docs/python_api/autodoc/easyidp.data.show_data_dir.rst`
- Delete or stop referencing: `docs/python_api/autodoc/easyidp.data.url_checker.rst`
- Delete or stop referencing: `docs/python_api/autodoc/easyidp.data.EasyidpDataSet.rst`
- Test: docs build if available

- [ ] **Step 1: Replace `docs/python_api/data.rst`**

Use this content:

```rst
====
Data
====

.. currentmodule:: easyidp.data

Purpose
=======

The data module is an optional shortcut for official EasyIDP demo datasets. It is not required for normal EasyIDP workflows. Most EasyIDP APIs accept ordinary file paths directly, so users can pass their own ``.shp``, ``.tif``, Pix4D, Metashape, or point-cloud paths without constructing an ``idp.data`` object.

The main purpose of this module is to keep examples readable:

.. code-block:: python

    import easyidp as idp

    lotus = idp.data.Lotus()
    roi = idp.ROI(lotus.shp)
    ms = idp.Metashape(lotus.metashape.project)

Construction is lightweight. It does not download or extract data. Call ``download()`` explicitly when needed:

.. code-block:: python

    lotus = idp.data.Lotus()
    if not lotus.is_ready():
        lotus.download()

Configuration
=============

The default data directory comes from ``idp.config``:

.. code-block:: python

    import easyidp as idp

    idp.config.update(data_dir="/path/to/easyidp.data")
    lotus = idp.data.Lotus()

OpenXLab downloads use anonymous public dataset CDN URLs. They do not require the OpenXLab SDK, login, Access Key, or Secret Key:

.. code-block:: python

    import easyidp as idp

    lotus = idp.data.Lotus()
    lotus.download(mirror="openxlab")

For Google Drive downloads, install the smaller optional backend instead:

.. code-block:: bash

    pip install "easyidp[gdrive]"

Datasets
========

.. autosummary::
    :toctree: autodoc

    Lotus
    ForestBirds
    TestData

Functions
=========

.. autosummary::
    :toctree: autodoc

    list_datasets
```

- [ ] **Step 2: Update API summary wording**

In `docs/python_api/index.rst`, change the data module summary to:

```rst
- :doc:`Data Module <./data>` : Optional official demo-data path shortcuts for examples and tutorials.
- :doc:`Advanced Notes <./advanced>` : Internal implementation notes for advanced users and contributors.
```

- [ ] **Step 3: Add advanced note for internal path namespaces**

Create `docs/python_api/advanced.rst` with this content. Keep `_PathNamespace` out of `docs/python_api/data.rst` and out of the common class autosummary:

```rst
========
Advanced
========

Data Internals
==============

``easyidp.data`` builds short demo-data attributes from JSON manifest keys. Dotted keys such as ``ms.project`` and ``ms.outputs.dom`` are expanded into runtime namespaces so users can write ``lotus.metashape.project`` or ``lotus.metashape.outputs.dom``.

The recursive namespace object is implemented as ``easyidp.data.dataset._PathNamespace``. It is an internal helper for advanced users and contributors who need to understand manifest parsing. It is intentionally not exported from ``easyidp.data`` and should not be treated as a stable public API.
```

- [ ] **Step 4: Remove obsolete autodoc references**

Remove references to old functions/classes from data docs:

```text
easyidp.data.download_all
easyidp.data.user_data_dir
easyidp.data.show_data_dir
easyidp.data.url_checker
easyidp.data.EasyidpDataSet
```

- [ ] **Step 5: Add or regenerate new autodoc stubs**

Ensure these autodoc pages exist or are generated by the docs command:

```text
docs/python_api/autodoc/easyidp.data.Lotus.rst
docs/python_api/autodoc/easyidp.data.ForestBirds.rst
docs/python_api/autodoc/easyidp.data.TestData.rst
docs/python_api/autodoc/easyidp.data.list_datasets.rst
```

Do not add autodoc stubs for `PathNamespace` or `_PathNamespace`.

- [ ] **Step 6: Build docs**

Run: `uv run sphinx-build -b html docs docs/_build/html`

Expected: build completes without unresolved references to removed data APIs.

---

### Task 8: Run Verification and Prepare Merge Back

**Files:**

- No new files unless test failures require focused fixes.
- Test: config, data, docs, lint if available.

- [ ] **Step 1: Run focused tests**

Run: `uv run pytest tests/test_config.py tests/test_data.py -q`

Expected: all selected tests pass.

- [ ] **Step 2: Run broader tests**

Run: `uv run pytest -q`

Expected: pass, or data-dependent tests skip when official data is not downloaded.

- [ ] **Step 3: Run docs build**

Run: `uv run sphinx-build -b html docs docs/_build/html`

Expected: pass without references to removed `data` APIs.

- [ ] **Step 4: Inspect git diff**

Run: `git diff -- src/easyidp tests docs .agents/plans/20260619_data_dataset_v21_refactor.md`

Expected: diff only includes the simplified data implementation, related tests, and docs.

- [ ] **Step 5: Merge strategy**

After verification, merge branch `data-json-plan` back to `dev` with a normal non-force merge. Do not reset `dev` unless explicitly approved.

## Self-Review

- Spec coverage: the plan removes TestData duplication, removes aliases/registry/builtin Python specs, uses JSON manifests, removes `user_data_dir`, keeps config as pure dataclass/json, replaces Aliyun OSS with anonymous OpenXLab CDN downloads, adds manual gdown/OpenXLab smoke tests, and includes existing docs updates.
- Placeholder scan: no placeholder markers or unspecified implementation steps remain.
- Type consistency: public names are `Lotus`, `ForestBirds`, `TestData`, `Dataset`, and `list_datasets`; internal `_PathNamespace` is not exported from `easyidp.data`, is only documented in advanced notes, and can represent deeper dotted keys.
