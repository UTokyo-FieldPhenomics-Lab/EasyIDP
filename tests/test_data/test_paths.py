import easyidp as idp


def test_lotus_paths_are_short_namespaces(tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)

    assert lotus.name == "lotus"
    assert lotus.root == tmp_path / "2017_tanashi_lotus"
    assert lotus._archive_path() == tmp_path / ".downloads" / "2017_tanashi_lotus.zip"
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


def test_public_api_is_small():
    expected = {"Lotus", "ForestBirds", "TestData", "list_datasets"}
    forbidden = {"DatasetRegistry", "registry", "user_data_dir", "PathNamespace"}

    missing = [n for n in expected if not hasattr(idp.data, n)]
    present = [n for n in forbidden if hasattr(idp.data, n)]

    assert not missing, f"Expected exports missing: {missing}"
    assert not present, f"Forbidden exports found: {present}"
