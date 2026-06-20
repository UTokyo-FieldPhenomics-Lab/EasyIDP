import easyidp as idp


def test_dataset_repr_shows_missing_status(tmp_path):
    birds = idp.data.ForestBirds(cache_root=tmp_path, notify_missing=False)
    text = repr(birds)
    assert object.__repr__(birds) in text
    assert "Official EasyIDP forest birds demo dataset from Florida." in text
    assert "Size: 1.97 GB" in text
    assert "Status: not downloaded. call .download() to save at" in text
    assert str(tmp_path / "2022_florida_forestbirds") in text
    assert 'idp.config.set(data_dir="/path/to/easyidp.data")' in text


def test_dataset_repr_shows_available_status(tmp_path):
    birds = idp.data.ForestBirds(cache_root=tmp_path, notify_missing=False)
    for key in ("shp", "metashape.project", "metashape.dom", "metashape.dsm"):
        path = birds.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    text = repr(birds)
    assert object.__repr__(birds) in text
    assert "Size: 1.97 GB" in text
    assert "Status: available at" in text
    assert str(birds.root) in text
    assert "not downloaded" not in text
