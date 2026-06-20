import easyidp as idp


def test_constructor_does_not_create_cache_dirs(tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)

    assert not lotus.root.exists()
    assert not lotus._archive_path().parent.exists()


def test_is_ready_uses_ready_check_keys_only(tmp_path):
    lotus = idp.data.Lotus(cache_root=tmp_path, notify_missing=False)
    for key in ("shp", "metashape.project", "pix4d.dom", "pix4d.dsm"):
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
    assert isinstance(plan["description"], str)
    assert "plots.shp" in plan["missing"]
    assert "archive" not in plan
    assert "mirrors" not in plan


def test_data_root_comes_from_config(tmp_path, monkeypatch):
    from types import SimpleNamespace

    fake = SimpleNamespace(data_dir=tmp_path / "configured")
    monkeypatch.setattr(idp.config, "get", lambda key: getattr(fake, key))
    lotus = idp.data.Lotus(notify_missing=False)
    assert lotus.root == tmp_path / "configured" / "2017_tanashi_lotus"
