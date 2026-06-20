import json

import pytest

from easyidp.config import EasyIDPConfig


def test_set_writes_json_immediately(tmp_path):
    config_path = tmp_path / "config.json"
    config = EasyIDPConfig(config_path=config_path)

    config.set(data_dir=tmp_path / "data", log_level="DEBUG", show_banner=False)

    assert config_path.exists()
    loaded = EasyIDPConfig(config_path=config_path)
    assert loaded.get("data_dir") == tmp_path / "data"
    assert loaded.get("log_level") == "DEBUG"
    assert loaded.get("show_banner") is False


def test_get_reloads_manual_json_edits(tmp_path):
    config_path = tmp_path / "config.json"
    config = EasyIDPConfig(config_path=config_path)
    config.set(data_dir=tmp_path / "old")

    config_path.write_text(
        json.dumps({
            "data_dir": str(tmp_path / "manual"),
            "log_level": "WARNING",
            "show_banner": False,
        }),
        encoding="utf-8",
    )

    assert config.get("data_dir") == tmp_path / "manual"
    assert config.get("log_level") == "WARNING"
    assert config.get("show_banner") is False


def test_get_without_key_returns_plain_snapshot(tmp_path):
    config = EasyIDPConfig(config_path=tmp_path / "config.json")
    config.set(data_dir=tmp_path / "data", show_banner=False)

    snapshot = config.get()

    assert snapshot == {
        "data_dir": str(tmp_path / "data"),
        "log_level": "INFO",
        "show_banner": False,
    }


def test_reset_writes_defaults_immediately(tmp_path):
    config_path = tmp_path / "config.json"
    config = EasyIDPConfig(config_path=config_path)
    config.set(data_dir=tmp_path / "data", log_level="DEBUG", show_banner=False)

    config.reset()

    loaded = EasyIDPConfig(config_path=config_path)
    assert loaded.get("data_dir").name == "easyidp.data"
    assert loaded.get("log_level") == "INFO"
    assert loaded.get("show_banner") is True
