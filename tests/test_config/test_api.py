import pytest

import easyidp as idp
from easyidp.config import EasyIDPConfig


def test_default_config_uses_default_data_dir(tmp_path):
    config = EasyIDPConfig(config_path=tmp_path / "config.json")
    assert config.get("data_dir").name == "easyidp.data"
    assert config.get("log_level") == "INFO"
    assert config.get("show_banner") is True


def test_package_exports_small_config_entrypoint():
    assert hasattr(idp, "config")
    assert hasattr(idp.config, "get")
    assert hasattr(idp.config, "set")
    assert hasattr(idp.config, "reset")
    assert not hasattr(idp.config, "save")
    assert not hasattr(idp.config, "update")


def test_get_unknown_key_raises_keyerror(tmp_path):
    config = EasyIDPConfig(config_path=tmp_path / "config.json")
    with pytest.raises(KeyError, match="Unknown EasyIDP config key: bad_key"):
        config.get("bad_key")


def test_set_unknown_key_raises_keyerror(tmp_path):
    config = EasyIDPConfig(config_path=tmp_path / "config.json")
    with pytest.raises(KeyError, match="Unknown EasyIDP config key: bad_key"):
        config.set(bad_key="anything")


def test_user_data_dir_removed_from_package_root():
    assert not hasattr(idp, "user_data_dir")
    assert hasattr(idp, "config")
    assert idp.config.get("data_dir").name == "easyidp.data"
