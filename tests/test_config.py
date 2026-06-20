import pytest

import easyidp as idp
from easyidp.config import EasyIDPConfig


def test_default_config_uses_default_data_dir(tmp_path):
    config = EasyIDPConfig(config_path=tmp_path / "config.json")

    assert config.data_dir.name == "easyidp.data"
    assert config.log_level == "INFO"
    assert config.show_banner is True


def test_update_changes_session_without_saving(tmp_path):
    config_path = tmp_path / "config.json"
    config = EasyIDPConfig(config_path=config_path)

    config.update(data_dir=tmp_path / "data", log_level="DEBUG", show_banner=False)

    assert config.data_dir == tmp_path / "data"
    assert config.log_level == "DEBUG"
    assert config.show_banner is False
    assert not config_path.exists()


def test_save_and_reload_json_config(tmp_path):
    config_path = tmp_path / "config.json"
    config = EasyIDPConfig(config_path=config_path)
    config.update(data_dir=tmp_path / "data", log_level="WARNING", show_banner=False)
    config.save()

    loaded = EasyIDPConfig(config_path=config_path)

    assert loaded.data_dir == tmp_path / "data"
    assert loaded.log_level == "WARNING"
    assert loaded.show_banner is False


def test_package_exports_config_entrypoint():
    assert hasattr(idp, "config")
    assert hasattr(idp.config, "get")
    assert hasattr(idp.config, "update")
    assert hasattr(idp.config, "save")
    assert hasattr(idp.config, "reset")


def test_config_can_store_log_level_and_banner(tmp_path):
    config = EasyIDPConfig(config_path=tmp_path / "config.json")
    config.update(log_level="ERROR", show_banner=False)
    config.save()

    loaded = EasyIDPConfig(config_path=tmp_path / "config.json")

    assert loaded.log_level == "ERROR"
    assert loaded.show_banner is False


def test_update_unknown_key_raises_keyerror(tmp_path):
    config = EasyIDPConfig(config_path=tmp_path / "config.json")

    with pytest.raises(KeyError, match="Unknown EasyIDP config key: bad_key"):
        config.update(bad_key="anything")


def test_reset_restores_defaults_without_saving(tmp_path):
    config_path = tmp_path / "config.json"
    config = EasyIDPConfig(config_path=config_path)
    config.update(data_dir=tmp_path / "data", log_level="DEBUG", show_banner=False)

    config.reset()

    assert config.data_dir.name == "easyidp.data"
    assert config.log_level == "INFO"
    assert config.show_banner is True
    assert not config_path.exists()


def test_reset_save_writes_defaults_and_can_reload(tmp_path):
    config_path = tmp_path / "config.json"
    config = EasyIDPConfig(config_path=config_path)
    config.update(data_dir=tmp_path / "data", log_level="DEBUG", show_banner=False)
    config.save()

    config.reset(save=True)

    assert config_path.exists()
    loaded = EasyIDPConfig(config_path=config_path)
    assert loaded.data_dir.name == "easyidp.data"
    assert loaded.log_level == "INFO"
    assert loaded.show_banner is True


def test_save_returns_config_path(tmp_path):
    config_path = tmp_path / "config.json"
    config = EasyIDPConfig(config_path=config_path)

    returned = config.save()

    assert returned == config_path.resolve()


def test_user_data_dir_removed_from_package_root():
    assert not hasattr(idp, "user_data_dir")
    assert hasattr(idp, "config")
    assert idp.config.get().data_dir.name == "easyidp.data"
