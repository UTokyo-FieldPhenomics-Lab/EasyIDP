import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def default_config_path() -> Path:
    """Return the default EasyIDP JSON config path.

    Returns
    -------
    pathlib.Path
        Platform-specific config file path.

    Examples
    --------
    >>> default_config_path().name
    'config.json'

    Notes
    -----
    This function only computes a path and does not create files.
    """
    if sys.platform.startswith("win"):
        root = Path.home() / "AppData" / "Roaming"
    elif sys.platform.startswith("darwin"):
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path.home() / ".config"
    return root / "easyidp" / "config.json"


def default_data_dir() -> Path:
    """Return the default EasyIDP dataset directory.

    Returns
    -------
    pathlib.Path
        Platform-specific data root path.

    Examples
    --------
    >>> default_data_dir().name
    'easyidp.data'
    """
    if sys.platform.startswith("win"):
        root = Path.home() / "AppData" / "Local"
    elif sys.platform.startswith("darwin"):
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path.home() / ".local" / "share"
    return root / "easyidp.data"


@dataclass
class EasyIDPConfig:
    """JSON-backed package configuration.

    Parameters
    ----------
    config_path : pathlib.Path, optional
        JSON config file path. Defaults to the platform user config path.

    Returns
    -------
    EasyIDPConfig
        Mutable session configuration object.

    Examples
    --------
    >>> cfg = EasyIDPConfig()
    >>> cfg.update(log_level="DEBUG")
    >>> cfg.log_level
    'DEBUG'

    Notes
    -----
    Loading may happen at import time, but saving requires an explicit call.
    """
    config_path: Path = field(default_factory=default_config_path)
    data_dir: Path = field(default_factory=default_data_dir)
    log_level: str = "INFO"
    show_banner: bool = True

    def __post_init__(self) -> None:
        self.config_path = self.config_path.expanduser()
        self._load_if_exists()

    def get(self) -> "EasyIDPConfig":
        return self

    def update(self, **kwargs: Any) -> "EasyIDPConfig":
        for key, value in kwargs.items():
            if key == "data_dir":
                self.data_dir = Path(value).expanduser()
                continue
            if key in {"log_level", "show_banner"}:
                setattr(self, key, value)
                continue
            raise KeyError(f"Unknown EasyIDP config key: {key}")
        return self

    def save(self) -> Path:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return self.config_path

    def reset(self, save: bool = False) -> "EasyIDPConfig":
        self.data_dir = default_data_dir()
        self.log_level = "INFO"
        self.show_banner = True
        if save:
            self.save()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "log_level": self.log_level,
            "show_banner": self.show_banner,
        }

    def _load_if_exists(self) -> None:
        if not self.config_path.exists():
            return
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.update(**data)


config = EasyIDPConfig()
get = config.get
update = config.update
save = config.save
reset = config.reset
