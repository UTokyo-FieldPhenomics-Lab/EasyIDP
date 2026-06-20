import json
import os
import sys
import tempfile
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


KNOWN_KEYS = {"data_dir", "log_level", "show_banner"}


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
    >>> cfg.set(log_level="DEBUG")
    >>> cfg.log_level
    'DEBUG'

    Notes
    -----
    ``set()`` and ``reset()`` persist JSON immediately.
    ``get()`` re-reads the file on each call so manual edits are visible.
    """
    config_path: Path = field(default_factory=default_config_path)
    data_dir: Path = field(default_factory=default_data_dir)
    log_level: str = "INFO"
    show_banner: bool = True

    def __post_init__(self) -> None:
        self.config_path = self.config_path.expanduser()
        self._load_if_exists()

    def get(self, key: str | None = None) -> Any:
        """Return current config value(s).

        Parameters
        ----------
        key : str or None, optional
            Config key name. ``None`` returns a plain dict snapshot.

        Returns
        -------
        Any
            Value for *key*, or a ``dict`` with all known settings.

        Raises
        ------
        KeyError
            If *key* is not a recognised config key.
        """
        self._load_if_exists()
        if key is None:
            return self.to_dict()
        if key not in KNOWN_KEYS:
            raise KeyError(f"Unknown EasyIDP config key: {key}")
        if key == "data_dir":
            return self.data_dir
        return getattr(self, key)

    def set(self, **kwargs: Any) -> "EasyIDPConfig":
        """Update config values and persist JSON immediately.

        Parameters
        ----------
        **kwargs : Any
            One or more recognised config keys with new values.

        Returns
        -------
        EasyIDPConfig
            Self (fluent API).

        Raises
        ------
        KeyError
            If any key in *kwargs* is not recognised.
        """
        self._apply(kwargs)
        self._save()
        return self

    def reset(self) -> "EasyIDPConfig":
        """Restore factory defaults and persist JSON immediately.

        Returns
        -------
        EasyIDPConfig
            Self (fluent API).
        """
        self.data_dir = default_data_dir()
        self.log_level = "INFO"
        self.show_banner = True
        self._save()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "log_level": self.log_level,
            "show_banner": self.show_banner,
        }

    def _apply(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if key not in KNOWN_KEYS:
                raise KeyError(f"Unknown EasyIDP config key: {key}")
            if key == "data_dir":
                self.data_dir = Path(value).expanduser()
            else:
                setattr(self, key, value)

    def _save(self) -> Path:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix="config-",
            dir=str(self.config_path.parent),
            text=True,
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=2)
                fh.write("\n")
            os.replace(tmp_path, str(self.config_path))
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return self.config_path

    def _load_if_exists(self) -> None:
        if not self.config_path.exists():
            return
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        self._apply(data)

config = EasyIDPConfig()
get = config.get
set = config.set
reset = config.reset
