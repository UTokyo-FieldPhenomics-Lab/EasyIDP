"""EasyIDP internal logger configuration.

This module configures a dedicated ``easyidp`` logger without mutating the
global/root logging settings from host applications.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final

from tqdm import tqdm


LOGGER_NAME: Final[str] = "easyidp"
LOG_MAX_BYTES: Final[int] = 100 * 1024 * 1024
LOG_BACKUP_COUNT: Final[int] = 3

_base_logger = logging.getLogger(LOGGER_NAME)

_STATE = {"configured": False}

BANNER: Final[str] = """
███████╗ █████╗ ███████╗██╗   ██╗██╗██████╗ ██████╗
██╔════╝██╔══██╗██╔════╝╚██╗ ██╔╝██║██╔══██╗██╔══██╗
█████╗  ███████║███████╗ ╚████╔╝ ██║██║  ██║██████╔╝
██╔══╝  ██╔══██║╚════██║  ╚██╔╝  ██║██║  ██║██╔═══╝
███████╗██║  ██║███████║   ██║   ██║██████╔╝██║
╚══════╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝╚═════╝ ╚═╝
"""


class BraceStyleAdapter(logging.LoggerAdapter):
    """Adapter that supports loguru-like ``{}`` message formatting.

    Notes
    -----
    Existing EasyIDP call sites use brace-style placeholders. Standard logging
    expects ``%`` style, so this adapter formats messages before dispatching.
    """

    _logging_kwargs: Final[set[str]] = {
        "exc_info",
        "stack_info",
        "stacklevel",
        "extra",
    }

    def log(self, level: int, msg: object, *args: object, **kwargs: object) -> None:
        """Format the message and emit one record.

        Parameters
        ----------
        level : int
            Logging level value.
        msg : object
            Message template.
        *args : object
            Positional formatting values.
        **kwargs : object
            Formatting values and logging control keywords.
        """
        if not self.isEnabledFor(level):
            return

        log_kwargs = self._extract_logging_kwargs(kwargs)
        rendered = self._render_message(msg, args, kwargs)
        self.logger.log(level, rendered, **log_kwargs)

    def success(self, msg: object, *args: object, **kwargs: object) -> None:
        """Compatibility helper mapping ``success`` to ``INFO`` level."""
        self.log(logging.INFO, msg, *args, **kwargs)

    def _extract_logging_kwargs(self, kwargs: dict[str, object]) -> dict[str, object]:
        """Extract standard logging keyword arguments."""
        output: dict[str, object] = {}
        for key in list(kwargs.keys()):
            if key in self._logging_kwargs:
                output[key] = kwargs.pop(key)
        return output

    def _render_message(
        self,
        msg: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> str:
        """Render message using brace-style first, percent-style second."""
        message = str(msg)
        if not args and not kwargs:
            return message

        try:
            return message.format(*args, **kwargs)
        except Exception:
            pass

        if args:
            try:
                return message % args
            except Exception:
                return f"{message} {' '.join(map(str, args))}"

        return message


logger = BraceStyleAdapter(_base_logger, {})


class DuplicateThrottleFilter(logging.Filter):
    """Filter duplicate and frequent messages.

    Parameters
    ----------
    cooldown : float, optional
        Minimum interval (seconds) for grouped repeated messages.

    Examples
    --------
    >>> msg_filter = DuplicateThrottleFilter(cooldown=1.0)
    >>> isinstance(msg_filter, logging.Filter)
    True
    """

    def __init__(self, cooldown: float = 2.0) -> None:
        super().__init__()
        self.cooldown = cooldown
        self._last_msg: str | None = None
        self._last_times: dict[str, float] = {}
        self.throttle_groups: dict[str, str] = {
            "Converted to affine mode": "affine_mode",
            "Reprojecting ROI": "roi_reproject",
            "GeoTiff successfully saved": "tiff_save",
        }

    def filter(self, record: logging.LogRecord) -> bool:
        """Return whether the log record should be emitted.

        Parameters
        ----------
        record : logging.LogRecord
            Input log record.

        Returns
        -------
        bool
            ``True`` to emit, ``False`` to drop.
        """
        message = record.getMessage()
        now = time.time()

        if message == self._last_msg:
            return False

        group = self._match_group(message)
        if group is not None:
            last_time = self._last_times.get(group, 0.0)
            if now - last_time < self.cooldown:
                return False
            self._last_times[group] = now

        self._last_msg = message
        return True

    def _match_group(self, message: str) -> str | None:
        """Match message to a throttle group name.

        Parameters
        ----------
        message : str
            Rendered log message text.

        Returns
        -------
        str | None
            Group name if matched, else ``None``.
        """
        for pattern, group in self.throttle_groups.items():
            if pattern in message:
                return group
        return None


class TqdmHandler(logging.Handler):
    """Logging handler that writes via ``tqdm.write``.

    Notes
    -----
    ``tqdm.write`` avoids breaking progress bar rendering in terminal output.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a single logging record.

        Parameters
        ----------
        record : logging.LogRecord
            Input log record.
        """
        try:
            text = self.format(record)
            tqdm.write(text)
        except Exception:
            self.handleError(record)


def _default_log_file() -> Path:
    """Build the default log file path.

    Returns
    -------
    pathlib.Path
        Absolute path to ``easyidp.log`` in OS-specific user data directory.
    """
    if sys.platform.startswith("win"):
        os_path = os.getenv("LOCALAPPDATA") or "~/.local/share"
    elif sys.platform.startswith("darwin"):
        os_path = "~/Library/Application Support"
    else:
        os_path = os.getenv("XDG_DATA_HOME") or "~/.local/share"

    data_dir = (Path(os_path) / "easyidp.data").expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "easyidp.log"


def _normalize_level(level: str | int) -> int:
    """Normalize user level input to logging numeric level.

    Parameters
    ----------
    level : str | int
        Log level name (e.g. ``"INFO"``) or logging integer.

    Returns
    -------
    int
        Standard logging level number.
    """
    if isinstance(level, int):
        return level
    return getattr(logging, level.upper(), logging.INFO)


def _build_formatter() -> logging.Formatter:
    """Create the standard easyidp formatter.

    Returns
    -------
    logging.Formatter
        Formatter shared by stream and file handlers.
    """
    pattern = (
        "%(levelname).1s %(asctime)s %(name)s.%(funcName)s:%(lineno)d: %(message)s"
    )
    return logging.Formatter(fmt=pattern, datefmt="%Y/%m/%d %H:%M:%S")


def _reset_handlers() -> None:
    """Detach and close all handlers from easyidp logger."""
    for handler in list(_base_logger.handlers):
        _base_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            continue


def setup_logger(
    level: str | int = "INFO",
    enable_file: bool = True,
    reset: bool = False,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure and return the easyidp logger.

    Parameters
    ----------
    level : str | int, optional
        Base logger level.
    enable_file : bool, optional
        Whether to enable rotating file output.
    reset : bool, optional
        Whether to clear existing easyidp handlers before configuration.
    log_file : str | pathlib.Path | None, optional
        Custom file path for file output. ``None`` uses the default path.

    Returns
    -------
    logging.Logger
        Configured ``easyidp`` logger instance.

    Examples
    --------
    >>> lg = setup_logger(level="DEBUG", enable_file=False, reset=True)
    >>> lg.name
    'easyidp'
    """
    if reset:
        _reset_handlers()
        _STATE["configured"] = False

    if _STATE["configured"]:
        _base_logger.setLevel(_normalize_level(level))
        return _base_logger

    _base_logger.setLevel(_normalize_level(level))
    _base_logger.propagate = False
    formatter = _build_formatter()

    stream_handler = TqdmHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(DuplicateThrottleFilter(cooldown=1.0))
    _base_logger.addHandler(stream_handler)

    if enable_file:
        target = (
            Path(log_file).expanduser() if log_file is not None else _default_log_file()
        )
        file_handler = RotatingFileHandler(
            filename=target,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        _base_logger.addHandler(file_handler)

    _STATE["configured"] = True
    return _base_logger


def _get_config():
    """Lazily import and return easyidp config singleton.

    Returns
    -------
    EasyIDPConfig or None
        Config singleton if importable, else ``None``.

    Notes
    -----
    Uses a deferred import to avoid introducing a package-level circular
    dependency from ``logger.py`` to ``easyidp.__init__``.
    """
    try:
        from easyidp.config import config
    except ImportError:
        return None
    return config


def init_easyidp_logger(version: str) -> None:
    """Initialize easyidp logger and emit startup diagnostics.

    Reads ``log_level`` and ``show_banner`` from ``idp.config`` when the
    config is available; falls back to ``"INFO"`` / ``True`` otherwise.

    Parameters
    ----------
    version : str
        EasyIDP package version string.

    Returns
    -------
    None
        This function configures logger side-effects only.

    Examples
    --------
    >>> init_easyidp_logger("2.0.2")
    """
    cfg = _get_config()
    log_level = cfg.get("log_level") if cfg is not None else "INFO"
    show_banner = cfg.get("show_banner") if cfg is not None else True

    enable_file = os.environ.get("IS_TESTING") != "True"
    setup_logger(level=log_level, enable_file=enable_file)

    if show_banner:
        logger.info(f"Welcome to use\n{BANNER}\nVersion: {version}")

    logger.debug(f"ENV: IS_TESTING = {os.environ.get('IS_TESTING')}")
