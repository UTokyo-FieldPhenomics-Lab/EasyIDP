__version__ = "2.0.2"

import os
import sys
import subprocess
import warnings
from pathlib import Path
from copy import deepcopy

import time
import numpy as np
from tqdm import tqdm
from loguru import logger

##############
# dict tools #
##############

from .structures import Container


def _find_key(mydict, value):
    """a simple function to using dict value to find key
    e.g.
    >>> mydict = {"a": 233, "b": 456}
    >>> _find_key(mydict, 233)
    "a"
    """
    key_idx = list(mydict.values()).index(value)
    return list(mydict.keys())[key_idx]


##############
# path tools #
##############


def get_full_path(short_path):
    if isinstance(short_path, str):
        return Path(short_path)
    elif isinstance(short_path, Path):
        return short_path
    else:
        return None


def parse_relative_path(root_path, relative_path):
    # for metashape frame.zip path use only
    if r"../../" in relative_path:
        frame_path = os.path.dirname(os.path.abspath(root_path))
        merge = os.path.join(frame_path, relative_path)
        return os.path.abspath(merge)
    else:
        logger.warning(f"Seems it is an absolute path [{relative_path}]")
        return relative_path


def user_data_dir(file_name=""):
    r"""Get OS specific data directory path for EasyIDP.

    Parameters
    ----------
    file_name : str
        file to be fetched from the data dir

    Returns
    -------
    str
        full path to the user-specific data dir

    Notes
    -----
    Typical user data directories are:

    .. code-block:: text

        macOS:    ~/Library/Application Support/easyidp.data
        Unix:     ~/.local/share/easyidp.data   # or in $XDG_DATA_HOME, if defined
        Win 10:   C:\Users\<username>\AppData\Local\easyidp.data

    For Unix, we follow the XDG spec and support ``$XDG_DATA_HOME`` if defined.

    Referenced from stackoverflow [1]_ then get github [2]_ .

    References
    ----------
    .. [1] Python: Getting AppData folder in a cross-platform way https://stackoverflow.com/questions/19078969/python-getting-appdata-folder-in-a-cross-platform-way
    .. [2] SwagLyrics-For-Spotify/swaglyrics/__init__.py https://github.com/SwagLyrics/SwagLyrics-For-Spotify/blob/master/swaglyrics/__init__.py#L8-L32

    """
    # get os specific path
    if sys.platform.startswith("win"):
        os_path = os.getenv("LOCALAPPDATA")
    elif sys.platform.startswith("darwin"):
        os_path = "~/Library/Application Support"
    else:
        # linux
        os_path = os.getenv("XDG_DATA_HOME", "~/.local/share")

    # join with easyidp.data dir
    path = Path(os_path) / "easyidp.data"

    add_usr = path.expanduser()

    if not os.path.exists(str(add_usr)):
        os.makedirs(str(add_usr))

    return add_usr / file_name


################
# logger tools #
################

# Generated in ANSI Shadow by:
# https://www.asciiart.eu/text-to-ascii-art

banner = """
███████╗ █████╗ ███████╗██╗   ██╗██╗██████╗ ██████╗ 
██╔════╝██╔══██╗██╔════╝╚██╗ ██╔╝██║██╔══██╗██╔══██╗
█████╗  ███████║███████╗ ╚████╔╝ ██║██║  ██║██████╔╝
██╔══╝  ██╔══██║╚════██║  ╚██╔╝  ██║██║  ██║██╔═══╝ 
███████╗██║  ██║███████║   ██║   ██║██████╔╝██║     
╚══════╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝╚═════╝ ╚═╝     
"""

logger_format = (
    "<level>{level:1.1}</level> "  # E for ERROR, I for INFO, etc.
    "<green>{time:YYYY/MM/DD HH:mm:ss}</green> "  # YYYY/MM/DD HH:mm:ss
    "{name}.{function}:{line}: "  # file.py:123
    "<level>{message}</level>"  # The actual log message
)

logger_file = user_data_dir() / "easyidp.log"


# Filter logic for duplicates and frequency limiting
class LogFilter:
    """Filter to handle duplicates and throttling of frequent messages"""

    def __init__(self, cooldown=2.0):
        self.cooldown = cooldown
        self._last_msg = None
        self._last_times = {}  # {group_key: timestamp}

        # Patterns to group and throttle
        # Key: substring to match, Value: group name
        self.throttle_groups = {
            "Converted to affine mode": "affine_mode",
            "Reprojecting ROI": "roi_reproject",
            "GeoTiff successfully saved": "tiff_save",
        }

    def __call__(self, record):
        msg = record["message"]
        now = time.time()

        # 1. Block exact consecutive duplicates
        if msg == self._last_msg:
            return False

        # 2. Check throttling groups
        matched_group = None
        for pattern, group in self.throttle_groups.items():
            if pattern in msg:
                matched_group = group
                break

        if matched_group:
            last_time = self._last_times.get(matched_group, 0)
            if now - last_time < self.cooldown:
                return False
            self._last_times[matched_group] = now

        # Update last message and allow
        self._last_msg = msg
        return True


# Sink to redirect logs to tqdm.write to avoid interfering with progress bars
def tqdm_sink(message):
    tqdm.write(message, end="")


# 1. Remove all default handlers
logger.remove()

# 2. Add tqdm sink with custom filter
logger.add(
    tqdm_sink, level="INFO", format=logger_format, filter=LogFilter(cooldown=1.0)
)

if not os.environ.get("IS_TESTING") == "True":
    # 3. 你也可以添加一个文件处理器，将日志同时保存到文件
    # 为解决vscode的test模块也会输出日志，使用环境变量进行区分
    logger.add(
        logger_file,
        level="DEBUG",  # 文件中记录更详细的 DEBUG 级别日志
        rotation="100 MB",  # 每 10 MB 切割一个新文件
        format=logger_format,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

logger.info(f"Welcome to use\n{banner}\nVersion: {__version__}")

logger.debug(f"ENV: IS_TESTING = {os.environ.get('IS_TESTING')}")


def logged_input(prompt: str, is_sensitive: bool = False) -> str:
    """
    一个包装了 loguru 日志记录功能的 input() 函数。
    Args:
        prompt (str): 显示给用户的提示信息。
        is_sensitive (bool): 如果为 True，用户的输入将被屏蔽，不会记录到日志中。
    Returns:
        str: 用户输入的字符串。
    """
    # 1. 记录提示信息
    logger.info(f"向用户显示输入提示: '{prompt}'")

    # 2. 调用原始的 input() 函数
    user_response = input(prompt)

    # 3. 记录用户的输入 (处理敏感信息)
    if is_sensitive:
        logger.info("用户输入了敏感信息 [内容已屏蔽]")
    else:
        logger.info(f"用户输入内容: '{user_response}'")

    return user_response


###############
# import APIs #
###############

from . import (
    visualize,
    cvtools,
    geotools,
    shp,
    jsonfile,
    reconstruct,
    data,
)

from .pointcloud import PointCloud
from .geotiff import GeoTiff
from .pix4d import Pix4D
from .metashape import Metashape
from .reconstruct import ProjectPool
from .roi import ROI


########################
# Dataset region check #
########################

aliyun_down = None
GOOGLE_AVAILABLE = True

if not data._can_access_google_cloud():
    GOOGLE_AVAILABLE = False

    try:
        import oss2
    except ImportError:
        logger.info("oss2 is not installed. Installing now...")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "oss2",
                "-i",
                "https://pypi.tuna.tsinghua.edu.cn/simple",
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to install oss2. pip exited with status {result.returncode}"
            )
        logger.info("oss2 has been installed.")

        try:
            import oss2
        except ImportError:
            raise ImportError(
                "Failed to import oss2 after installation, please manually install `oss2` package by:\n"
                "pip install oss2 -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )
