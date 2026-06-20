__version__ = "2.1.0"

import os
from pathlib import Path

from . import config as config
from .logger import init_easyidp_logger, logger, setup_logger

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


################
# logger tools #
################

init_easyidp_logger(__version__)


def logged_input(prompt: str, is_sensitive: bool = False) -> str:
    """
    一个包装了 logging 日志记录功能的 input() 函数。
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

