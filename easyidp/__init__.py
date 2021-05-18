from easyidp import (
    core, io
)

from easyidp.core.objects import (
    Calibration,
    MetashapeChunkTransform,
    GeoTiff,
    Image,
    Photo,
    ReconsProject,
    ROI,
    Sensor,
    ShapeFile,
    PointCloud,
    Points
)

import os
import sys
import subprocess


def test():
    try:
        subprocess.check_call(f"pytest {__path__[0]} -s",
                              shell=True,
                              stdout=sys.stdout,
                              stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        # print(f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}")
        pass


def test_full_path(path2data_folder):
    return os.path.join(__path__[0], path2data_folder)