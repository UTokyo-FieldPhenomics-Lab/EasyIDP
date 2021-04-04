from easyidp import (
    io
)

import os
import sys
import subprocess
import easyidp


def test():
    subprocess.check_call(f"pytest {easyidp.__path__[0]}",
                          shell=True,
                          stdout=sys.stdout,
                          stderr=subprocess.STDOUT)