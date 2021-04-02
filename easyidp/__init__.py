from easyidp import (
    io
)

import os
import easyidp


def test():
    os.system(f"pytest {easyidp.__path__[0]}")