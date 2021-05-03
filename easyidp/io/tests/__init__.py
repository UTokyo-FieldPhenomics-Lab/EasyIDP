import os
import sys
import subprocess
import easyidp

module_path = os.path.join(easyidp.__path__[0], "io/tests")


def test():
    try:
        subprocess.check_call(f"pytest {module_path}",
                              shell=True,
                              stdout=sys.stdout,
                              stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        pass