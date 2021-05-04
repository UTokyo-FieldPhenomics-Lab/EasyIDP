import os
import sys
import subprocess
import easyidp

module_path = os.path.join(easyidp.__path__[0], "core/tests")


def test():
    try:
        subprocess.check_call(f"pytest {module_path} -s",
                              shell=True,
                              stdout=sys.stdout,
                              stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        pass