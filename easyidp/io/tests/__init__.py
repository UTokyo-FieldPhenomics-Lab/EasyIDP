import os
import easyidp

module_path = os.path.join(easyidp.__path__[0], "io/tests")

def test():
    os.system(f"pytest {module_path}")