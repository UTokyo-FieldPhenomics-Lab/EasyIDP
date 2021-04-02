import pytest
import os
from easyidp.io import metashape
from easyidp.io.tests import module_path

def test_open_psx_test():
    test_path = os.path.join(module_path, "data/goya_test.psx")
    doc = metashape.Document()
    doc.open(test_path)

    assert doc.project_name == "goya_test"

