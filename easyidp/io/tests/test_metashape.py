import pytest
import os
from easyidp.io import metashape
from easyidp.io.tests import module_path

test_data = "data/goya_test.psx"
test_full_path = os.path.join(module_path, "data/goya_test.psx")

def test_open_psx_test():
    doc = metashape.Document()
    doc.open(test_full_path)

    assert doc.project_name == "goya_test"

