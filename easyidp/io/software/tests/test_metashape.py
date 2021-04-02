import pytest
from easyidp.io.software import metashape

def test_open_psx_test():
    test_path = "data/goya_test.psx"
    doc = metashape.Document()
    doc.open(test_path)

    assert doc.project_name == "goya_test"

