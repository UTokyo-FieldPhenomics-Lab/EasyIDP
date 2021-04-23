import os
import easyidp
from easyidp.core.softwares import MetaShape, Pix4D


def test_metashape_open_psx_project():
    test_data = "data/metashape/goya_test.psx"
    test_full_path = os.path.join(easyidp.__path__[0], test_data)

    doc = MetaShape()
    doc.open(test_full_path)

    assert doc.project_name == "goya_test"

