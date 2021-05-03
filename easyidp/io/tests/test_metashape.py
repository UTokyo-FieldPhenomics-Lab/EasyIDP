import os
import easyidp
from easyidp.io import metashape


def test_split_project_path():
    test_data = "data/metashape/goya_test.psx"
    folder_path, project_name, ext = metashape._split_project_path(test_data)

    assert folder_path == "data/metashape"
    assert project_name == "goya_test"
    assert ext == ".psx"


def test_check_is_software():
    test_data = "data/metashape/goya_test.psx"
    test_full_path = easyidp.test_full_path(test_data)

    judge = metashape._check_is_software(test_full_path)

    assert judge == True

