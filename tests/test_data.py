import os
import shutil
import pytest

import easyidp as idp

def test_usr_data_dir():
    root_dir = idp.data.user_data_dir("")
    assert "easyidp.data" in str(root_dir)

def test_gdown():
    gd_dir = idp.data.user_data_dir("gdown_test")
    
    if os.path.exists(gd_dir):
        shutil.rmtree(gd_dir)
    
    gd = idp.data.GDownTest()

    assert os.path.exists(gd.data_dir)
    assert os.path.exists(gd.data_dir / "file1.txt")

    assert gd.pix4d.proj == os.path.join(str(gd.data_dir), "file1.txt")
    assert gd.pix4d.param == os.path.join(str(gd.data_dir), "folder1")
