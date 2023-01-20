import shutil
from pathlib import Path

import easyidp as idp

def test_get_download_url_without_download():

    assert idp.data.Lotus.name == '2017_tanashi_lotus'
    assert idp.data.TestData.name == 'data_for_tests'

    url = idp.data.Lotus.url_list

    assert isinstance(url, list)
    assert isinstance(url[1], str)

def test_usr_data_dir():
    root_dir = idp.data.user_data_dir("")
    assert "easyidp.data" in str(root_dir)

def test_gdown():
    gd_dir = idp.data.user_data_dir("gdown_test")
    
    if gd_dir.exists():
        shutil.rmtree(gd_dir)
    
    gd = idp.data.GDownTest()

    assert gd.data_dir.exists()
    assert (gd.data_dir / "file1.txt").exists()

    assert Path(gd.pix4d.proj).resolve() == (gd.data_dir / "file1.txt").resolve()
    assert Path(gd.metashape.param).resolve() == (gd.data_dir / "folder1").resolve()

    # test remove and reload
    gd.remove_data()
    assert not gd.data_dir.exists()

    gd.reload_data()
    assert gd.data_dir.exists()