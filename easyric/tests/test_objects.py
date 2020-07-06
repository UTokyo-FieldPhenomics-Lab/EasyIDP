from easyric.objects import Pix4D


def test_pix4d_objects_init_origin():
    test_folder0 = r'file\pix4d.origin'
    origin = Pix4D(test_folder0, raw_img_path='file\pix4d.origin\photos')

    assert origin.project_name == 'pix4d.origin'
    assert origin.project_path == 'D:/OneDrive/Program/GitHub/EasyRIC/easyric/tests/file/pix4d.origin'

    assert origin.xyz_file == 'D:/OneDrive/Program/GitHub/EasyRIC/easyric/tests/file/pix4d.origin/1_initial/params/pix4d.origin_offset.xyz'
    assert origin.ccp_file == 'D:/OneDrive/Program/GitHub/EasyRIC/easyric/tests/file/pix4d.origin/1_initial/params/pix4d.origin_calibrated_camera_parameters.txt'
    assert origin.dom_file == 'D:/OneDrive/Program/GitHub/EasyRIC/easyric/tests/file/pix4d.origin/3_dsm_ortho/2_mosaic/pix4d.origin_mosaic_group1.empty.tif'


def test_pix4d_objects_init_diy(capsys):
    test_folder = r'file\pix4d.diy'

    diy = Pix4D(test_folder, raw_img_path=r'../../easyric/tests/file/pix4d.diy/photos',
                param_folder=r'file\pix4d.diy\params',
                project_name='test', dsm_path=r'tests\file\pix4d.origin\3_dsm_ortho\1_dsm\pix4d.origin_empty.tif')

    captured = capsys.readouterr()
    assert captured.out == "[Init][Pix4D] No ply given, however find 'test_group1_densified_point_cloud.ply' at current project folder\n" \
                           "[Init][Pix4D] No dom given, however find 'test_transparent_mosaic_group1.tif' at current project folder\n"

    assert diy.dsm_file == 'D:/OneDrive/Program/GitHub/EasyRIC/easyric/tests/tests/file/pix4d.origin/3_dsm_ortho/1_dsm/pix4d.origin_empty.tif'
    assert diy.xyz_file == 'D:/OneDrive/Program/GitHub/EasyRIC/easyric/tests/file/pix4d.diy/params/test_offset.xyz'
    assert diy.dom_file == 'D:/OneDrive/Program/GitHub/EasyRIC/easyric/tests/file/pix4d.diy/test_transparent_mosaic_group1.tif'