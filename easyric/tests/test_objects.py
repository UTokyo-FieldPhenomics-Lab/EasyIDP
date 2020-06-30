import unittest
import os
from easyric.objects import Pix4D

class TestSoftware(unittest.TestCase):

    def test_pix4d_objects_init_origin(self):
        print('============ test_pix4d_files =============')
        print('Current work directory and folder: ', os.listdir('.'))
        test_folder0 = r'file\pix4d.origin'

        origin = Pix4D(test_folder0, raw_img_path='file\pix4d.origin\photos')

        self.assertEqual(origin.project_name, 'pix4d.origin')
        self.assertEqual(origin.project_path, 'D:/OneDrive/Program/GitHub/EasyRIC/easyric/tests/file/pix4d.origin')

        self.assertEqual(origin.xyz_file, 'D:/OneDrive/Program/GitHub/EasyRIC/easyric/tests/file/pix4d.origin/1_initial/params/pix4d.origin_offset.xyz')
        self.assertEqual(origin.ccp_file, 'D:/OneDrive/Program/GitHub/EasyRIC/easyric/tests/file/pix4d.origin/1_initial/params/pix4d.origin_calibrated_camera_parameters.txt')
        self.assertEqual(origin.dom_file, 'D:/OneDrive/Program/GitHub/EasyRIC/easyric/tests/file/pix4d.origin/3_dsm_ortho/2_mosaic/pix4d.origin_mosaic_group1.empty.tif')

    def test_pix4d_objects_init_diy(self):
        test_folder = r'file\pix4d.diy'

        diy = Pix4D(test_folder, raw_img_path=r'file\pix4d.diy\photos', param_folder=r'file\pix4d.diy\params',
                    project_name='test', dsm_path=r'tests\file\pix4d.origin\3_dsm_ortho\1_dsm\pix4d.origin_empty.tif')

        self.assertEqual(diy.dsm_file, 'D:/OneDrive/Program/GitHub/EasyRIC/easyric/tests/tests/file/pix4d.origin/3_dsm_ortho/1_dsm/pix4d.origin_empty.tif')
        self.assertEqual(diy.xyz_file, 'D:/OneDrive/Program/GitHub/EasyRIC/easyric/tests/file/pix4d.diy/params/test_offset.xyz')
        self.assertEqual(diy.dom_file, 'D:/OneDrive/Program/GitHub/EasyRIC/easyric/tests/file/pix4d.diy/test_transparent_mosaic_group1.tif')


if __name__ == '__main__':
    unittest.main()
