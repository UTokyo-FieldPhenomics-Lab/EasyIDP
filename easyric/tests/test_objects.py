import unittest
import os
import sys
from easyric.objects.software import Pix4dFiles

class TestSoftware(unittest.TestCase):

    def test_pix4d_files(self):
        print('============ test_pix4d_files =============')
        print('Current work directory and folder: ', os.listdir('.'))
        test_folder0 = r'file\pix4d.origin'

        origin = Pix4dFiles(test_folder0, raw_img_path='file\pix4d.diy\photos')

        self.assertEqual(origin.project_name, 'pix4d.origin')
        print('full path: ', origin.project_path)

        print(origin.xyz_file)
        print(origin.ccp_file)
        print(origin.dom_file)

        print('-'*20)
        test_folder1 = r'file\pix4d.diy'

        diy = Pix4dFiles(test_folder1, raw_img_path=r'file\pix4d.diy\photos')
        diy.manual_specify(param_folder=r'file\pix4d.diy\params',
                           dsm_path=r'tests\file\pix4d.origin\3_dsm_ortho\1_dsm\pix4d.origin_empty.tif')

        print(diy.dsm_file)
        print(diy.xyz_file)
        print(diy.dom_file)


if __name__ == '__main__':
    unittest.main()
