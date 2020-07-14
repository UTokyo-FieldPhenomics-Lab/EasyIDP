import numpy as np
from easyric.objects import Pix4D
from easyric.calculate import raw2raw

def test_get_another_photo_pixel():
    origin = 'DJI_0209.JPG'
    target = 'DJI_0210.JPG'

    # this data path is "Data" in previous file tree
    data_path = f"D:/OneDrive/Documents/4_PhD/10_Data/02_UAV.Broccoli.Pix4D/2019tanashi_broccoli5_AP_new/20191008"
    # this project_name is "PROJECTNAME" in previous file tree
    project_name = f"broccoli_tanashi_5_20191008_mavicRGB_15m_M"
    raw_img_folder = f'{data_path}/photos'

    proj = Pix4D(project_path=data_path, raw_img_path=raw_img_folder, project_name=project_name,
                 param_folder=data_path)

    xu1 = 1025
    yu1 = 3099

    out_coord = raw2raw.get_another_photo_pixel(proj, origin, target, xu1, yu1, -50)

    assert out_coord == (1020.9574367156933, 2918.9854500688016)
