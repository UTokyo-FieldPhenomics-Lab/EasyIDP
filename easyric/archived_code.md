# debug_for_test.py
example 1
```python
import pyproj
from easyric.io.shp import read_shp3d

lonlat_z = read_shp3d(r'../easyric/tests/file/pix4d.diy/plots.shp',
                      r'../easyric/tests/file/pix4d.diy/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif',
                      get_z_by='local', given_proj=pyproj.CRS.from_epsg(4326))
print(lonlat_z)
```

example 2
```python
import numpy as np
from easyric.io import geotiff
from skimage.io import imread, imshow, show
from skimage.color import rgb2gray

photo_path = '../easyric/tests/file/pix4d.diy/photos/DJI_0174.JPG'
roi = np.asarray([[2251, 1223], [2270, 1270], [2227, 1263], [2251, 1223]])
#imarray = rgb2gray(imread(photo_path))
imarray = imread(photo_path)

im_out, offsets = geotiff._imarray_clip(imarray, roi)
imshow(im_out/255)
show()
```

example 3
```python
from easyric.io import shp, geotiff
poly = shp.read_shp2d('../easyric/tests/file/shp_test/test.shp')
imarray, offset = geotiff.clip_roi(poly['0'], '../easyric/tests/file/tiff_test/2_12.tif', is_geo=True)
```

```python
import pyproj
from easyric.calculate import geo2tiff
from skimage.io import imshow, show

out_dict = geo2tiff.shp_clip_geotiff(shp_path='../easyric/tests/file/pix4d.diy/plots.shp',
                                     geotiff_path='../easyric/tests/file/pix4d.diy/hasu_tanashi_20170525_Ins1RGB_30m_transparent_mosaic_group1.tif',
                                     out_folder='hasu_out', shp_proj=pyproj.CRS.from_epsg(4326))

imshow(out_dict['N1W2.png']['imarray'])
show()
```

example 4
```python
from easyric.calculate import geo2tiff

shp_file = r"D:\OneDrive\Documents\4_PhD\12_CAAS_Tasks\02_ZLC_error\shp\plot.shp"
dom_path = r"D:\OneDrive\Documents\4_PhD\12_CAAS_Tasks\02_ZLC_error\0328DOM.tif"
dsm_path = r"D:\OneDrive\Documents\4_PhD\12_CAAS_Tasks\02_ZLC_error\0328DSM.tif"

dom_dict = geo2tiff.shp_clip_geotiff(shp_path=shp_file,
                                     geotiff_path=dom_path,
                                     out_folder=r'D:\OneDrive\Documents\4_PhD\12_CAAS_Tasks\02_ZLC_error\out')

dsm_dict = geo2tiff.shp_clip_geotiff(shp_path=shp_file,
                                     geotiff_path=dsm_path,
                                     out_folder=r'D:\OneDrive\Documents\4_PhD\12_CAAS_Tasks\02_ZLC_error\out')
```

# Points2RawImgPixel.py

```python
import numpy as np
from easyric.objects import Pix4D
from calculate.geo2raw import get_img_coords_dict

import matplotlib.pyplot as plt
from skimage.io import imread
import matplotlib.patches as pts
from matplotlib.collections import PatchCollection

project_name  = r"wheat_hand_ukyoto_lowD1_20200427"
project_path = r"D:\OneDrive\Documents\4_PhD\10_Data\03_Camera.Wheat.Pix4D"
params_path = f"{project_path}\params"
raw_img_folder = f"{project_path}\photos"
dom_path = f"{project_path}\\{project_name}_transparent_mosaic_group1.tif"
dsm_path = f"{project_path}\\{project_name}_dsm.tif"

points_txt = f"{project_path}/picking_list.txt"

center_points = np.loadtxt(points_txt, delimiter=',')
num = int(len(center_points) / 4)
center_point_list = []
for i in range(0, num):
    points = center_points[4*i:4*i+4, :]
    points = np.vstack([points, center_points[4*i, :][None,:]])
    center_point_list.append(points)

p4d = Pix4D(project_path=project_path, raw_img_path=raw_img_folder, project_name=project_name,
            param_folder=params_path, dom_path=dom_path, dsm_path=dsm_path)

for i, center_point in enumerate(center_point_list):
    # for those have no ground defined, pmatrix is the only way to get the correct projection coordinate
    img_name, img_coords = get_img_coords_dict(p4d, center_point, method='pmat')

    for img_n, img_c in zip(img_name, img_coords):
        print(i, img_n)
        #'''
        fig, ax = plt.subplots(1,1, figsize=(8,6), dpi=72)
        # using [::-1] to revert image along axis=0, and origin='lower' to change to 'lower-left' coordinate
        ax.imshow(imread(p4d.img[img_n].path))

        polygon = pts.Polygon(img_c, True)
        p = PatchCollection([polygon], alpha=0.5, facecolors='red')
        #ax.plot(proj.img[0].w - img_c[:,0], proj.img[0].h - img_c[:,1])

        ax.add_collection(p)
        ax.set_title(f"Reprojection on {img_n} via PMatrix files")
        # patches not need
        #ax.invert_yaxis()
        # but plt.plot need
        plt.xlabel('x')
        plt.ylabel('y')

        plt.tight_layout()
        plt.savefig(f"{project_path}/picking_list.result/id{i}-{img_n}.png")
        plt.close()
        #'''
```