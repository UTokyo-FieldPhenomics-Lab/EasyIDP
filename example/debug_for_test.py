#import pyproj
#from easyric.io.shp import read_shp3d
#
#lonlat_z = read_shp3d(r'../easyric/tests/file/pix4d.diy/plots.shp',
#                      r'../easyric/tests/file/pix4d.diy/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif',
#                      get_z_by='local', given_proj=pyproj.CRS.from_epsg(4326))
#print(lonlat_z)
#---------------------------------------------
#import numpy as np
#from easyric.io import geotiff
#from skimage.io import imread, imshow, show
#from skimage.color import rgb2gray
#
#photo_path = '../easyric/tests/file/pix4d.diy/photos/DJI_0174.JPG'
#roi = np.asarray([[2251, 1223], [2270, 1270], [2227, 1263], [2251, 1223]])
##imarray = rgb2gray(imread(photo_path))
#imarray = imread(photo_path)
#
#im_out, offsets = geotiff._imarray_clip(imarray, roi)
#imshow(im_out/255)
#show()
#---------------------------------------------
#from easyric.io import shp, geotiff
#poly = shp.read_shp2d('../easyric/tests/file/shp_test/test.shp')
#imarray, offset = geotiff.clip_roi(poly['0'], '../easyric/tests/file/tiff_test/2_12.tif', is_geo=True)
#---------------------------------------------
import pyproj
from easyric.calculate import geo2tiff
from skimage.io import imshow, show

out_dict = geo2tiff.shp_clip_geotiff(shp_path='../easyric/tests/file/pix4d.diy/plots.shp',
                                     geotiff_path='../easyric/tests/file/pix4d.diy/hasu_tanashi_20170525_Ins1RGB_30m_transparent_mosaic_group1.tif',
                                     out_folder='hasu_out', shp_proj=pyproj.CRS.from_epsg(4326))

imshow(out_dict['N1W2.png']['imarray'])
show()