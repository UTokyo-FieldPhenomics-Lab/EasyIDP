import easyric
import numpy as np
from easyric.objects import Params
from easyric.calculation import external_internal_calc, get_img_name_and_coords

import matplotlib.pyplot as plt
from skimage.io import imread

center_points = np.asarray([[3.539000034332,49.007999420166,5.769000053406],
                            [6.722000122070,49.053001403809,5.737999916077],
                            [6.689000129700,47.173000335693,1.628000020981],
                            [3.582999944687,47.242000579834,1.787999987602],
                            [3.539000034332,49.007999420166,5.769000053406]])

project_name  = r"wheat_hand_ukyoto_lowD1_20200427"
project_path = r"D:\OneDrive\Documents\4_PhD\14_[Paper]BroccoliPaper\Data\wheat_revert.demo"
params_path = f"{project_path}\params"
raw_img_folder = f"{project_path}\photos"
dom_path = f"{project_path}\\{project_name}_transparent_mosaic_group1.tif"
dsm_path = f"{project_path}\\{project_name}_dsm.tif"

proj = Params(param_path=params_path, project_name=project_name,
              raw_img_folder=raw_img_folder, dom_path=dom_path, dsm_path=dsm_path)

coords = external_internal_calc(proj, center_points, 'R0010743.JPG')
print(coords)

img_name, img_coords = easyric.calculation.get_img_name_and_coords(proj, center_points, method='exin')

fig, ax = plt.subplots(1,1, figsize=(16,9), dpi=300)
# using [::-1] to revert image along axis=0, and origin='lower' to change to 'lower-left' coordinate
ax.imshow(imread(f"{raw_img_folder}/{img_name[0]}")[::-1,:,:], origin="lower")

coords = img_coords[0]
ax.plot(coords[:,0],coords[:,1])

ax.set_title(f"Reprojection on {img_name[0]} via Ex/Internal files")
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()