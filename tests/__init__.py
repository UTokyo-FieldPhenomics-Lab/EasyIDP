import os
import sys

sys.path.insert(0, ".")

# check if output path exists
out_dir = r"./tests/out"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

out_folders = ["json_test", "pcd_test", "cv_test", "tiff_test", "visual_test"]

for o in out_folders:
    sub_dir = os.path.join(out_dir, o)
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)