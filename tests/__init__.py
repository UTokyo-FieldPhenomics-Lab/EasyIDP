import sys
from pathlib import Path

sys.path.insert(0, ".")

# check if output path exists
out_dir = Path("./tests/out")
if not out_dir.exists():
    out_dir.mkdir()

out_folders = ["json_test", "pcd_test", "cv_test", "tiff_test", "visual_test"]

for o in out_folders:
    sub_dir = out_dir / o
    if not sub_dir.exists():
        sub_dir.mkdir()