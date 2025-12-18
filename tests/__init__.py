import os
import sys
from pathlib import Path
from loguru import logger
import pytest

sys.path.insert(0, ".")

# diable loguru to log file when vscode testing init
os.environ["IS_TESTING"] = "True"

import easyidp as idp

# check if output path exists
out_dir = Path("./tests/out")
if not out_dir.exists():
    out_dir.mkdir()

out_folders = [
    "json_test", "pcd_test", "cv_test", "tiff_test", 
    "visual_test", "back2raw_test", "data_test"
]

for o in out_folders:
    sub_dir = out_dir / o
    if not sub_dir.exists():
        sub_dir.mkdir()

@pytest.fixture(scope="module")
def shared_data():
    test_data = idp.data.TestData()

    roi_all = idp.ROI(test_data.shp.lotus_shp, name_field=0)

    # global variable for testing
    # shorten for quick for loops
    roi_select = idp.ROI() 
    for key in ["N1W1", "N1W2", "N2E2", "S1W1"]:
        roi_select[key] = roi_all[key]
        roi_select.crs = roi_all.crs
        roi_select.source = roi_all.source

    p4d = idp.Pix4D(project_path=test_data.pix4d.lotus_folder, 
                    raw_img_folder=test_data.pix4d.lotus_photos,
                    param_folder=test_data.pix4d.lotus_param)
    ms = idp.Metashape(test_data.metashape.lotus_psx, chunk_id=0)

    roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)

    # only pick 1 plots as testing data
    roi = roi[0:1]
    roi.get_z_from_dsm(test_data.pix4d.lotus_dsm)

    out_all = p4d.back2raw(roi)

    # for visualization.test
    roi_vis = idp.ROI(test_data.shp.lotus_shp, name_field='plot_id')
    roi_vis.get_z_from_dsm(test_data.metashape.lotus_dsm, mode="point")

    return {
        "test_data": test_data, "roi_all": roi_all, "roi_select": roi_select, 
        "p4d": p4d, "ms": ms, "roi": roi, "out_all": out_all,
        "roi_vis": roi_vis
    }

@pytest.fixture
def report_loguru_to_caplog(caplog):
    """
    将 loguru 的日志重定向到 pytest 的 caplog handler 中，
    这样就可以在测试中使用 caplog 来断言 loguru 的输出了。
    """
    # 添加 caplog 的 handler 到 loguru
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level="WARNING",
        filter=lambda record: record["level"].no >= caplog.handler.level,
    )
    yield caplog
    # 测试结束后移除 handler，避免污染其他测试
    logger.remove(handler_id)


if __name__ == "__main__":
    # Download test data when running this script directly
    # Used by GitHub Actions workflow to pre-download test data
    print("Downloading test data...")
    test_data = idp.data.TestData()
    print(f"Test data downloaded to: {test_data.data_dir}")