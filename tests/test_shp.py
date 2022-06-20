import os
import pytest
import easyidp as idp


data_path =  "./tests/data/shp_test"

########################
# test show shp fields #
########################

complex_shp_review_num5 = '| [1] ID                      | [2] MASSIFID        | [3] CROPTYPE   | [4] CROPDATE   | [5] CROPAREA   | [6] ATTID   |\n|-----------------------------|---------------------|----------------|----------------|----------------|-------------|\n| 230104112201809010000000000 | 2301041120000000000 | 小麦           | 2018-09-01     | 61525.26302    |             |\n| 230104112201809010000000012 | 2301041120000000012 | 蔬菜           | 2018-09-01     | 2802.33512     |             |\n| 230104112201809010000000014 | 2301041120000000014 | 玉米           | 2018-09-01     | 6960.7745      |             |\n| 230104112201809010000000061 | 2301041120000000061 | 牧草           | 2018-09-01     | 25349.08639    |             |\n| 230104112201809010000000062 | 2301041120000000062 | 玉米           | 2018-09-01     | 71463.27666    |             |\n| ...                         | ...                 | ...            | ...            | ...            | ...         |\n| 230104112201809010000000582 | 2301041120000000582 | 胡萝卜         | 2018-09-01     | 288.23876      |             |\n| 230104112201809010000000577 | 2301041120000000577 | 杂豆           | 2018-09-01     | 2001.80384     |             |\n| 230104112201809010000000583 | 2301041120000000583 | 大豆           | 2018-09-01     | 380.41704      |             |\n| 230104112201809010000000584 | 2301041120000000584 | 其它           | 2018-09-01     | 9133.25998     |             |\n| 230104112201809010000000585 | 2301041120000000585 | 其它           | 2018-09-01     | 1704.27193     |             |'

roi_shp_preview = '|   [1] id |\n|----------|\n|        0 |\n|        1 |\n|        2 |'

plots_shp_preview5 = '| [1] plot_id   |\n|---------------|\n| N1W1          |\n| N1W2          |\n| N1W3          |\n| N1W4          |\n| N1W5          |\n| ...           |\n| S4E3          |\n| S4E4          |\n| S4E5          |\n| S4E6          |\n| S4E7          |'

plots_shp_preview3 = '| [1] plot_id   |\n|---------------|\n| N1W1          |\n| N1W2          |\n| N1W3          |\n| ...           |\n| S4E5          |\n| S4E6          |\n| S4E7          |'

def test_show_shp_fields_complex(capfd):
    test_shp = os.path.join(data_path, "complex_shp_review.shp")

    idp.shp.show_shp_fields(test_shp, encoding="GBK") 
    out, err = capfd.readouterr()
    assert out ==  complex_shp_review_num5 + '\n'

def test_show_shp_fields_roi(capfd):
    test_shp = os.path.join(data_path, "roi.shp")

    idp.shp.show_shp_fields(test_shp) 
    out, err = capfd.readouterr()
    assert out ==  roi_shp_preview + '\n'

def test_show_shp_fields_plots_5(capfd):
    test_shp = r"./tests/data/pix4d/lotus_tanashi_full/plots.shp"

    idp.shp.show_shp_fields(test_shp) 
    out, err = capfd.readouterr()
    assert out ==  plots_shp_preview5 + '\n'

def test_show_shp_fields_plots_3(capfd):
    test_shp = r"./tests/data/pix4d/lotus_tanashi_full/plots.shp"

    idp.shp.show_shp_fields(test_shp, show_num=3) 
    out, err = capfd.readouterr()
    assert out ==  plots_shp_preview3 + '\n'