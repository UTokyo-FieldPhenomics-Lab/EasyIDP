import re
import pytest
import pyproj
import numpy as np
from pathlib import Path
import easyidp as idp

test_data = idp.data.TestData(test_out="./tests/out")

def test_dict2json():
    a = {"test": {"rua": np.asarray([[12, 34], [45, 56]])}, "hua":[np.int32(34), np.float64(34.567)]}

    # with indient = 0
    out_json = Path(test_data.json.out) / "json_indient0.json"

    idp.jsonfile.dict2json(a, out_json)

    assert out_json.exists()

    with open(out_json, 'r') as f:
        js_str = f.read()
    assert js_str == '{"test": {"rua": [[12, 34], [45, 56]]}, "hua": [34, 34.567]}'

    # with indient = 4
    out_json4 = test_data.json.out / "json_indient4.json"

    idp.jsonfile.dict2json(a, out_json4, indent=4)
    assert out_json.exists()

    with open(out_json4, 'r') as f:
        js_str4 = f.read()

    assert js_str4 == '{\n    "test": {\n        "rua": [\n            [\n                12,\n                34\n            ],\n            [\n                45,\n                56\n            ]\n        ]\n    },\n    "hua": [\n        34,\n        34.567\n    ]\n}'

def test_read_dict():
    with pytest.raises(FileNotFoundError, match=re.escape("Could not locate the given json file")):
        out = idp.jsonfile.read_json(test_data.json / "non_exist.json")

    out = idp.jsonfile.read_json(test_data.json.for_read_json)

    assert out == {"test": {"rua": [[12, 34], [45, 56]]}, "hua": [34, 34.567]}

def test_read_geojson_wrong_format():
    with pytest.raises(TypeError, match=re.escape("The input file format should be *.geojson, not [.json]")):
        # input a *.json file
        out = idp.jsonfile.read_geojson(test_data.json.labelme_demo)

def test_read_geojson():
    out, crs = idp.jsonfile.read_geojson(test_data.json.geojson_soy, return_proj=True)

    assert crs == pyproj.CRS.from_epsg(6677)

    assert len(out) == 260

def test_read_geojson_key_names():
    gjs_path = test_data.json.geojson_soy

    str_no_name_field_title_true= idp.jsonfile.read_geojson(gjs_path, include_title=True)
    assert "# 1" in str_no_name_field_title_true.keys()

    str_name_field_title_false = idp.jsonfile.read_geojson(gjs_path, name_field="FID")
    assert "259" in str_name_field_title_false.keys()

    int_name_field_title_false = idp.jsonfile.read_geojson(gjs_path, name_field=0)
    assert "259" in int_name_field_title_false.keys()

    str_name_field_title_true = idp.jsonfile.read_geojson(gjs_path, name_field="FID", include_title=True)
    assert "FID 65" in str_name_field_title_true.keys()

    int_name_field_title_true = idp.jsonfile.read_geojson(gjs_path, name_field=0, include_title=True)
    assert "FID 65" in int_name_field_title_true.keys()

    index_name_field_title_true = idp.jsonfile.read_geojson(gjs_path, name_field=-1, include_title=True)
    assert "# 1" in index_name_field_title_true.keys()

    str_index_name_field_title_true = idp.jsonfile.read_geojson(gjs_path, name_field='#', include_title=True)
    assert "# 1" in str_index_name_field_title_true.keys()

    str_index_name_field_title_false = idp.jsonfile.read_geojson(gjs_path, name_field='#', include_title=False)
    assert "1" in str_index_name_field_title_false.keys()

def test_read_geojson_key_names_merge():
    # merge several columns
    gjs_path = test_data.json.geojson_soy

    str_name_field_list_title_false = idp.jsonfile.read_geojson(
        gjs_path, name_field=["FID", "plotName"]
    )
    assert "65|Enrei-10" in str_name_field_list_title_false.keys()

    str_name_field_list_title_true = idp.jsonfile.read_geojson(
        gjs_path, name_field=["FID", "plotName"], include_title=True
    )
    assert "FID 65|plotName Enrei-10" in str_name_field_list_title_true.keys()

    int_name_field_list_title_false = idp.jsonfile.read_geojson(
        gjs_path, name_field=[0, 4]
    )
    assert "65|Enrei-10" in int_name_field_list_title_false.keys()

    int_name_field_list_title_true = idp.jsonfile.read_geojson(
        gjs_path, name_field=[0, 4], include_title=True
    )
    assert "FID 65|plotName Enrei-10" in int_name_field_list_title_true.keys()
    
    int_index_name_field_list_title_true = idp.jsonfile.read_geojson(
        gjs_path, name_field=[-1, 4], include_title=True
    )
    assert "# 0|plotName Enrei-10" in int_index_name_field_list_title_true.keys()
    
    str_index_name_field_list_title_false = idp.jsonfile.read_geojson(
        gjs_path, name_field=["#", "FID"], include_title=False
    )
    assert "0|65" in str_index_name_field_list_title_false.keys()

geojson_table_preview = \
"  [-1] #   [0] FID    [1] 試験区    [2] ID    [3] 除草剤    [4] plotName    [5] lineNum\n"\
"--------  ---------  ------------  --------  ------------  --------------  -------------\n"\
"       0     65       SubBlk 2b       0           有          Enrei-10           1\n"\
"       1     97       SubBlk 2b       0           有          Enrei-20           1\n"\
"       2     147      SubBlk 2b       0           有        Nakasenri-10         1\n"\
"     ...     ...         ...         ...         ...            ...             ...\n"\
"     257     259        SB 0a         0                    Tachinagaha-10        3\n"\
"     258      4         SB 0a         0                    Fukuyutaka-10         3\n"\
"     259      1       SubBlk 2a       0           無          Enrei-20           1\n"
def test_show_geojson_fields(capfd):

    idp.jsonfile.show_geojson_fields(test_data.json.geojson_soy)

    out, err = capfd.readouterr()

    assert out.replace(' ', '').replace('-','') ==  geojson_table_preview.replace(' ', '').replace('-','') 