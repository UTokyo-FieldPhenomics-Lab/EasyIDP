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
    out = idp.jsonfile.read_geojson(test_data.json.geojson_soy)

    assert out['crs'] == pyproj.CRS.from_epsg(6677)

    assert len(out['geometry']) == len(out['property'])

geojson_table_preview_unix = \
"                   Properties of /Users/hwang/Library/Application                   \n"\
"      Support/easyidp.data/data_for_tests/json_test/2023_soybean_field.geojson      \n"\
" ────────────────────────────────────────────────────────────────────────────────── \n"\
"  [-1]   [0] FID   [1] 試験区   [2] ID   [3] 除草剤    [4] plotName    [5] lineNum  \n"\
" ────────────────────────────────────────────────────────────────────────────────── \n"\
"     0     65      SubBlk 2b      0          有          Enrei-10           1       \n"\
"     1     97      SubBlk 2b      0          有          Enrei-20           1       \n"\
"     2     147     SubBlk 2b      0          有        Nakasenri-10         1       \n"\
"   ...     ...        ...        ...        ...            ...             ...      \n"\
"   257     259       SB 0a        0                   Tachinagaha-10        3       \n"\
"   258      4        SB 0a        0                   Fukuyutaka-10         3       \n"\
"   259      1      SubBlk 2a      0          無          Enrei-20           1       \n"\
" ────────────────────────────────────────────────────────────────────────────────── \n"
def test_show_geojson_fields(capfd):

    idp.jsonfile.show_geojson_fields(test_data.json.geojson_soy)

    out, err = capfd.readouterr()

    # just ensure the function can run, no need to guarantee the same string outputs
    assert "Properties of" in out