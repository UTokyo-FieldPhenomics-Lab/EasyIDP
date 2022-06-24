import os
import re
import pytest
import numpy as np
import easyidp as idp

####################
# global variables #
####################
data_path =  "./tests/data/json_test"

out_path = "./tests/out/json_test"
if not os.path.exists(out_path):
    os.makedirs(out_path)

##############
# test start #
##############

def test_dict2json():
    a = {"test": {"rua": np.asarray([[12, 34], [45, 56]])}, "hua":[np.int32(34), np.float64(34.567)]}

    # with indient = 0
    out_json = os.path.join(out_path, "json_indient0.json")

    idp.jsonfile.dict2json(a, out_json)

    assert os.path.exists(out_json)

    with open(out_json, 'r') as f:
        js_str = f.read()
    assert js_str == '{"test": {"rua": [[12, 34], [45, 56]]}, "hua": [34, 34.567]}'

    # with indient = 4
    out_json4 = os.path.join(out_path, "json_indient4.json")

    idp.jsonfile.dict2json(a, out_json4, indent=4)
    assert os.path.exists(out_json)

    with open(out_json4, 'r') as f:
        js_str4 = f.read()

    assert js_str4 == '{\n    "test": {\n        "rua": [\n            [\n                12,\n                34\n            ],\n            [\n                45,\n                56\n            ]\n        ]\n    },\n    "hua": [\n        34,\n        34.567\n    ]\n}'

def test_read_dict():
    test_json_error_path = os.path.join(data_path, "non_exist.json")
    with pytest.raises(FileNotFoundError, match=re.escape("Could not locate the given json file")):
        out = idp.jsonfile.read_json(test_json_error_path)

    test_json_path =  os.path.join(data_path, "for_read_json.json")
    out = idp.jsonfile.read_json(test_json_path)

    assert out == {"test": {"rua": [[12, 34], [45, 56]]}, "hua": [34, 34.567]}
