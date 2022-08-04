import re
import pytest
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
    assert out_json.exists

    with open(out_json4, 'r') as f:
        js_str4 = f.read()

    assert js_str4 == '{\n    "test": {\n        "rua": [\n            [\n                12,\n                34\n            ],\n            [\n                45,\n                56\n            ]\n        ]\n    },\n    "hua": [\n        34,\n        34.567\n    ]\n}'

def test_read_dict():
    with pytest.raises(FileNotFoundError, match=re.escape("Could not locate the given json file")):
        out = idp.jsonfile.read_json(test_data.json / "non_exist.json")

    out = idp.jsonfile.read_json(test_data.json.for_read_json)

    assert out == {"test": {"rua": [[12, 34], [45, 56]]}, "hua": [34, 34.567]}
