import numpy as np
from easyric.io import json

def test_dict2json():
    a = {'test': {'rua': np.asarray([[12, 34], [45, 56]])}, 'hua':[np.int32(34), np.float64(34.567)]}

    json.dict2json(a, 'out/dict2json.json')