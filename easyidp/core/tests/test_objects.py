import numpy as np
import pandas as pd
from easyidp.core.objects import Points, Container, Photo

def test_def_points():
    p1 = Points([1,2,3])
    assert type(p1) == pd.DataFrame

    p2 = Points([[1,2,3]])
    np.testing.assert_array_almost_equal(p1.values, p2.values)

    p3 = Points([[1,2,3], [4,5,6]])
    assert list(p3.columns.values) == ['x', 'y', 'z']

    p4 = Points([1,2])
    assert list(p4.columns.values) == ['x', 'y']

    p5 = Points([1,2], columns=['lon', 'lat', 'alt'])
    assert list(p5.columns.values) == ['lon', 'lat']


def test_container():
    p1 = Photo()
    p2 = Photo()
    p1.id = 1
    p1.label = "aaa.jpg"
    p2.id = 2
    p2.label = "bbb.jpg"

    a = Container()
    a[p1.id] = p1
    a[p2.id] = p2

    assert len(a) == 2
    assert a[1] == p1
    assert a["bbb.jpg"] == p2
