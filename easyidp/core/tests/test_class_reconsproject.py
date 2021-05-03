import os
import pytest
import easyidp
from easyidp.core.objects import ReconsProject

module_path = os.path.join(easyidp.__path__[0], "io/tests")

def test_init_reconsproject():
    attempt1 = ReconsProject("agisoft")
    assert attempt1.software == "metashape"

    attempt2 = ReconsProject("Metashape")
    assert attempt2.software == "metashape"

    with pytest.raises(LookupError):
        attempt3 = ReconsProject("not_supported_sfm")
