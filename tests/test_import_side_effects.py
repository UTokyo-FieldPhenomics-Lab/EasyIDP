import importlib
import subprocess

import pytest


def test_import_easyidp_does_not_check_network_or_install(monkeypatch):
    pytest.importorskip("requests")
    import requests

    def fail_get(*args, **kwargs):
        raise AssertionError("import easyidp must not call requests.get")

    def fail_run(*args, **kwargs):
        raise AssertionError("import easyidp must not run subprocess")

    monkeypatch.setattr(requests, "get", fail_get)
    monkeypatch.setattr(subprocess, "run", fail_run)

    import easyidp

    importlib.reload(easyidp)
