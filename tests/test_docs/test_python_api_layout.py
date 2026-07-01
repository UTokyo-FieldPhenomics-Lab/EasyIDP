from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON_API = ROOT / "docs" / "python_api"
AUTODOC = PYTHON_API / "autodoc"


def _read_doc(relative_path):
    """Return a documentation source file as text.

    Parameters
    ----------
    relative_path : str
        Path below ``docs/python_api``.

    Returns
    -------
    str
        Source text.

    Examples
    --------
    >>> text = _read_doc("config.rst")
    >>> "Config" in text
    True
    """
    return (PYTHON_API / relative_path).read_text(encoding="utf-8")


def test_config_public_functions_are_inline():
    text = _read_doc("config.rst")
    assert ".. autofunction:: get" in text
    assert ".. autofunction:: set" in text
    assert ".. autofunction:: reset" in text
    assert "autodoc/easyidp.config.advanced" in text


def test_data_public_function_is_inline():
    text = _read_doc("data.rst")
    assert ".. autofunction:: list_datasets" in text
    assert "autodoc/easyidp.data.advanced" in text


def test_data_page_keeps_quick_start_concise():
    text = _read_doc("data.rst")
    assert "Quick Start\n===========" in text
    assert ">>> lotus = idp.data.Lotus()" in text
    assert ">>> lotus" in text
    assert ">>> lotus.download()" in text
    assert ">>> lotus.shp" in text
    assert "lotus.metashape.project" in text
    assert "lotus.download(mirror=\"modelscope\")" in text
    assert "Google Drive" in text
    assert "mainland China" in text
    assert "REPL Representation" not in text
    assert "Mirrors\n=======" not in text
    assert "uv sync --all-groups --all-extras" not in text


def test_hidden_config_advanced_page_consolidates_helpers():
    text = (AUTODOC / "easyidp.config.advanced.rst").read_text(encoding="utf-8")
    assert text.startswith(":orphan:")
    assert ".. autoclass:: EasyIDPConfig" in text
    assert ".. autofunction:: default_config_path" in text
    assert ".. autofunction:: default_data_dir" in text
    assert ".. autosummary::" not in text
    assert ":toctree:" not in text


def test_hidden_data_advanced_page_consolidates_helpers():
    text = (AUTODOC / "easyidp.data.advanced.rst").read_text(encoding="utf-8")
    assert text.startswith(":orphan:")
    assert text.index(
        ".. autofunction:: easyidp.data.downloader.download_dataset"
    ) < text.index(
        ".. autofunction:: easyidp.data.dataset._insert_path"
    )
    assert text.index(
        ".. autofunction:: easyidp.data.downloader.safe_extract_zip"
    ) < text.index(
        ".. autofunction:: easyidp.data.downloader._select_mirror"
    )
    assert ".. autoclass:: easyidp.data.dataset.Dataset" in text
    assert ".. autoclass:: easyidp.data.dataset._PathNamespace" in text
    assert ".. autosummary::" not in text
    assert ":toctree:" not in text


def test_data_dataset_pages_link_to_inherited_dataset_api():
    for page, class_name in [
        ("easyidp.data.Lotus.rst", "Lotus"),
        ("easyidp.data.ForestBirds.rst", "ForestBirds"),
        ("easyidp.data.TestData.rst", "TestData"),
    ]:
        text = (AUTODOC / page).read_text(encoding="utf-8")
        assert "inherits the common dataset API" in text
        assert ":class:`easyidp.data.dataset.Dataset`" in text
        assert ":attr:`name <easyidp.data.dataset.Dataset.name>`" in text
        assert ":attr:`root <easyidp.data.dataset.Dataset.root>`" in text
        assert ":meth:`download() <easyidp.data.dataset.Dataset.download>`" in text
        assert ":meth:`dry_run() <easyidp.data.dataset.Dataset.dry_run>`" in text
        assert ":meth:`is_ready() <easyidp.data.dataset.Dataset.is_ready>`" in text
        assert ":meth:`path() <easyidp.data.dataset.Dataset.path>`" in text
        assert f"~{class_name}.download" not in text
        assert f"~{class_name}.name" not in text
        assert "Inherited Dataset API\n=====================" not in text
        assert "Attributes\n----------" not in text
        assert "Methods\n-------" not in text


def test_obsolete_config_data_autodoc_pages_are_removed():
    obsolete_pages = [
        "easyidp.config.get.rst",
        "easyidp.config.set.rst",
        "easyidp.config.reset.rst",
        "easyidp.config.EasyIDPConfig.rst",
        "easyidp.config.default_config_path.rst",
        "easyidp.config.default_data_dir.rst",
        "easyidp.data.list_datasets.rst",
        "easyidp.data.dataset.Dataset.rst",
        "easyidp.data.dataset._PathNamespace.rst",
        "easyidp.data.downloader.download_dataset.rst",
        "easyidp.data.downloader.safe_extract_zip.rst",
    ]
    for page in obsolete_pages:
        assert not (AUTODOC / page).exists(), page
