"""EasyIDP dataset models backed by JSON manifests."""

import json
import types
from pathlib import Path

import easyidp.config as _cfg
from easyidp.logger import logger

_MANIFEST_DIR = Path(__file__).parent / "datasets"

_RESERVED_ATTRS = frozenset({
    "name", "title", "description", "size_bytes", "mirrors", "required",
    "files", "cache_root", "root", "archive", "data_dir", "zip_file",
    "path", "is_ready", "dry_run", "download", "test_out",
})


class _PathNamespace:
    """Recursively expose nested file mappings as path attributes.

    String leaf values become ``root / value``.  Intermediate mappings
    become nested ``_PathNamespace`` objects sharing the same root.

    Parameters
    ----------
    root : Path
        Absolute base directory.
    tree : dict
        Nested dict of relative path strings.

    Examples
    --------
    >>> from pathlib import Path
    >>> ns = _PathNamespace(Path("/data"), {"ms": {"dom": "outputs/dom.tif"}})
    >>> ns.ms.dom
    PosixPath('/data/outputs/dom.tif')
    """

    def __init__(self, root, tree):
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_items", dict(tree))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        items = object.__getattribute__(self, "_items")
        root = object.__getattribute__(self, "_root")
        if not isinstance(items, dict) or name not in items:
            raise AttributeError(name)
        value = items[name]
        if isinstance(value, dict):
            return _PathNamespace(root, value)
        return root / value

    def __setattr__(self, name, value):
        items = object.__getattribute__(self, "_items")
        if isinstance(items, dict) and name in items:
            raise AttributeError(f"readonly attribute: {name!r}")
        object.__setattr__(self, name, value)

    def __truediv__(self, other):
        return object.__getattribute__(self, "_root") / other

    def __repr__(self):
        root = object.__getattribute__(self, "_root")
        items = object.__getattribute__(self, "_items")
        return f"_PathNamespace({root}, {items!r})"


def _load_manifest(path):
    """Load and parse a JSON manifest file.

    Parameters
    ----------
    path : Path
        Path to the JSON manifest.

    Returns
    -------
    dict
        Parsed manifest data.

    Raises
    ------
    FileNotFoundError
        If the manifest file does not exist.
    ValueError
        If the JSON is malformed.
    """
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _validate_attr_name(name):
    """Check that a dotted file key does not conflict with reserved
    Dataset attributes.

    Parameters
    ----------
    name : str
        Dotted key (e.g. ``"metashape.project"``).

    Raises
    ------
    ValueError
        If any segment of *name* is a reserved attribute.
    """
    for part in name.split("."):
        if part in _RESERVED_ATTRS:
            raise ValueError(
                f"file key {name!r} contains reserved attribute {part!r}"
            )


def _validate_manifest(data):
    """Validate manifest structure and raise ``ValueError`` on problems.

    Parameters
    ----------
    data : dict
        Loaded manifest dictionary.

    Raises
    ------
    ValueError
        If required fields are missing or have wrong types.
    """
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a dict, got {type(data).__name__}")

    spec = data.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("manifest missing 'spec' section")

    for field in ("name", "title", "folder", "size_bytes"):
        if field not in spec:
            raise ValueError(f"manifest spec missing required field: {field!r}")

    if "mirrors" in spec and not isinstance(spec["mirrors"], dict):
        raise ValueError("spec.mirrors must be a dict")

    files = data.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("manifest 'files' must be a dict")

    for key in files:
        _validate_attr_name(key)

    required = data.get("required", [])
    if not isinstance(required, list):
        raise ValueError("manifest 'required' must be a list")
    for key in required:
        if key not in files:
            raise ValueError(f"required key {key!r} not found in files")


def _insert_path(obj, files, root):
    """Build ``_PathNamespace`` attributes from dotted file keys on *obj*.

    Parameters
    ----------
    obj : object
        Target object (typically a ``Dataset`` instance).
    files : Mapping
        Flat dotted-key → relative-path mapping.
    root : Path
        Absolute base directory.
    """
    tree = {}
    for key, value in files.items():
        node = tree
        parts = key.split(".")
        *groups, leaf = parts
        for group in groups:
            next_node = node.setdefault(group, {})
            if not isinstance(next_node, dict):
                raise ValueError(
                    f"key conflict: {key!r} — {group!r} is already a leaf"
                )
            node = next_node
        if isinstance(node.get(leaf), dict):
            raise ValueError(
                f"key conflict: {key!r} — {leaf!r} is already a group"
            )
        node[leaf] = value

    for name, subtree in tree.items():
        if isinstance(subtree, dict):
            setattr(obj, name, _PathNamespace(root, subtree))
        else:
            setattr(obj, name, root / subtree)


def list_datasets():
    """Return the names of available EasyIDP demo datasets.

    Returns
    -------
    list of str
        Manifest names (without ``.json`` extension), excluding internal
        manifests such as ``download_smoke``.
    """
    skip = {"download_smoke"}
    names = [p.stem for p in _MANIFEST_DIR.glob("*.json") if p.stem not in skip]
    names.sort()
    return names


class Dataset:
    """EasyIDP dataset backed by a JSON manifest.

    Parameters
    ----------
    manifest_name : str
        Name of the JSON manifest without extension (e.g. ``"lotus"``).
    cache_root : Path or str, optional
        Root directory for cached datasets.  Defaults to the value
        returned by :func:`easyidp.config.get().data_dir`.
    notify_missing : bool, optional
        Whether to log a warning when required files are missing.
    """

    def __init__(self, manifest_name, cache_root=None, notify_missing=True):
        if cache_root is None:
            cache_root = _cfg.get().data_dir
        self._cache_root = Path(cache_root).expanduser()
        self._notify_missing = notify_missing

        manifest_path = _MANIFEST_DIR / f"{manifest_name}.json"
        data = _load_manifest(manifest_path)
        _validate_manifest(data)

        spec = data["spec"]
        self._manifest = types.MappingProxyType(data)
        self._name = spec["name"]
        self._title = spec["title"]
        self._description = spec.get("description", "")
        self._size_bytes = spec["size_bytes"]
        self._mirrors = types.MappingProxyType(spec.get("mirrors", {}))
        self._folder = spec["folder"]
        self._archive_name = spec.get("archive", f"{self._folder}.zip")

        self._required = tuple(data.get("required", ()))
        self._files = types.MappingProxyType(data.get("files", {}))

        self._ns_root = self._cache_root / self._folder
        _insert_path(self, self._files, self._ns_root)
        if self._notify_missing and not self.is_ready():
            logger.warning(
                "Dataset '{}' is not ready. Call .download() to fetch it.",
                self._name,
            )

    # -- read-only properties ------------------------------------------------

    @property
    def name(self):
        return self._name

    @property
    def title(self):
        return self._title

    @property
    def description(self):
        return self._description

    @property
    def size_bytes(self):
        return self._size_bytes

    @property
    def mirrors(self):
        return self._mirrors

    @property
    def required(self):
        return self._required

    @property
    def files(self):
        return self._files

    @property
    def cache_root(self):
        return self._cache_root

    @property
    def root(self):
        return self._ns_root

    @property
    def data_dir(self):
        return self._ns_root

    @property
    def archive(self):
        return self._cache_root / ".downloads" / self._archive_name

    @property
    def zip_file(self):
        return self.archive

    # -- public methods ------------------------------------------------------

    def path(self, key):
        """Return the absolute ``Path`` for a dotted file key.

        Parameters
        ----------
        key : str
            Dotted file key (e.g. ``"metashape.project"``).

        Returns
        -------
        Path
            Absolute path.
        """
        return self._ns_root / self._files[key]

    def is_ready(self):
        """Return ``True`` when every required file exists on disk.

        If no ``required`` list is present, all files are checked.
        """
        check_keys = self._required if self._required else self._files
        return all(self.path(k).exists() for k in check_keys)

    def dry_run(self):
        """Return a JSON-friendly summary dict without touching the network.

        Returns
        -------
        dict
            Summary with keys ``name``, ``root``, ``archive``, ``ready``,
            ``needs_download``, ``size_bytes``, ``mirrors``, ``missing``.
            All path values are plain strings.
        """
        ready = self.is_ready()
        check_keys = self._required if self._required else list(self._files)
        missing = [
            self._files[k] for k in check_keys if not self.path(k).exists()
        ]
        return {
            "name": self._name,
            "root": str(self._ns_root),
            "archive": str(self.archive),
            "ready": ready,
            "needs_download": not ready,
            "size_bytes": self._size_bytes,
            "mirrors": dict(self._mirrors),
            "missing": missing,
        }

    def download(self, mirror="auto", force=False, progress=True):
        """Download this dataset to *cache_root*.

        Parameters
        ----------
        mirror : str, optional
            Mirror name or ``"auto"`` (default) to pick the first available.
        force : bool, optional
            Re-download even if the dataset is ready.
        progress : bool, optional
            Show a progress bar during download.
        """
        from .downloader import download_dataset

        return download_dataset(self, mirror=mirror, force=force, progress=progress)


class Lotus(Dataset):
    """Dataset for the Tanashi lotus plot."""

    def __init__(self, *, cache_root=None, notify_missing=True):
        super().__init__("lotus", cache_root=cache_root, notify_missing=notify_missing)


class ForestBirds(Dataset):
    """Dataset for the Florida forest birds survey."""

    def __init__(self, *, cache_root=None, notify_missing=True):
        super().__init__(
            "forestbirds", cache_root=cache_root, notify_missing=notify_missing
        )


class TestData(Dataset):
    """Developer and package test dataset."""

    __test__ = False

    _OUT_GROUPS = {
        "json": "json_test",
        "shp": "shp_test",
        "pcd": "pcd_test",
        "tiff": "tiff_test",
        "cv": "cv_test",
        "vis": "visual_test",
        "b2r": "back2raw_test",
    }

    def __init__(self, *, cache_root=None, test_out="./tests/out", notify_missing=True):
        super().__init__(
            "testdata", cache_root=cache_root, notify_missing=notify_missing
        )
        self.test_out = Path(test_out).expanduser()
        for group, subdir in self._OUT_GROUPS.items():
            ns = getattr(self, group, None)
            if ns is None:
                ns = _PathNamespace(self.root, {})
                object.__setattr__(self, group, ns)
            ns.out = self.test_out / subdir
