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
            cache_root = _cfg.get("data_dir")
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
        """Dataset manifest name."""
        return self._name

    @property
    def title(self):
        """Human-readable dataset title."""
        return self._title

    @property
    def description(self):
        """Short dataset description from the manifest."""
        return self._description

    @property
    def size_bytes(self):
        """Compressed archive size in bytes."""
        return self._size_bytes

    @property
    def mirrors(self):
        """Read-only mapping of configured download mirrors."""
        return self._mirrors

    @property
    def required(self):
        """File keys used to decide whether the dataset is ready."""
        return self._required

    @property
    def files(self):
        """Read-only mapping from dotted file keys to relative paths."""
        return self._files

    @property
    def cache_root(self):
        """Root directory that stores EasyIDP dataset caches."""
        return self._cache_root

    @property
    def root(self):
        """Extracted dataset directory."""
        return self._ns_root

    @property
    def data_dir(self):
        """Alias of :attr:`root` kept for dataset path workflows."""
        return self._ns_root

    @property
    def archive(self):
        """Temporary archive path removed after successful extraction."""
        return self._cache_root / ".downloads" / self._archive_name

    @property
    def zip_file(self):
        """Alias of :attr:`archive` for zip-based dataset downloads."""
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
    """Dataset for the lotus plot in Tanashi, Tokyo.

    .. image:: ../../_static/images/data/2017_tanashi_lotus.png
        :width: 600
        :alt: 2017_tanashi_lotus.png

    - **Crop** : lotus
    - **Location** : Tanashi, Nishi-Tokyo, Japan
    - **Flight date** : May 31, 2017
    - **UAV model** : DJI Inspire 1
    - **Flight height** : 30 m
    - **Image number** : 142
    - **Image size** : 4608 x 3456
    - **Software** : Pix4D, Metashape
    - **Outputs** : DOM, DSM, PCD
    """

    def __init__(self, *, cache_root=None, notify_missing=True):
        """Create a lightweight handle to the lotus demo dataset.

        The constructor only exposes paths. It does not download files; call
        :meth:`download` explicitly when :meth:`is_ready` returns ``False``.

        Accessible path attributes include:

        - ``.photo`` : raw image folder
        - ``.shp`` : plot ROI shapefile
        - ``.pix4d.project`` : Pix4D project folder
        - ``.pix4d.param`` : Pix4D parameter folder
        - ``.pix4d.dom`` : Pix4D orthomosaic GeoTIFF
        - ``.pix4d.dsm`` : Pix4D digital surface model GeoTIFF
        - ``.pix4d.pcd`` : Pix4D point cloud file
        - ``.metashape.project`` : Metashape project file
        - ``.metashape.param`` : Metashape project folder
        - ``.metashape.dom`` : Metashape orthomosaic GeoTIFF
        - ``.metashape.dsm`` : Metashape digital surface model GeoTIFF
        - ``.metashape.pcd`` : Metashape point cloud file

        Parameters
        ----------
        cache_root : Path or str, optional
            Root directory for cached datasets. Defaults to
            ``idp.config.get().data_dir``.
        notify_missing : bool, optional
            Whether to log a warning when required files are missing.

        Examples
        --------
        >>> lotus = idp.data.Lotus()
        >>> lotus.shp
        PosixPath('.../2017_tanashi_lotus/plots.shp')
        >>> lotus.pix4d.dom.name
        'hasu_tanashi_20170531_Ins1RGB_30m_transparent_mosaic_group1.tif'
        """
        super().__init__("lotus", cache_root=cache_root, notify_missing=notify_missing)


class ForestBirds(Dataset):
    """Dataset for the forest ecology survey in Florida.

    .. image:: ../../_static/images/data/2022_florida_forestbirds.png
        :width: 600
        :alt: 2022_florida_forestbirds.png

    - **Author** : Prof. Ben Weinstein, The University of Florida
    - **Location** : Florida, US
    - **Flight date** : March 24, 2022
    - **UAV model** : DJI FC6540
    - **Image number** : 93
    - **Image size** : 6016 x 4008
    - **Software** : Metashape
    - **Outputs** : DOM, DSM
    """

    def __init__(self, *, cache_root=None, notify_missing=True):
        """Create a lightweight handle to the forest birds demo dataset.

        The constructor only exposes paths. It does not download files; call
        :meth:`download` explicitly when :meth:`is_ready` returns ``False``.

        Accessible path attributes include:

        - ``.photo`` : raw image folder
        - ``.shp`` : plot ROI shapefile
        - ``.metashape.project`` : Metashape project file
        - ``.metashape.param`` : Metashape project folder
        - ``.metashape.dom`` : Metashape orthomosaic GeoTIFF
        - ``.metashape.dsm`` : Metashape digital surface model GeoTIFF

        Parameters
        ----------
        cache_root : Path or str, optional
            Root directory for cached datasets. Defaults to
            ``idp.config.get().data_dir``.
        notify_missing : bool, optional
            Whether to log a warning when required files are missing.

        Examples
        --------
        >>> fb = idp.data.ForestBirds()
        >>> fb.photo
        PosixPath('.../2022_florida_forestbirds/Hidden_Little_03_24_2022')
        """
        super().__init__(
            "forestbirds", cache_root=cache_root, notify_missing=notify_missing
        )


class TestData(Dataset):
    """Developer and package test dataset.

    This dataset is mainly used by EasyIDP tests and examples. Normal users
    usually want :class:`Lotus` or :class:`ForestBirds` instead.
    """

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
        """Create a lightweight handle to the developer test dataset.

        Accessible path groups include:

        **json test module**

        - ``.json.for_read_json``
        - ``.json.labelme_demo``
        - ``.json.labelme_warn``
        - ``.json.labelme_err``
        - ``.json.geojson_soy``

        **shp test module**

        - ``.shp.lotus_shp``
        - ``.shp.lotus_prj``
        - ``.shp.complex_shp``
        - ``.shp.complex_prj``
        - ``.shp.lonlat_shp``
        - ``.shp.utm53n_shp``
        - ``.shp.utm53n_prj``
        - ``.shp.rice_shp``
        - ``.shp.rice_prj``
        - ``.shp.roi_shp``
        - ``.shp.roi_prj``
        - ``.shp.testutm_shp``
        - ``.shp.testutm_prj``
        - ``.shp.jp_crs_shp``
        - ``.shp.jp_crs_prj``
        - ``.shp.mlayer_shp``
        - ``.shp.mask_rice_roi``
        - ``.shp.mask_rice_prj``
        - ``.shp.mask_rice_gt_shp``
        - ``.shp.mask_rice_gt_prj``

        **pcd test module**

        - ``.pcd.lotus_las``
        - ``.pcd.lotus_laz``
        - ``.pcd.lotus_pcd``
        - ``.pcd.lotus_las13``
        - ``.pcd.lotus_laz13``
        - ``.pcd.lotus_ply_asc``
        - ``.pcd.lotus_ply_bin``
        - ``.pcd.maize_las``
        - ``.pcd.maize_laz``
        - ``.pcd.maize_ply``

        **roi test module**

        - ``.roi.dxf``
        - ``.roi.lxyz_txt``
        - ``.roi.xyz_txt``

        **geotiff test module**

        - ``.tiff.soyweed_part``
        - ``.tiff.mlayer_ndvi``
        - ``.tiff.mlayer_multi``
        - ``.tiff.mask_rice_geotiff_empty_polygon``
        - ``.tiff.mask_rice_geotiff_with_polygon``
        - ``.tiff.out``

        **metashape test module**

        - ``.metashape.goya_psx``
        - ``.metashape.goya_param``
        - ``.metashape.lotus_psx``
        - ``.metashape.lotus_param``
        - ``.metashape.lotus_dsm``
        - ``.metashape.wheat_psx``
        - ``.metashape.wheat_param``
        - ``.metashape.multichunk_psx``
        - ``.metashape.multichunk_param``
        - ``.metashape.multifolder_psx``
        - ``.metashape.multifolder_param``
        - ``.metashape.nestedfolder_psx``
        - ``.metashape.nestedfolder_param``
        - ``.metashape.camera_disorder_psx``
        - ``.metashape.camera_disorder_param``
        - ``.metashape.two_calib_psx``
        - ``.metashape.two_calib_param``
        - ``.metashape.multi_spectral_psx``
        - ``.metashape.multi_spectral_param``

        **pix4d test module**

        - ``.pix4d.lotus_folder``
        - ``.pix4d.lotus_param``
        - ``.pix4d.lotus_photos``
        - ``.pix4d.lotus_dom``
        - ``.pix4d.lotus_dsm``
        - ``.pix4d.lotus_pcd``
        - ``.pix4d.lotus_dom_part``
        - ``.pix4d.lotus_dsm_part``
        - ``.pix4d.lotus_pcd_part``
        - ``.pix4d.maize_folder``
        - ``.pix4d.maize_dom``
        - ``.pix4d.maize_dsm``
        - ``.pix4d.maize_noparam``
        - ``.pix4d.maize_empty``
        - ``.pix4d.maize_noout``

        **output folders**

        - ``.json.out``
        - ``.shp.out``
        - ``.pcd.out``
        - ``.tiff.out``
        - ``.cv.out``
        - ``.vis.out``
        - ``.b2r.out``

        Parameters
        ----------
        cache_root : Path or str, optional
            Root directory for cached datasets. Defaults to
            ``idp.config.get().data_dir``.
        test_out : Path or str, optional
            Folder for temporary test outputs, by default ``"./tests/out"``.
        notify_missing : bool, optional
            Whether to log a warning when required files are missing.
        """
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
