import os
import pyproj
import shapefile
import numpy as np
import warnings
from tabulate import tabulate
from tqdm import tqdm
from pathlib import Path
from loguru import logger

import easyidp as idp


def read_proj(prj_path):
    """read \\*.prj file to pyproj object

    Parameters
    ----------
    prj_path : str
        the file path of shp \\*.prj

    Returns
    -------
    <pyproj.CRS> object

    Example
    -------

    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> prj_path = test_data.shp.roi_prj
        PosixPath('/Users/<user>/Library/Application Support/easyidp.data/data_for_tests/shp_test/roi.prj')

        >>> out_proj = idp.shp.read_proj(prj_path)
        >>> out_proj
        <Derived Projected CRS: EPSG:32654>
        Name: WGS 84 / UTM zone 54N
        Axis Info [cartesian]:
        - E[east]: Easting (metre)
        - N[north]: Northing (metre)
        Area of Use:
        - undefined
        Coordinate Operation:
        - name: UTM zone 54N
        - method: Transverse Mercator
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

    """
    with open(str(prj_path), "r") as f:
        wkt_string = f.readline()

    proj = pyproj.CRS.from_wkt(wkt_string)

    if proj.name == "WGS 84":
        proj = pyproj.CRS.from_epsg(4326)

    return proj


def show_shp_fields(shp_path, encoding="utf-8"):
    """
    Show geojson properties data, for better setting ``name_field`` of :py:obj:`read_shp <easyidp.roi.ROI.read_shp>`

    Parameters
    ----------
    shp_path : str
        the file path of \\*.shp
    encoding : str
        default is 'utf-8', however, or some chinese characters, 'gbk' is required

    Example
    -------

    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> idp.shp.show_shp_fields(test_data.shp.complex_shp, encoding="GBK")
          [-1]            [0] ID                [1] MASSIFID       [2] CROPTYPE    [3] CROPDATE    [4] CROPAREA    [5] ATTID
        ------  ---------------------------  -------------------  --------------  --------------  --------------  -----------
             0  230104112201809010000000000  2301041120000000000       小麦         2018-09-01     61525.26302
             1  230104112201809010000000012  2301041120000000012       蔬菜         2018-09-01      2802.33512
             2  230104112201809010000000014  2301041120000000014       玉米         2018-09-01      6960.7745
           ...              ...                      ...               ...             ...             ...            ...
           320  230104112201809010000000583  2301041120000000583       大豆         2018-09-01      380.41704
           321  230104112201809010000000584  2301041120000000584       其它         2018-09-01      9133.25998
           322  230104112201809010000000585  2301041120000000585       其它         2018-09-01      1704.27193

        >>> idp.shp.show_shp_fields(test_data.shp.lotus_shp)
        [-1] #   [0] plot_id
        ------  -------------
             0      N1W1
             1      N1W2
             2      N1W3
           ...       ...
           109      S4E5
           110      S4E6
           111      S4E7

    See also
    --------
    easyidp.jsonfile.show_geojson_fields
    """
    shp = shapefile.Reader(str(shp_path), encoding=encoding)

    # read shp file fields
    shp_fields = _get_field_key(shp)

    head = ["[-1] #"] + [f"[{v}] {k}" for k, v in shp_fields.items()]
    data = []

    row_num = len(shp.records())
    col_num = len(shp.records()[0])

    col_align = ["right"] + ["center"] * col_num

    if row_num > 6:
        show_idx = [0, 1, 2, -3, -2, -1]
    else:
        # print all without omit
        show_idx = list(range(row_num))

    for i in show_idx:
        if i >= 0:
            data.append([i] + list(shp.records()[i]))
        else:
            data.append([row_num + i] + list(shp.records()[i]))

    if row_num > 6:
        data.insert(3, ["..."] * (col_num + 1))

    table_str = tabulate(data, headers=head, tablefmt="simple", colalign=col_align)
    print(table_str)


def read_shp_field_schema(shp_path, encoding="utf-8"):
    """Read field schema from shapefile DBF header.

    Parameters
    ----------
    shp_path : str | pathlib.Path
        Path to source shapefile.
    encoding : str, optional
        Character encoding for DBF, by default ``"utf-8"``.

    Returns
    -------
    dict[str, tuple[str, int, int]]
        Field schema mapping in format
        ``{"FIELD": (field_type, field_size, decimal_size)}``.
    """
    shp = shapefile.Reader(str(shp_path), encoding=encoding)
    return _get_field_schema(shp)


def read_shp(
    shp_path,
    shp_proj=None,
    encoding="utf-8",
    return_proj=False,
):
    """read shp file to python numpy object

    Parameters
    ----------
    shp_path : str
        the file path of \\*.shp
    shp_proj : str | pyproj object
        by default None, will read automatically from prj file with the same name of shp filename,
        or give manually by ``read_shp(..., shp_proj=pyproj.CRS.from_epsg(4326), ...)`` or
        ``read_shp(..., shp_proj=r'path/to/{shp_name}.prj', ...)``
    encoding : str
        by default 'utf-8', for some chinese characters, 'gbk' may required
    return_proj : bool, optional
        by default False, if given as true, will return extra pyproj.CRS object of current shp file.

    Returns
    -------
    list[np.ndarray]
        Polygon coordinates for each shape in original order.
    list[dict]
        Attribute records of each shape in original order.
    dict
        Field map in format ``{"FIELD_NAME": int_id}``.

        .. code-block:: python

            {'id1': np.array([[x1,y1],[x2,y2],...]),
             'id2': np.array([[x1,y1],[x2,y2],...]),...}
    pyproj.CRS, optional
        once set return_proj=True

    Example
    -------

    The example shp file has the following columns:

    +--------------+--------------+----------------+----------------+----------------+-------------+
    | [0] ID       | [1] MASSIFID | [2] CROPTYPE   | [3] CROPDATE   | [4] CROPAREA   | [5] ATTID   |
    +==============+==============+================+================+================+=============+
    | 23010...0000 | 23010...0000 | 小麦           | 2018-09-01     | 61525.26302    |             |
    +--------------+--------------+----------------+----------------+----------------+-------------+
    | 23010...0012 | 23010...0012 | 蔬菜           | 2018-09-01     | 2802.33512     |             |
    +--------------+--------------+----------------+----------------+----------------+-------------+
    | 23010...0014 | 23010...0014 | 玉米           | 2018-09-01     | 6960.7745      |             |
    +--------------+--------------+----------------+----------------+----------------+-------------+
    | 23010...0061 | 23010...0061 | 牧草           | 2018-09-01     | 25349.08639    |             |
    +--------------+--------------+----------------+----------------+----------------+-------------+
    | 23010...0062 | 23010...0062 | 玉米           | 2018-09-01     | 71463.27666    |             |
    +--------------+--------------+----------------+----------------+----------------+-------------+
    | ...          | ...          | ...            | ...            | ...            | ...         |
    +--------------+--------------+----------------+----------------+----------------+-------------+
    | 23010...0582 | 23010...0582 | 胡萝卜         | 2018-09-01     | 288.23876      |             |
    +--------------+--------------+----------------+----------------+----------------+-------------+
    | 23010...0577 | 23010...0577 | 杂豆           | 2018-09-01     | 2001.80384     |             |
    +--------------+--------------+----------------+----------------+----------------+-------------+
    | 23010...0583 | 23010...0583 | 大豆           | 2018-09-01     | 380.41704      |             |
    +--------------+--------------+----------------+----------------+----------------+-------------+
    | 23010...0584 | 23010...0584 | 其它           | 2018-09-01     | 9133.25998     |             |
    +--------------+--------------+----------------+----------------+----------------+-------------+
    | 23010...0585 | 23010...0585 | 其它           | 2018-09-01     | 1704.27193     |             |
    +--------------+--------------+----------------+----------------+----------------+-------------+

    First, prepare data:

    .. code-block:: python

        >>> import easyidp as idp
        >>> testdata = idp.data.TestData()
        >>> data_path = testdata.shp.complex_shp

    Then read geometry and attributes:

    .. code-block:: python

        >>> polygons, records, fields = idp.shp.read_shp(data_path, encoding='gbk')
        >>> polygons[0]
        array([[ 45.83319255, 126.84383445],
               [ 45.83222256, 126.84212197],
               ...,
               [ 45.83321205, 126.84381378],
               [ 45.83319255, 126.84383445]])
        >>> records[0]
        {'ID': '230104112201809010000000000', 'MASSIFID': '2301041120000000000', ...}
        >>> fields
        {'ID': 0, 'MASSIFID': 1, 'CROPTYPE': 2, ...}

    See also
    --------
    easyidp.jsonfile.read_geojson

    """
    #####################################
    # check projection coordinate first #
    #####################################
    if shp_proj is None:
        prj_path = Path(shp_path).with_suffix(".prj")

        if Path(prj_path).exists():
            shp_proj = read_proj(prj_path)
        else:
            raise ValueError(
                f"Unable to find the proj coordinate info [{prj_path}], please either specify `shp_proj='path/to/{{shp_name}}.prj'` or `shp_proj=pyproj.CRS.from_epsg(xxxx)`"
            )
    # or give a prj file path
    elif (
        isinstance(shp_proj, (Path, str))
        and str(shp_proj)[-4:] == ".prj"
        and Path(shp_proj).exists
    ):
        shp_proj = read_proj(shp_proj)
    # or give a CRS projection object
    elif isinstance(shp_proj, pyproj.CRS):
        pass
    else:
        raise ValueError(
            f"Unable to find the projection coordinate, please either specify `shp_proj='path/to/{{shp_name}}.prj'` or `shp_proj=pyproj.CRS.from_epsg(xxxx)`"
        )

    print(
        f"[shp][proj] Use projection [{shp_proj.name}] for loaded shapefile [{Path(shp_path).name}]"
    )

    # read shapefile
    shp_data = shapefile.Reader(str(shp_path), encoding=encoding)

    # read shp file fields (headers)
    shp_fields = _get_field_key(shp_data)

    ########################
    # read shp coordinates #
    ########################
    polygons = []
    records = []

    # Use iterShapeRecords for better performance (O(N) vs O(N^2)) and memory usage
    pbar = tqdm(
        shp_data.iterShapeRecords(),
        total=len(shp_data),
        desc=f"[shp] Read shapefile [{os.path.basename(shp_path)}]",
    )
    for i, shape_record in enumerate(pbar):
        shape = shape_record.shape
        record = shape_record.record

        ##################################
        # get the shape coordinate value #
        ##################################
        coord_np = np.asarray(shape.points)
        # check if the last point == first point
        if (coord_np[0, :] != coord_np[-1, :]).all():
            # otherwise duplicate first point to last point to fit the polygon definition
            coord_np = np.append(coord_np, coord_np[0, :][None, :], axis=0)

        polygons.append(coord_np)
        record_dict = {
            field_name: record[fid] for field_name, fid in shp_fields.items()
        }
        records.append(record_dict)

    if return_proj:
        return polygons, records, shp_fields, shp_proj

    return polygons, records, shp_fields


def _get_field_key(shp):
    """
    Convert shapefile header {"Column": int_id}

    Parameters
    ----------
    shp : shapefile.Reader object
        shp = shapefile.Reader(shp_path, encoding=encoding)

    Returns
    -------
    dict
        Format: {"Column": int_id};
        Example: {"ID":0, "MASSIFID":1, "CROPTYPE":2, ...}

    Notes
    -----
    This function is compatible with both old and new versions of pyshp:
    - Old pyshp  <3.0.3: fields are list/tuple, e.g. ['plot_id', 'C', 80, 0]
    - New pyshp >=3.0.3: fields are Field namedtuple, e.g. Field(name='plot_id', ...)
    """
    shp_fields = {}
    f_count = 0
    for field in shp.fields:
        # Skip DeletionFlag field (first field)
        # In old pyshp: DeletionFlag is a tuple, other fields are lists
        # In new pyshp: all fields are Field namedtuples

        # Get field name - works for both list/tuple and namedtuple
        if hasattr(field, "name"):
            # New pyshp: Field namedtuple with 'name' attribute
            field_name = field.name
        else:
            # Old pyshp: list or tuple, first element is name
            field_name = field[0]

        # Skip DeletionFlag
        if field_name == "DeletionFlag":
            continue

        shp_fields[field_name] = f_count
        f_count += 1

    return shp_fields


def _get_field_schema(shp):
    """Read shapefile field schema.

    Parameters
    ----------
    shp : shapefile.Reader
        Opened shapefile reader object.

    Returns
    -------
    dict[str, tuple[str, int, int]]
        Field schema mapping as ``{name: (type, size, decimal)}``.
    """
    schema = {}
    for field in shp.fields:
        if hasattr(field, "name"):
            field_name = field.name
            field_type = field.field_type
            field_size = field.size
            decimal_size = field.decimal
        else:
            field_name = field[0]
            field_type = field[1]
            field_size = field[2]
            decimal_size = field[3]

        if field_name == "DeletionFlag":
            continue
        schema[field_name] = (field_type, field_size, decimal_size)
    return schema


def _infer_field_schema_from_attrs(attrs_rows):
    """Infer DBF field schema from attribute rows."""
    schema = {}
    for attrs in attrs_rows:
        for key, value in attrs.items():
            if key in schema:
                continue
            if isinstance(value, bool):
                schema[key] = ("L", 1, 0)
            elif isinstance(value, int):
                schema[key] = ("N", 18, 0)
            elif isinstance(value, float):
                schema[key] = ("F", 18, 8)
            else:
                schema[key] = ("C", 80, 0)
    return schema


def _build_record_values(field_order, attrs, name_field, key_name, subplot_values):
    """Build DBF row values by field order."""
    attrs[name_field] = key_name
    for meta_key, meta_value in subplot_values.items():
        attrs[meta_key] = meta_value
    return [attrs.get(field_name, None) for field_name in field_order]


def _find_name_related_int_id(shp_fields, name_field):
    """
    Inner function to get the number of given `name_field`.

    Parameters
    ----------
    shp_fields : dict
        the output of _get_field_key()
        Format: {"Column": int_id}
        Exmaple: {"ID":0, "MASSIFID":1, "CROPTYPE":2, ...}
    name_field : str or int or list[ str|int ],
        the id or name of shp file fields as output dictionary keys

    Returns
    -------
    field_id : int or list[ int ]

        For example:

        .. code-block:: python

            >>> a = {"ID":0, "MASSIFID":1, "CROPTYPE":2, ...}
            >>> b = "ID"
            >>> _find_name_related_int_id(a, b)
            0
            >>> c = ["ID", "CROPTYPE"]
            >>> _find_name_related_int_id(a, b)
            [0, 2]
    """
    if isinstance(name_field, int):
        if name_field >= len(shp_fields) or name_field < -1:
            raise IndexError(
                f"Int key [{name_field}] is outside the number of fields {shp_fields}"
            )
        field_id = name_field
    elif isinstance(name_field, str):
        if name_field == "#":
            field_id = -1
        else:
            if name_field not in shp_fields.keys():
                raise KeyError(f"Can not find key {name_field} in {shp_fields}")
            field_id = shp_fields[name_field]
    else:
        raise KeyError(f"Can not find key {name_field} in {shp_fields}")

    return field_id


def _get_plot_name_template(roi_fields, field_id, include_title):
    """
    Parameters
    ----------
    roi_fields : dict
        example: {"ID":0, "MASSIFID":1, "CROPTYPE":2, ...}
    field_id : int or list[ int ]
        the output of _find_name_related_int_id(), the column of property used for index
    include_title : bool, optional
        by default False, whether add column name to roi key.

    Returns
    -------
    plot_name : str
        >>> a = "{} {}"
        >>> a.format("hello", "world")
        'hello world'
    keyring : str or list[ str ]
        a variable to save key of geo_field:dict
    """

    def _fetch_single_field(roi_fields, field_id):
        plot_name_template = ""
        if field_id == -1:  # the row index
            _key = "#"
        else:
            _key = idp._find_key(roi_fields, field_id)

        if include_title:
            plot_name_template += _key + " {}"
        else:
            plot_name_template += "{}"

        return plot_name_template, _key

    if isinstance(field_id, list):
        plot_name_template = ""
        keyring = []
        for j, fid in enumerate(field_id):
            _plot_name_template, _key = _fetch_single_field(roi_fields, fid)

            plot_name_template += _plot_name_template
            keyring.append(_key)

            # not adding the last key A|B|C| --> A|B|C
            if j < len(field_id) - 1:
                plot_name_template += "|"
    else:
        plot_name_template, keyring = _fetch_single_field(roi_fields, field_id)

    return plot_name_template, keyring


def _crs_to_wkt(crs, wkt_version=1):
    """Convert CRS to requested WKT version string.

    Parameters
    ----------
    crs : pyproj.CRS
        Input CRS object.
    wkt_version : int, optional
        WKT version selector: 1 for WKT1_ESRI, 2 for WKT2_2019.

    Returns
    -------
    str
        WKT string for .prj file output.

    Raises
    ------
    ValueError
        If wkt_version is not 1 or 2.
    """
    if wkt_version == 1:
        return crs.to_wkt(version=pyproj.enums.WktVersion.WKT1_ESRI)
    if wkt_version == 2:
        return crs.to_wkt(version=pyproj.enums.WktVersion.WKT2_2019)
    raise ValueError(f"wkt_version must be 1 or 2, got {wkt_version}")


def write_shp(
    shp_path,
    roi_dict,
    crs=None,
    name_field="id",
    encoding="utf-8",
    subplot_meta=None,
    attrs_rows=None,
    field_schema=None,
    wkt_version=1,
):
    """Save ROI polygons to shapefile.

    Parameters
    ----------
    shp_path : str | pathlib.Path
        Output shapefile path (with or without .shp extension).
    roi_dict : dict or idp.ROI
        Dictionary of polygons or ROI object.
    crs : pyproj.CRS, optional
        Coordinate reference system.
    name_field : str, optional
        Name of the attribute field for polygon names, by default 'id'.
    encoding : str, optional
        Character encoding for the shapefile, by default 'utf-8'.
    subplot_meta : dict, optional
        Metadata for subplots (row, col, status) if available.
    attrs_rows : list[dict], optional
        Attribute rows aligned with ROI order.
    field_schema : dict[str, tuple[str, int, int]], optional
        DBF field schema mapping, usually from source shapefile.
    wkt_version : int, optional
        WKT version for PRJ output (1 for WKT1_ESRI, 2 for WKT2_2019),
        by default 1.

    Returns
    -------
    pathlib.Path
        Path to the saved shapefile.

    Raises
    ------
    ValueError
        If roi_dict is empty.
    """
    if len(roi_dict) == 0:
        raise ValueError("Cannot save empty ROI to shapefile")

    if attrs_rows is not None and len(attrs_rows) != len(roi_dict):
        raise ValueError("Length of attrs_rows must match the number of ROI polygons")

    shp_path = Path(shp_path)
    if shp_path.suffix.lower() != ".shp":
        shp_path = shp_path.with_suffix(".shp")

    if attrs_rows is None:
        field_schema_out = {name_field: ("C", 80, 0)}
    elif field_schema is None:
        field_schema_out = _infer_field_schema_from_attrs(attrs_rows)
    else:
        field_schema_out = dict(field_schema)

    if name_field not in field_schema_out:
        field_schema_out[name_field] = ("C", 80, 0)

    if subplot_meta:
        field_schema_out.setdefault("row", ("N", 18, 0))
        field_schema_out.setdefault("col", ("N", 18, 0))
        field_schema_out.setdefault("status", ("C", 20, 0))

    field_order = list(field_schema_out.keys())

    # Create shapefile writer
    with shapefile.Writer(str(shp_path), encoding=encoding) as w:
        for field_name, schema in field_schema_out.items():
            field_type, field_size, decimal_size = schema
            w.field(field_name, field_type, field_size, decimal_size)

        # Write each polygon
        for idx, name in enumerate(roi_dict.keys()):
            coords = roi_dict[name]

            # Ensure 2D coordinates for shapefile
            # Standardizing input: coords might be numpy array
            if isinstance(coords, np.ndarray):
                if coords.shape[1] >= 2:
                    poly_coords = coords[:, :2].tolist()
                else:
                    poly_coords = coords.tolist()
            else:
                poly_coords = coords  # assume list

            # Write polygon geometry
            w.poly([poly_coords])

            if attrs_rows is None:
                attrs = {}
            else:
                attrs = dict(attrs_rows[idx])

            subplot_values = {}
            if subplot_meta and name in subplot_meta:
                meta = subplot_meta[name]
                subplot_values = {
                    "row": meta["row"],
                    "col": meta["col"],
                    "status": meta["status"],
                }

            record_values = _build_record_values(
                field_order,
                attrs,
                name_field,
                name,
                subplot_values,
            )
            w.record(*record_values)

    # Write .prj file if CRS is available
    if crs is not None:
        prj_path = shp_path.with_suffix(".prj")
        prj_path.write_text(_crs_to_wkt(crs, wkt_version=wkt_version))
        logger.debug(f"Saved projection file to {prj_path}")

    logger.info(f"Saved {len(roi_dict)} polygons to {shp_path}")

    return shp_path
