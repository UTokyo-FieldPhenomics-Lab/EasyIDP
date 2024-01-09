import os
import pyproj
import shapefile
import numpy as np
import warnings
from tabulate import tabulate
from tqdm import tqdm
from pathlib import Path

import easyidp as idp


def read_proj(prj_path):
    """read \*.prj file to pyproj object
    
    Parameters
    ----------
    prj_path : str
        the file path of shp \*.prj
    
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
    with open(str(prj_path), 'r') as f:
        wkt_string = f.readline()

    proj = pyproj.CRS.from_wkt(wkt_string)
    
    if proj.name == 'WGS 84':
        proj = pyproj.CRS.from_epsg(4326)

    return proj


def show_shp_fields(shp_path, encoding="utf-8"):
    """
    Show geojson properties data, for better setting ``name_field`` of :py:obj:`read_shp <easyidp.roi.ROI.read_shp>`
    
    Parameters
    ----------
    shp_path : str
        the file path of \*.shp
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
          [-1]   [0] plot_id
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

    head = ["[-1]"] + [f"[{v}] {k}" for k, v in shp_fields.items()]
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
        data.insert(3, ['...'] * (col_num + 1))

    table_str = tabulate(data, headers=head, tablefmt='simple', colalign=col_align)
    print(table_str)


def read_shp(shp_path, shp_proj=None, name_field=None, include_title=False, encoding='utf-8', return_proj=False):
    """read shp file to python numpy object
    
    Parameters
    ----------
    shp_path : str
        the file path of \*.shp
    shp_proj : str | pyproj object
        by default None, will read automatically from prj file with the same name of shp filename, 
        or give manually by ``read_shp(..., shp_proj=pyproj.CRS.from_epsg(4326), ...)`` or 
        ``read_shp(..., shp_proj=r'path/to/{shp_name}.prj', ...)``
    name_field : str or int or list[ str|int ], optional
        by default None, the id or name of shp file fields as output dictionary keys
    include_title : bool, optional
        by default False, whether add column name to roi key.
    encoding : str
        by default 'utf-8', for some chinese characters, 'gbk' may required
    return_proj : bool, optional
        by default False, if given as true, will return extra pyproj.CRS object of current shp file.
    
    Returns
    -------
    dict, 
        the dictionary with read numpy polygon coordinates

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

    First, prepare data

    .. code-block:: python

        >>> import easyidp as idp
        >>> testdata = idp.data.TestData()
        >>> data_path = testdata.shp.complex_shp

    Then using the second column ``MASSIFID`` as shape keys:

    .. code-block:: python

        >>> out = idp.shp.read_shp(data_path, name_field="MASSIFID", encoding='gbk')
        >>> # or 
        >>> out = idp.shp.read_shp(data_path, name_field=1, encoding='gbk')
        [shp][proj] Use projection [WGS 84] for loaded shapefile [complex_shp_review.shp]
        [shp] read shp [complex_shp_review.shp]: 100%|███████████| 323/323 [00:02<00:00, 143.13it/s] 
        >>> out['23010...0000'] 
        array([[ 45.83319255, 126.84383445],
               [ 45.83222256, 126.84212197],
               ...
               [ 45.83321205, 126.84381378],
               [ 45.83319255, 126.84383445]])

    Due to the duplication of ``CROPTYPE``, you can not using it as the unique key, but you can combine several columns together by passing a list to ``name_field``:

    .. code-block:: python

        >>> out = idp.shp.read_shp(data_path, name_field=["CROPTYPE", "MASSIFID"], encoding='gbk') 
        >>> # or
        >>> out = idp.shp.read_shp(data_path, name_field=[2, 1], include_title=True, encoding='gbk') 
        [shp][proj] Use projection [WGS 84] for loaded shapefile [complex_shp_review.shp]
        [shp] read shp [complex_shp_review.shp]: 100%|███████████| 323/323 [00:02<00:00, 143.13it/s] 
        >>> out.keys()
        dict_keys(['小麦_23010...0000', '蔬菜_23010...0012', '玉米_23010...0014', ... ])

    And you can also add column_names to id by ``include_title=True`` :

    .. code-block:: python

        >>> out = idp.shp.read_shp(data_path, name_field=["CROPTYPE", "MASSIFID"], include_title=True, encoding='gbk') 
        >>> out.keys()
        dict_keys(['CROPTYPE_小麦_MASSIFID_23010...0000', 'CROPTYPE_蔬菜_MASSIFID_23010...0012', ... ])

    See also
    --------
    easyidp.jsonfile.read_geojson

    """
    #####################################
    # check projection coordinate first #
    #####################################
    if shp_proj is None:
        prj_path = Path(shp_path).with_suffix('.prj')

        if Path(prj_path).exists():
            shp_proj = read_proj(prj_path)
        else:
            raise ValueError(f"Unable to find the proj coordinate info [{prj_path}], please either specify `shp_proj='path/to/{{shp_name}}.prj'` or `shp_proj=pyproj.CRS.from_epsg(xxxx)`")
    # or give a prj file path
    elif isinstance(shp_proj, (Path, str)) and str(shp_proj)[-4:]=='.prj' and Path(shp_proj).exists:
        shp_proj = read_proj(shp_proj)
    # or give a CRS projection object
    elif isinstance(shp_proj, pyproj.CRS):
        pass
    else:
        raise ValueError(f"Unable to find the projection coordinate, please either specify `shp_proj='path/to/{{shp_name}}.prj'` or `shp_proj=pyproj.CRS.from_epsg(xxxx)`")

    print(f'[shp][proj] Use projection [{shp_proj.name}] for loaded shapefile [{Path(shp_path).name}]')

    # read shapefile
    shp_data = shapefile.Reader(str(shp_path), encoding=encoding)
    
    # read shp file fields (headers)
    shp_fields = _get_field_key(shp_data)

    ########################
    # read shp coordinates #
    ########################
    shp_dict = {}

    ### do not put it into the following loop, save calculation time.
    if isinstance(name_field, list):
        field_id = [_find_name_related_int_id(shp_fields, nf) for nf in name_field]
    else:
        field_id = _find_name_related_int_id(shp_fields, name_field)

    # build the format template
    plot_name_template, keyring = idp.shp._get_plot_name_template(
        shp_fields, field_id, include_title
    )
    ### the ``keyring`` only for dict like object,
    ### but shp.shapes() is not dict, so not useable
    ### keyring designed for read_geojson function in jsonfile.py

    pbar = tqdm(
        shp_data.shapes(), 
        desc=f"[shp] Read shapefile [{os.path.basename(shp_path)}]"
    )
    for i, shape in enumerate(pbar):
        # convert dict_key name string by given name_field
        if isinstance(field_id, list):
            values = [shp_data.records()[i][fid] for fid in field_id]
            plot_name = plot_name_template.format(*values)
        elif field_id is None:
            plot_name = plot_name_template.format(i)
        else:
            plot_name = plot_name_template.format(shp_data.records()[i][field_id])

        plot_name = plot_name.replace(r'/', '_')
        plot_name = plot_name.replace(r'\\', '_')

        ##################################
        # get the shape coordinate value #
        ##################################
        coord_np = np.asarray(shape.points)
        # check if the last point == first point
        if (coord_np[0, :] != coord_np[-1, :]).all():
            # otherwise duplicate first point to last point to fit the polygon definition
            coord_np = np.append(coord_np, coord_np[0,:][None,:], axis = 0)

        # check if has duplicated key, otherwise will cause override
        if plot_name in shp_dict.keys():
            raise KeyError(f"Meet with duplicated key [{plot_name}] for current shapefile, please specify another `name_field` from {shp_fields} or simple leave it blank `name_field=None`")

        shp_dict[plot_name] = coord_np

    if return_proj:
        return shp_dict, shp_proj
    else:
        return shp_dict


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
    """
    shp_fields = {}
    f_count = 0
    for l in shp.fields:
        if isinstance(l, list):
            '''
            the fields 0 -> delection flags, and is a tuple type, ignore this tag
            [('DeletionFlag', 'C', 1, 0),
             ['ID', 'C', 36, 0],
             ['MASSIFID', 'C', 19, 0],
             ['CROPTYPE', 'C', 36, 0],
             ['CROPDATE', 'D', 8, 0],
             ['CROPAREA', 'N', 13, 5],
             ['ATTID', 'C', 36, 0]]
            '''
            shp_fields[l[0]] = f_count
            f_count += 1

    return shp_fields
    
    
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
    if name_field is None:
        field_id = None
        warnings.warn(
            "Not specifying parameter 'name_field', will using the row id (from 0 to end) as the index for each polygon."\
            "Please using idp.shp.show_shp_field(shp_path) to display the full available indexs")
    elif isinstance(name_field, int):
        if name_field >= len(shp_fields):
            raise IndexError(f'Int key [{name_field}] is outside the number of fields {shp_fields}')
        field_id = name_field
    elif isinstance(name_field, str):
        if name_field not in shp_fields.keys():
            raise KeyError(f'Can not find key {name_field} in {shp_fields}')
        field_id = shp_fields[name_field]
    else:
        raise KeyError(f'Can not find key {name_field} in {shp_fields}')
    
    return field_id

def _get_plot_name_template(roi_fields: dict, field_id:int | list, include_title:bool):
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
    if isinstance(field_id, list):
        plot_name_template = ""
        keyring = []
        for j, fid in enumerate(field_id):
            _key = idp._find_key(roi_fields, fid)
            keyring.append(_key)
            if include_title:
                plot_name_template +=  _key + "_{}"
            else:
                plot_name_template += "{}"

            # not adding the last key A_B_C_ --> A_B_C
            if j < len(field_id)-1:
                plot_name_template += "_"
            
    elif field_id is None:
        keyring = None
        if include_title:
            plot_name_template = "line_{}"
        else:
            plot_name_template = "{}"
    else:
        keyring = idp._find_key(roi_fields, field_id)
        if include_title:
            plot_name_template =  keyring + "_{}"
        else:
            plot_name_template = "{}"

    return plot_name_template, keyring