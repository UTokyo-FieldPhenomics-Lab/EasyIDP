import os
import pyproj
import shapefile
import numpy as np
from tabulate import tabulate


def show_shp_fields(shp_path, encoding="utf-8", show_num=5):
    """
    Read shp field data to pandas.DataFrame, for further json metadata usage
    
    Parameters
    ----------
    shp_path: str, the file path of *.shp
    encoding: str, default is 'utf-8', however, or some chinese characters, 'gbk' is required
    
    Returns
    -------
    field_df: pandas.DataFrame, all shp field records
    
    """
    shp = shapefile.Reader(shp_path, encoding=encoding)

    keys = {}
    for i, l in enumerate(shp.fields):
        if isinstance(l, list):
            keys[l[0]] = i

    header_str_list = [f"[{v}] {k}" for k, v in keys.items()]

    content_str_list = []
    if len(shp.records()) > show_num * 2:
        # first {show_num} rows
        for i in range(0, show_num):
            content_str_list.append(list(shp.records()[i]))

        # omitted row
        omit_list = ['...'] * len(shp.records()[i])
        content_str_list.append(omit_list)

        # last {show_num} rows
        for i in range(-show_num, 0):
            content_str_list.append(list(shp.records()[i]))
    else:
        # print all rows without omit
        for i in shp.records():
            content_str_list.append(list(i))

    table_str = tabulate(content_str_list, header_str_list, tablefmt="github")
    print(table_str)