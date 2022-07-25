__version__ = "2.0.0.dev2"

import os
import warnings

##############
# dict tools #
##############

class Container(dict):
    """Self designed dictionary like classto contain items like {"id": item.label}
    but enable using both [item.id] and [item.label] to fetch items
    https://stackoverflow.com/questions/4014621/a-python-class-that-acts-like-dict
    """
    # 

    def __init__(self):
        super().__init__()
        self.id_item = {}   # {0: item1, 1: item2}
        self.item_label = {}  #{"N1W1": 0, "N1W2": 1}, just index it position

    def __setitem__(self, key, item):
        if isinstance(key, int):
            # default method to set values
            # Container[0] = A, while A.label = "N1W1"
            self.id_item[key] = item
            self.item_label[item.label] = key
        elif isinstance(key, str):
            # advanced method to change items
            # Container["N1W1"] = B, here assuemt B.label already == "N1W1"

            # item already exists
            if key in self.item_label.keys():   
                idx = self.item_label[key]
                self.id_item[idx] = item
            else:  # add new item
                idx = len(self.id_item)
                self.id_item[idx] = item
                if 'label' in dir(item):  # has item.label
                    self.item_label[item.label] = idx
                else:  # act as common dictionary
                    self.item_label[key] = idx
        else:
            raise KeyError(f"Key should be 'int', 'str', 'slice', not {key}")

    def __getitem__(self, key):
        if isinstance(key, int):  # index by photo order
            return self.id_item[key]
        elif isinstance(key, str):  # index by photo name
            return self.id_item[self.item_label[key]]
        # elif isinstance(key, slice):
        #     return list(self.id_item.values())[key]
        else:
            raise KeyError(f"Key should be 'int', 'str', 'slice', not {key}")

    def __repr__(self):
        return repr(self.id_item)

    def __len__(self):
        return len(self.id_item)

    def __delitem__(self, key):
        if isinstance(key, int):
            k = key
            del self.item_label[self.id_item[key]]
            del self.id_item[key]
        elif isinstance(key, str):
            k = self.item_label[key]
            del self.id_item[self.item_label[key]]
            del self.item_label[key]
        else:
            raise KeyError(f"Key should be 'int', 'str', 'slice', not {key}")

        # update the id
        # a[5] = a.pop(1)
        # https://stackoverflow.com/questions/4406501/change-the-name-of-a-key-in-dictionary
        id_item_keys = list(self.id_item.keys())
        for idx in id_item_keys:
            # e,g. k = 3, idx in [0, 1, 2, 4, 5]
            if idx > k:
                self.id_item[idx-1] = self.id_item.pop(idx)

                # e.g. {"N1W1": 0, "N1W2": 1},
                label = _find_key(self.item_label, idx)
                self.item_label[label] = idx - 1

    def __iter__(self):
        return iter(self.id_item.values())

    def keys(self):
        return self.item_label.keys()

    def values(self):
        return self.id_item.values()

    def items(self):
        out_dict = {}
        for k, idx in self.item_label.items():
            out_dict[k] = self.id_item[idx]
        return out_dict.items()


def _find_key(mydict, value):
    """a simple function to using dict value to find key
    e.g. 
    >>> mydict = {"a": 233, "b": 456}
    >>> _find_key(mydict, 233)
    "a"
    """
    key_idx = list(mydict.values()).index(value)
    return list(mydict.keys())[key_idx]


##############
# path tools #
##############

def get_full_path(short_path):
    if isinstance(short_path, str):
        return os.path.abspath(os.path.normpath(short_path))
    else:
        return None

def parse_relative_path(root_path, relative_path):
    # for metashape frame.zip path use only
    if r"../../" in relative_path:
        frame_path = os.path.dirname(os.path.abspath(root_path))
        merge = os.path.join(frame_path, relative_path)
        return os.path.abspath(merge)
    else:
        warnings.warn(f"Seems it is an absolute path [{relative_path}]")
        return relative_path

###############
# import APIs #
###############

from . import (
    visualize, 
    shp, 
    jsonfile, 
    cvtools
)
from .reconstruct import ProjectPool
from .pointcloud import PointCloud
from .geotiff import GeoTiff
from .pix4d import Pix4D
from .metashape import Metashape
from .roi import ROI