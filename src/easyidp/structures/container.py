"""Container data structure used across EasyIDP."""

import os
from copy import deepcopy

import numpy as np


def _find_key(mydict, value):
    """Find key in dict by value.

    Parameters
    ----------
    mydict : dict
        Input dictionary.
    value : Any
        Target value to find.

    Returns
    -------
    Any
        The key that maps to ``value``.
    """
    key_idx = list(mydict.values()).index(value)
    return list(mydict.keys())[key_idx]


class Container(dict):
    """Self designed dictionary class to fetch items by id or label.

    Caution
    -------
    This object can not be saved by ``pickle``, it will cause problem when
    loading [1]_.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/4014621/
       a-python-class-that-acts-like-dict
    """

    def __init__(self, suffix=""):
        super().__init__()
        self.id_item = {}
        self.item_label = {}
        self._suffix = str(suffix)

    def __setitem__(self, key, item):
        if isinstance(key, int):
            if key < len(self.item_label):
                if "label" in dir(item):
                    old_key = self.id_item[key].label
                    del self.item_label[old_key]
                    self.item_label[item.label] = key
                self.id_item[key] = item
            elif key == len(self.item_label):
                self.id_item[key] = item
                if "label" in dir(item):
                    if item.label in self.item_label.keys():
                        raise KeyError(
                            "The given item's label "
                            f"[{item.label}] already exists -> "
                            f"{self.item_label.keys()}"
                        )
                    self.item_label[item.label] = key
                else:
                    self.item_label[key] = key
            else:
                raise IndexError(
                    f"Index [{key}] out of range (0, {len(self.item_label)})"
                )

        elif isinstance(key, str):
            if key in self.item_label.keys():
                idx = self.item_label[key]
                self.id_item[idx] = item
            else:
                idx = len(self.id_item)
                self.id_item[idx] = item
                if "label" in dir(item):
                    self.item_label[item.label] = idx
                else:
                    self.item_label[key] = idx
        else:
            raise KeyError(f"Key should be 'int', 'str', not {key}")

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < len(self.item_label):
                return self.id_item[key]
            raise IndexError(f"Index [{key}] out of range (0, {len(self.item_label)})")
        if isinstance(key, str):
            if key in self.item_label.keys():
                return self.id_item[self.item_label[key]]
            if (
                self._suffix in key
                and os.path.splitext(key)[0] in self.item_label.keys()
            ):
                return self.id_item[self.item_label[os.path.splitext(key)[0]]]
            raise KeyError(f"Can not find key [{key}]")
        if isinstance(key, slice):
            idx_list = list(self.id_item.keys())[key]
            out = self.copy()
            out.id_item = {k: v for k, v in self.id_item.items() if k in idx_list}
            out.item_label = {k: v for k, v in self.item_label.items() if v in idx_list}
            return out
        raise KeyError(f"Key should be 'int', 'str', 'slice', not {key}")

    def __repr__(self) -> str:
        return self._btf_print()

    def __str__(self) -> str:
        return self._btf_print()

    def _btf_print(self):
        title = f"easyidp.{self.__class__.__name__}"
        key_list = list(self.item_label.keys())
        num = len(key_list)
        out_str = f"<{title}> with {num} items\n"

        default_np_thresh = np.get_printoptions()["threshold"]
        default_np_suppress = np.get_printoptions()["suppress"]
        np.set_printoptions(threshold=4, suppress=True)
        if num == 0:
            out_str = "<Empty easyidp.Container object>"
        elif num > 5:
            for i, k in enumerate(key_list[:2]):
                out_str += f"[{i}]\t{k}\n"
                out_str += repr(self.id_item[self.item_label[k]])
                out_str += "\n"
            out_str += "...\n"
            for i, k in enumerate(key_list[-2:]):
                out_str += f"[{num - 2 + i}]\t{k}\n"
                out_str += repr(self.id_item[self.item_label[k]])
                out_str += "\n"
        else:
            for i, k in enumerate(key_list):
                out_str += f"[{i}]\t{k}\n"
                out_str += repr(self.id_item[self.item_label[k]])
                out_str += "\n"
        np.set_printoptions(
            threshold=default_np_thresh,
            suppress=default_np_suppress,
        )

        out_str = out_str[:-1]
        return out_str

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

        id_item_keys = list(self.id_item.keys())
        for idx in id_item_keys:
            if idx > k:
                self.id_item[idx - 1] = self.id_item.pop(idx)
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

    def copy(self):
        return deepcopy(self)
