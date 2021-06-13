# Copyright 2017 Virgil Dupras

# This software is licensed under the "BSD" License as described in the "LICENSE" file,
# which should be included with this package. The terms are also available at
# http://www.hardcoded.net/licenses/bsd_license

from __future__ import unicode_literals
import os.path as op
from .compat import text_type
from ctypes import (
    windll,
    Structure,
    byref,
    c_uint,
    create_unicode_buffer,
    addressof,
    GetLastError,
    FormatError,
)
from ctypes.wintypes import HWND, UINT, LPCWSTR, BOOL

kernel32 = windll.kernel32
GetShortPathNameW = kernel32.GetShortPathNameW

shell32 = windll.shell32
SHFileOperationW = shell32.SHFileOperationW


class SHFILEOPSTRUCTW(Structure):
    _fields_ = [
        ("hwnd", HWND),
        ("wFunc", UINT),
        ("pFrom", LPCWSTR),
        ("pTo", LPCWSTR),
        ("fFlags", c_uint),
        ("fAnyOperationsAborted", BOOL),
        ("hNameMappings", c_uint),
        ("lpszProgressTitle", LPCWSTR),
    ]


FO_MOVE = 1
FO_COPY = 2
FO_DELETE = 3
FO_RENAME = 4

FOF_MULTIDESTFILES = 1
FOF_SILENT = 4
FOF_NOCONFIRMATION = 16
FOF_ALLOWUNDO = 64
FOF_NOERRORUI = 1024


def get_short_path_name(long_name):
    if not long_name.startswith("\\\\?\\"):
        long_name = "\\\\?\\" + long_name
    buf_size = GetShortPathNameW(long_name, None, 0)
    # FIX: https://github.com/hsoft/send2trash/issues/31
    # If buffer size is zero, an error has occurred.
    if not buf_size:
        err_no = GetLastError()
        raise WindowsError(err_no, FormatError(err_no), long_name[4:])
    output = create_unicode_buffer(buf_size)
    GetShortPathNameW(long_name, output, buf_size)
    return output.value[4:]  # Remove '\\?\' for SHFileOperationW


def send2trash(paths):
    if not isinstance(paths, list):
        paths = [paths]
    # convert data type
    paths = [
        text_type(path, "mbcs") if not isinstance(path, text_type) else path
        for path in paths
    ]
    # convert to full paths
    paths = [op.abspath(path) if not op.isabs(path) else path for path in paths]
    # get short path to handle path length issues
    paths = [get_short_path_name(path) for path in paths]
    # convert to a single string of null terminated paths
    paths = "\0".join(paths)
    fileop = SHFILEOPSTRUCTW()
    fileop.hwnd = 0
    fileop.wFunc = FO_DELETE
    # FIX: https://github.com/hsoft/send2trash/issues/17
    # Starting in python 3.6.3 it is no longer possible to use:
    # LPCWSTR(path + '\0') directly as embedded null characters are no longer
    # allowed in strings
    # Workaround
    #  - create buffer of c_wchar[] (LPCWSTR is based on this type)
    #  - buffer is two c_wchar characters longer (double null terminator)
    #  - cast the address of the buffer to a LPCWSTR
    # NOTE: based on how python allocates memory for these types they should
    # always be zero, if this is ever not true we can go back to explicitly
    # setting the last two characters to null using buffer[index] = '\0'.
    buffer = create_unicode_buffer(paths, len(paths) + 2)
    fileop.pFrom = LPCWSTR(addressof(buffer))
    fileop.pTo = None
    fileop.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_NOERRORUI | FOF_SILENT
    fileop.fAnyOperationsAborted = 0
    fileop.hNameMappings = 0
    fileop.lpszProgressTitle = None
    result = SHFileOperationW(byref(fileop))
    if result:
        raise WindowsError(result, FormatError(result), paths)
