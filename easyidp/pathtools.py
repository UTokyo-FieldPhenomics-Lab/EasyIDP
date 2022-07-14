import os
import warnings

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