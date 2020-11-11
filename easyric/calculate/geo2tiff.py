import os
import numpy as np
from easyric.io import shp, geotiff, json
from easyric.external.send2trash import send2trash
from skimage.io import imsave


def shp_clip_geotiff(shp_path, geotiff_path, out_folder, refresh_folder=False, shp_proj=None, geotiff_proj=None):
    # [Todo] Export DSM still uint8 loss many accuracy
    if geotiff_proj is None:
        header = geotiff.get_header(geotiff_path)
        shp_dict = shp.read_shp2d(shp_path, shp_proj=shp_proj, geotiff_proj=header['proj'])
    else:
        shp_dict = shp.read_shp2d(shp_path, shp_proj=shp_proj, geotiff_proj=geotiff_proj)

    shp_names = list(shp_dict.keys())
    shp_poly = list(shp_dict.values())
    imarray_list, offset_list = geotiff.clip_roi(shp_poly, geotiff_path, is_geo=True)

    offset_dict = {}
    return_dict = {}
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    else:
        if refresh_folder:
            send2trash(out_folder)
            os.makedirs(out_folder)

    i = 1
    for imarray, offset, shp_name in zip(imarray_list, offset_list, shp_names):
        print(f"[calcu][geo2tiff] {shp_name}, {i} of {len(shp_names)}", end='\r')

        if len(imarray.shape) == 2:
            save_name = f"{shp_name}.tif"
            imsave(f'{out_folder}/{save_name}', imarray)
        else:
            save_name = f"{shp_name}.png"
            imsave(f'{out_folder}/{save_name}', imarray.astype(np.uint8))


        offset_dict[save_name] = offset.tolist()
        return_dict[shp_name] = {'offset': offset, 'imarray': imarray.astype(np.uint8)}

        i += 1

    if len(imarray.shape) == 2:
        json.dict2json(offset_dict, f"{out_folder}/offsets_dsm.json")
    else:
        json.dict2json(offset_dict, f"{out_folder}/offsets_dom.json")

    return return_dict