
from pathlib import Path
import time
from tqdm import tqdm

import easyidp as idp

if __name__ == "__main__":
    start_time = time.time()
    # Define paths
    shp_path       = Path('data/grid_all/grid_32.shp')
    tiepoints_path = Path('data/metashape_project/large_tie_pointsrice.las')
    metashape_path = Path('data/metashape_project/large_rice.psx')
    save_path      = Path('outputs/final_dataset_fast')

    rgb_full_out = save_path / "images_demo"

    if not rgb_full_out.exists():
        rgb_full_out.mkdir(parents=True)
    
    # Initialize ROI and get Z coordinates
    roi = idp.ROI(shp_path, name_field="fid")
    roi.get_z_from_pcd(tiepoints_path, mode="face", kernel="mean", buffer=15)
    
    # Initialize Metashape and set CRS
    ms = idp.Metashape(metashape_path, chunk_id=0)
    ms.crs = roi.crs
    
    # Back project to raw images
    img_dict = ms.back2raw(roi)
    img_dict_sort = ms.sort_img_by_distance(
        img_dict, roi, 
        distance_thresh=9999, 
        num=1
    )

    gtif_dict = idp.geotiff.back2raw2geotiff(
        ms,
        back2raw_result=img_dict_sort,
        roi=roi,
        nodata=-32768,
        has_alpha=True,
        use_affine=False,  # rotate geotiff
    )

    # # save to geotiff with affine
    for roi_key, value in tqdm(gtif_dict.items(), desc="Saving Full Maps"):

        for img_key, gtif_item in value.items():

            gtif_item.save( rgb_full_out / f"grid_{roi_key}.tif", use_affine=False, overwrite=True)

            break

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")