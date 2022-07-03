<div align="center">

<p>
   <!-- <a align="left" href="https://ultralytics.com/yolov5" target="_blank"> -->
   <img width="850" src="https://github.com/HowcanoeWang/EasyIDP/wiki/static/easyidp_head.svg"></a>
</p>

<p align="center">
  <img alt="GitHub code size in bytes" src="https://img.shields.io/tokei/lines/github/UTokyo-FieldPhenomics-Lab/EasyIDP?style=plastic">
  <img alt="GitHub" src="https://img.shields.io/github/license/UTokyo-FieldPhenomics-Lab/EasyIDP?style=plastic">
  <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/UTokyo-FieldPhenomics-Lab/EasyIDP?style=plastic">
</p>

<a href="README_CN.md">中文</a>

</div>

EasyIDP (Easy Intermediate Data Processor), A handy tool for dealing with region of interest (ROI) on the image reconstruction (Metashape & Pix4D) outputs, mainly in agriculture applications. It provides the following functions: 

1. Clip ROI on GeoTiff Maps (DOM & DSM) and Point Cloud.
2. Backward Projection ROI to original images.

This project tried to use packges based on pure-python, instead of installing some heavy packages (Open3D, OpenCV) and hard to install packages (GDAL dependices) for one or two individual functions. This may cause efficiency loss and differences in coding habit.

## <div align="center">Documentation</div>

Please check [Official Documents](https://easyidp.readthedocs.io/en/latest/) for full documentations. And please also use the [Github Discussion](https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP/discussions) when you meet any problems.


## <div align="center">Quick Start Examples (Processing)</div>

You can install the packages by PyPi:

```bash
pip install easyidp
```

And import the packages in your python code:

```python
import easyidp as idp
```

---

Before doing the following example, please understand the basic pipeline for image 3D reconstruction by Pix4D or Metashape. And know how to export the DOM, DSM (\*.tiff), and Point cloud (\*.ply). Also require some basic knowledge about GIS shapefile format (\*.shp).

> Please note, if you see this sentence, it means the following examples are not suppported yet.

<details close>
<summary>1. Read ROI</summary>

```python
roi = idp.ROI("xxxx.shp")  # lon and lat 2D info
  
# get z values from DSM
roi.get_z_from_dsm("xxxx_dsm.tiff")  # add height 3D info
```

The 2D roi can be used to clip the DOM, DSM, and point cloud (`2.Clip by ROI`). While the 3D roi can be used for Backward projection (`4. Backward projection`)
</details>

<details close>
<summary>2. Clip by ROI</summary>

Read the DOM and DSM Geotiff Maps
```python
dom = idp.GeoTiff("xxx_dom.tif")
dsm = idp.GeoTiff("xxx_dsm.tif")
```
  
Read point cloud data
```python
ply = idp.PointCloud("xxx_pcd.ply")
```
  
Clip the region of interest from ROI:
```python
dom_parts = roi.clip(dom)
dsm_parts = roi.clip(dsm)
pcd_parts = roi.clip(ply)
```

If you want to save these clips to given folder:
```python
dom_parts = roi.clip(dom, save_folder="./clip_dom")
dsm_parts = roi.clip(dsm, save_folder="./clip_dsm")
pcd_parts = roi.clip(ply, save_folder="./clip_pcd")
```

  
</details>

<details close>
<summary>3. Read Reconstruction projects</summary>

Add the reconstruction projects to processing pools (different flight time for the same field):
  
```python
proj = idp.ProjectPool()
proj.add_pix4d(["date1.p4d", "date2.p4d", ...])
proj.add_metashape(["date1.psx", "date2.psx", ...])
```

Then you can specify each chunk by:

```python
p1 = proj[0]
# or
p1 = proj["chunk_or_project_name"]
```

</details>

<details close>
<summary>4. Backward Projection</summary>
  
```python
>>> img_dict = roi.back2raw(chunk1)
```
  
Then check the results:
```python
# find the raw image name list
>>> img_dict.keys()   
dict_keys(['DJI_0177.JPG', 'DJI_0178.JPG', 'DJI_0179.JPG', 'DJI_0180.JPG', ... ]

# the roi pixel coordinate on that image
>>> img_dict['DJI_0177.JPG'] 
array([[ 779,  902],
       [1043,  846],
       [1099, 1110],
       [ 834, 1166],
       [ 779,  902]])
```
 
</details>

## <div align="center">References</div>

Please cite this paper if this project helps you：

```latex
@Article{wang_easyidp_2021,
AUTHOR = {Wang, Haozhou and Duan, Yulin and Shi, Yun and Kato, Yoichiro and Ninomiya, Seish and Guo, Wei},
TITLE = {EasyIDP: A Python Package for Intermediate Data Processing in UAV-Based Plant Phenotyping},
JOURNAL = {Remote Sensing},
VOLUME = {13},
YEAR = {2021},
NUMBER = {13},
ARTICLE-NUMBER = {2622},
URL = {https://www.mdpi.com/2072-4292/13/13/2622},
ISSN = {2072-4292},
DOI = {10.3390/rs13132622}
}
```

We also thanks the benefits from the following open source projects:

* numpy: [https://numpy.org/](https://numpy.org/)
* matplotlib:[https://matplotlib.org/](https://matplotlib.org/)
* pillow: [https://github.com/python-pillow/Pillow](https://github.com/python-pillow/Pillow)
* pyproj: [https://github.com/pyproj4/pyproj](https://github.com/pyproj4/pyproj)
* tifffile: [https://github.com/cgohlke/tifffile](https://github.com/cgohlke/tifffile)
* shapely: [https://github.com/shapely/shapely](https://github.com/shapely/shapely)
* laspy/lasrs/lasio: [https://github.com/laspy/laspy](https://github.com/laspy/laspy)
* plyfile: [https://github.com/dranjan/python-plyfile](https://github.com/dranjan/python-plyfile)
* ezdxf: [https://github.com/mozman/ezdxf](https://github.com/mozman/ezdxf)
* pyshp: [https://github.com/GeospatialPython/pyshp](https://github.com/GeospatialPython/pyshp)
* tabulate: [https://github.com/astanin/python-tabulate](https://github.com/astanin/python-tabulate)
* tqdm: [https://github.com/tqdm/tqdm](https://github.com/tqdm/tqdm)

This project was partially funded by:

* the JST AIP Acceleration Research “Studies of CPS platform to raise big-data-driven AI agriculture”; 
* the SICORP Program JPMJSC16H2; 
* CREST Programs JPMJCR16O2 and JPMJCR16O1; 
* the International Science & Technology Innovation Program of Chinese Academy of Agricultural Sciences (CAASTIP); 
* the National Natural Science Foundation of China U19A2061.