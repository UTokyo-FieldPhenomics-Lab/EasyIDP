<div align="center">

<p>
   <!-- <a align="left" href="https://ultralytics.com/yolov5" target="_blank"> -->
   <img width="850" src="https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP/raw/v2.0/docs/_static/images/header_v2.0.png"></a>
</p>

<p align="center">
  <img alt="GitHub code size in bytes" src="https://img.shields.io/tokei/lines/github/UTokyo-FieldPhenomics-Lab/EasyIDP?style=plastic">
  <img alt="GitHub" src="https://img.shields.io/github/license/UTokyo-FieldPhenomics-Lab/EasyIDP?style=plastic">
  <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/UTokyo-FieldPhenomics-Lab/EasyIDP?style=plastic">
  <img alt="GitHub Downloads" src="https://img.shields.io/github/downloads/UTokyo-FieldPhenomics-Lab/EasyIDP/total?label=github%20downloads&style=plastic">
  <img alt="GitHub Downloads" src="https://img.shields.io/pypi/dm/easyidp?color=%233775A9&label=pypi%20downloads&style=plastic">
</p>

</div>

EasyIDP (Easy Intermediate Data Processor), A handy tool for dealing with region of interest (ROI) on the image reconstruction (Metashape & Pix4D) outputs, mainly in agriculture applications. It provides the following functions: 

1. Backward Projection ROI to original images (`Backward Projector`).
2. Crop ROI on GeoTiff Maps (DOM & DSM) and Point Cloud (`ROI Cropper`).
3. Save cropped results to corresponding files (`ROI Saver`).

This project tried to use packges based on pure-python, instead of installing some heavy packages (Open3D, OpenCV) and hard to install packages (GDAL dependices) for one or two individual functions. This may cause efficiency loss and differences in coding habit.

<p align="center">
  <img alt="GitHub Downloads" height="400px" src="https://api.star-history.com/svg?repos=UTokyo-FieldPhenomics-Lab/EasyIDP&type=Date">
</p>

## <div align="center">Documentation</div>

Please check [Official Documents](https://easyidp.readthedocs.io/en/latest/) ( [中文](https://easyidp.readthedocs.io/zh_CN/latest/) | [日本語(翻訳募集)](https://easyidp.readthedocs.io/ja/latest/) ) for full documentations. And please also use the [Github Discussion](https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP/discussions) when you meet any problems.

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

* package main (**for users**)
  * numpy: [https://numpy.org/](https://numpy.org/)
  * matplotlib:[https://matplotlib.org/](https://matplotlib.org/)
  * scikit-image: [https://github.com/scikit-image/scikit-image](https://github.com/scikit-image/scikit-image)
  * pyproj: [https://github.com/pyproj4/pyproj](https://github.com/pyproj4/pyproj)
  * tifffile: [https://github.com/cgohlke/tifffile](https://github.com/cgohlke/tifffile)
  * imagecodecs: [https://github.com/cgohlke/imagecodecs](https://github.com/cgohlke/imagecodecs)
  * shapely: [https://github.com/shapely/shapely](https://github.com/shapely/shapely)
  * laspy/lasrs/lasio: [https://github.com/laspy/laspy](https://github.com/laspy/laspy)
  * plyfile: [https://github.com/dranjan/python-plyfile](https://github.com/dranjan/python-plyfile)
  * pyshp: [https://github.com/GeospatialPython/pyshp](https://github.com/GeospatialPython/pyshp)
  * tabulate: [https://github.com/astanin/python-tabulate](https://github.com/astanin/python-tabulate)
  * tqdm: [https://github.com/tqdm/tqdm](https://github.com/tqdm/tqdm)
  * gdown: [https://github.com/wkentaro/gdown](https://github.com/wkentaro/gdown)
* package documentation (**for developers**)
  * sphinx: [https://github.com/sphinx-doc/sphinx](https://github.com/sphinx-doc/sphinx)
  * nbsphinx: [https://github.com/spatialaudio/nbsphinx](https://github.com/spatialaudio/nbsphinx)
  * sphinx-gallery: [https://github.com/sphinx-gallery/sphinx-gallery](https://github.com/sphinx-gallery/sphinx-gallery)
  * sphinx-inline-tabs: [https://github.com/pradyunsg/sphinx-inline-tabs](https://github.com/pradyunsg/sphinx-inline-tabs)
  * sphinx-intl: [https://github.com/sphinx-doc/sphinx-intl](https://github.com/sphinx-doc/sphinx-intl)
  * sphinx-rtc-theme: [https://github.com/readthedocs/sphinx_rtd_theme](https://github.com/readthedocs/sphinx_rtd_theme)
  * furo: [https://github.com/pradyunsg/furo](https://github.com/pradyunsg/furo)
* package testing and releasing (**for developers**)
  * pytest: [https://github.com/pytest-dev/pytest](https://github.com/pytest-dev/pytest)
  * packaging: [https://github.com/pypa/packaging](https://github.com/pypa/packaging)
  * wheel: [https://github.com/pypa/wheel](https://github.com/pypa/wheel)

This project was partially funded by:

* the JST AIP Acceleration Research “Studies of CPS platform to raise big-data-driven AI agriculture”; 
* the SICORP Program JPMJSC16H2; 
* CREST Programs JPMJCR16O2 and JPMJCR16O1; 
* the International Science & Technology Innovation Program of Chinese Academy of Agricultural Sciences (CAASTIP); 
* the National Natural Science Foundation of China U19A2061.
