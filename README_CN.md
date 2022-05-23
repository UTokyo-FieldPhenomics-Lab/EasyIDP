<div align="center">

<p>
   <!-- <a align="left" href="https://ultralytics.com/yolov5" target="_blank"> -->
   <img width="850" src="https://github.com/HowcanoeWang/EasyIDP/wiki/static/easyidp_head.svg"></a>
</p>

<p align="center">
  <img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/UTokyo-FieldPhenomics-Lab/EasyIDP?style=plastic">
  <img alt="GitHub" src="https://img.shields.io/github/license/UTokyo-FieldPhenomics-Lab/EasyIDP?style=plastic">
  <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/UTokyo-FieldPhenomics-Lab/EasyIDP?style=plastic">
</p>

<a href="README.md">English</a>

</div>
EasyIDP(中间数据处理助手)是一个处理三维重建(Metashape和Pix4D)产品上感兴趣区域(ROI)的软件包(主要是农业应用)，提供如下的功能：

1. 在正射地图(DOM)、高程图(DSM)和点云上把ROI切出来
2. 把ROI反投影回原始图片上


## <div align="center">快速上手 (填坑中)</div>

<details open>
<summary>配置环境</summary>

在[**Python>=3.8.0**](https://www.python.org/)环境里，克隆并下载[requirements.txt](https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP/blob/master/requirements.txt)

```bash
git clone https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP.git  # 克隆
cd EasyIDP
pip install -r requirements.txt  # 安装需要的依赖包
```

</details>

<details open>
<summary>加载包</summary>

这个包可以直接作为源码使用，而不需要安装在python环境里。所以如果需要使用的时候，使用下面的代码来实现导入：

```python
import sys
sys.path.insert(0, f'C:/之前/下载/的/包/路径/EasyIDP')
  
import easyidp as idp
```
</details>

在执行下面的示例代码前，请确保基本理解了Metashape或Pix4D的图像三维重建的工作流程，并且知道如何导出地图(DOM和DSM的tiff文件格式)和点云(ply文件格式)。并且知道一些基本的GIS矢量文件(shp)的格式与制作方法。

> 请注意，如果你看到了这句话，说明下面的这些功能还没有完成

<details close>
<summary>1. 读取ROI</summary>

```python
roi = idp.ROI("xxxx.shp")  # 经纬度二维信息
  
# 从高程图DSM里获取高度信息
roi.get_z_from_dsm("xxxx_dsm.tiff")  # 增加高度成为三维信息
```

二维的ROI可以用来切正射地图、高程图和点云(参考`2.切ROI`)。三维点ROI可以用来反投影回原始图片上(参考`4.反投影`)。

  
或者你可以直接自动创建一个网格ROI：
  
```python
roi = idp.ROI(grid_h=300, grid_w=300, tif_path="xxxx.tif")
```
</details>

<details close>
<summary>2. 切ROI</summary>
  
```python
# 读取正射地图和高程图文件
dom = idp.GeoTiff("xxx_dom.tif")
dsm = idp.GeoTiff("xxx_dsm.tif")
  
# 读取点云文件
ply = idp.PointCloud("xxx_pcd.ply")
  
# 切ROI
dom_parts = roi.clip(dom)
dsm_parts = roi.clip(dsm)
pcd_parts = roi.clip(ply)
```
  
</details>

<details close>
<summary>3. 读取重建项目</summary>
  
```python
proj = idp.Recons()
proj.add_pix4d(["aaa.p4d", "bbb.p4d", ...])  # 支持使用列表来输入时间序列项目
proj.add_metashape(["aaa.psx", "bbb.psx"])
```

请注意，对于Metashape的时间序列项目，推荐在一个项目中建立多个Chunk来记录不同的时间，如下图所示：
  
<div align="center"><img width="350" src="images/metashape_multi_chunks.png"></a></div>

但是每个时间序列单独一个只有一个chunk的Metashape文件，也是可接受的。EasyIDP包会自动的按照给定的顺序分离出里面的每一个Chunk。

<div align="center"><img width="550" src="images/metashape_single_chunk.png"></a></div>

然后你可以按照下面两种方法获取每一个Chunk：

```python
chunk1 = proj[0]
# or
chunk1 = proj["chunk_or_project_name"]
```

</details>

<details close>
<summary>4. 反投影</summary>
  
```python
>>> img_dict = roi.back_to_raw(chunk1)
```
  
然后检查运算结果：
```python
# 所有找到的原始图片
>>> img_dict.keys()   
dict_keys(['DJI_0177.JPG', 'DJI_0178.JPG', 'DJI_0179.JPG', 'DJI_0180.JPG', ... ]

# ROI在该图片上的像素坐标
>>> img_dict['DJI_0177.JPG'] 
array([[ 779,  902],
       [1043,  846],
       [1099, 1110],
       [ 834, 1166],
       [ 779,  902]])
```
 
</details>


<details close>
<summary>小技巧</summary>
  
如果用的是Pix4D的话，只要你没有移动原始的项目文件，包可以自动找到输出的正射地图等路径：
```python
>>> proj[0].kind
"pix4D"
>>> proj[0].dom_path
"E:\...\pix4d_project_folder\3_dsm_ortho\2_mosaic\project_name_transparent_mosaic_group1.tif"
```

但是Metashape项目，导出的路径非常自由，需要手动指定路径
```python
>>> proj[0].kind
"metashape"
>>> proj[0].dom_path = r"E:\where\you\export\metashape\results\dom.tif"
```

</details>


## <div align="center">文档</div>

完整的文档请查阅：[Github Wiki](https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP/wiki)。如果遇到问题，请先查阅是否已经在[Github Discussion](https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP/discussions)被解决了。


## <div align="center">参考论文</div>

如果您的研究受益于该项目，请引用我们的论文：

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

我们也感谢下面所有开源工程对本项目的贡献：

* numpy: [https://numpy.org/](https://numpy.org/)
* matplotlib:[https://matplotlib.org/](https://matplotlib.org/)
* scikit-image: [https://scikit-image.org/](https://scikit-image.org/)
* pyproj: [https://github.com/pyproj4/pyproj](https://github.com/pyproj4/pyproj)
* tifffile: [https://github.com/cgohlke/tifffile](https://github.com/cgohlke/tifffile)
* shapely: [https://github.com/shapely/shapely](https://github.com/shapely/shapely)
* laspy/lasrs/lasio: [https://github.com/laspy/laspy](https://github.com/laspy/laspy)
* plyfile: [https://github.com/dranjan/python-plyfile](https://github.com/dranjan/python-plyfile)
* ezdxf: [https://github.com/mozman/ezdxf](https://github.com/mozman/ezdxf)
* pyshp: [https://github.com/GeospatialPython/pyshp](https://github.com/GeospatialPython/pyshp)