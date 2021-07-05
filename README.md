![](https://github.com/HowcanoeWang/EasyIDP/wiki/static/easyidp_head.svg)

<p align="center">
  <img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/UTokyo-FieldPhenomics-Lab/EasyIDP?style=plastic">
  <img alt="GitHub" src="https://img.shields.io/github/license/UTokyo-FieldPhenomics-Lab/EasyIDP?style=plastic">
  <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/UTokyo-FieldPhenomics-Lab/EasyIDP?style=plastic">
</p>


EasyIDP (Easy Intermediate Data Processor), A tool to build a bridge from dealing with structure from motion (SfM) outputs, including point cloud data(PCD), orthomosaic (digital ortho maps, DOM), digital surface model(DSM), properly. Currently, Pix4D projects are fully supported, Agisoft has been supported in the testing version. The detailed documentation and GUI support is one the way.

> EasyIDP(中间数据处理助手)是一个用来方便的预处理`运动恢复结构`(SfM)三维重建软件的各类输出文件的工具，如点云、`正射影像`(DOM)、`数字高程模型`(DSM)等。目前完全支持Pix4D项目，Agisoft已在测试版中测试完成，详细的API说明文档和操作界面支持正在开发中。

* Documentation(文档)：    
  https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP/wiki
* Source code(源代码)：    
  https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP
* Bug reports(缺陷汇报)：    
  https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP/issues
* Please check forum before report bugs: 
  > 在提问之前，请先在论坛里寻找是否有解决方式     
  
  https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP/discussions

---

Called **EasyRIC** during inner developing (package `easyric`), the folder `easyidp` is currently under heavy reconstruction, thus do not use it in productive codes. However, playing with it is totally welcome now.

> 在内部测试期间的曾用名是**EasyRIC**(对应`easyric`文件夹)，最新的`easyidp`文件夹正在进行频繁的重构中, 因此不要把它使用在生产环境里，但是很欢迎“玩”一下这个包

Please cite this paper if this project helps you

> 如果您的研究受益于该项目，请引用我们的论文(尚在准备中)

```latex
@Article{rs13132622,
AUTHOR = {Wang, Haozhou and Duan, Yulin and Shi, Yun and Kato, Yoichiro and Ninomiya, Seish and Guo, Wei},
TITLE = {EasyIDP: A Python Package for Intermediate Data Processing in UAV-Based Plant Phenotyping},
JOURNAL = {Remote Sensing},
VOLUME = {13},
YEAR = {2021},
NUMBER = {13},
ARTICLE-NUMBER = {2622},
URL = {https://www.mdpi.com/2072-4292/13/13/2622},
ISSN = {2072-4292},
ABSTRACT = {Unmanned aerial vehicle (UAV) and structure from motion (SfM) photogrammetry techniques are widely used for field-based, high-throughput plant phenotyping nowadays, but some of the intermediate processes throughout the workflow remain manual. For example, geographic information system (GIS) software is used to manually assess the 2D/3D field reconstruction quality and cropping region of interests (ROIs) from the whole field. In addition, extracting phenotypic traits from raw UAV images is more competitive than directly from the digital orthomosaic (DOM). Currently, no easy-to-use tools are available to implement previous tasks for commonly used commercial SfM software, such as Pix4D and Agisoft Metashape. Hence, an open source software package called easy intermediate data processor (EasyIDP; MIT license) was developed to decrease the workload in intermediate data processing mentioned above. The functions of the proposed package include 1) an ROI cropping module, assisting in reconstruction quality assessment and cropping ROIs from the whole field, and 2) an ROI reversing module, projecting ROIs to relative raw images. The result showed that both cropping and reversing modules work as expected. Moreover, the effects of ROI height selection and reversed ROI position on raw images to reverse calculation were discussed. This tool shows great potential for decreasing workload in data annotation for machine learning applications.},
DOI = {10.3390/rs13132622}
}
```
