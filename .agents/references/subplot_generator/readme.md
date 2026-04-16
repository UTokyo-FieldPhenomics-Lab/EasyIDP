shp文件的读写（fiona/pyogrio）是否可以进一步使用pyshp来替代？

案是肯定的：pyshp (PyShp) 是目前 Python 生态中唯一能够完全脱离 GDAL 依赖、实现纯 Python 读写 Shapefile 的方案。

如果你追求的是“极致的零依赖”和“全平台 pip 一键安装”，pyshp 是终极选择。

1. 为什么用 pyshp 替代 fiona？
真正的零依赖： fiona 虽然比 geopandas 轻量，但它底层依然链接了 GDAL 的 C++ 库。在 Windows 上安装 fiona 有时仍会报错。而 pyshp 是纯 Python 编写，不依赖任何 C 库，只要有 Python 就能跑。

体积微小： 整个库只有一个文件，非常适合集成在 QGIS 插件或轻量级脚本中。

2. 代码重写：使用 pyshp + shapely
我们将你代码中的 load_boundary 和 generate（保存部分）改写为 pyshp 逻辑。

读取部分 (load_boundary)
```Python

import shapefile # pyshp 导入名为 shapefile
from shapely.geometry import shape, mapping

def load_boundary(self, shp_path: str):
    try:
        # 使用 context manager 确保文件关闭
        with shapefile.Reader(shp_path) as sf:
            if len(sf) != 1:
                logger.error("Boundary file must contain exactly one polygon.")
                return None
            
            # 获取几何和属性
            shape_record = sf.shapeRecord(0)
            # pyshp 的 __geo_interface__ 可以直接被 shapely 识别
            polygon = shape(shape_record.shape.__geo_interface__)
            
            # 注意：pyshp 不处理 .prj 文件，需手动读取
            prj_path = shp_path.replace('.shp', '.prj')
            crs_wkt = None
            if Path(prj_path).exists():
                crs_wkt = Path(prj_path).read_text()
                
            return polygon, crs_wkt
    except Exception as e:
        logger.error(f"Failed to load: {e}")
        return None
```

写入部分 (generate)

```Python
def save_subplots(self, results, output_path, crs_wkt):
    # results 是包含 {'geometry': poly, 'properties': {...}} 的列表
    with shapefile.Writer(output_path) as w:
        # 1. 定义字段 (id, row, col)
        w.field("id", "N")
        w.field("row", "N")
        w.field("col", "N")
        
        for item in results:
            # 2. 写入几何 (Shapely -> PyShp)
            # 使用 __geo_interface__ 转换
            w.shape(mapping(item['geometry']))
            
            # 3. 写入属性
            props = item['properties']
            w.record(props['id'], props['row'], props['col'])
            
    # 4. 手动写入 .prj 文件（pyshp 不自动生成）
    if crs_wkt:
        with open(output_path.replace('.shp', '.prj'), 'w') as f:
            f.write(crs_wkt)
```