我的提问：
-----

Geotiff是否可以存储旋转角度的信息？即一个正方形的geotiff（200x200）记录了这个应该有13度的旋转角，把这个geotiff拖到qgis里，自动把这个旋转应用上？（不用存储边缘mask的信息，正常记录的话，需要220x220以上的像素）

---

Gemini 回答：
----------

显示思路

**Investigating GeoTIFF Rotation**

I'm delving into whether a GeoTIFF, specifically a 200x200 square one, can store rotation data. I'm focusing on the technical capabilities of the format.

**Verifying GeoTIFF Capabilities**

I'm now focusing on how GeoTIFFs store georeferencing data. Specifically, I'm examining the ModelTiepointTag, ModelPixelScaleTag, and ModelTransformationTag. My internal knowledge indicates that ModelPixelScaleTag and ModelTiepointTag are common but don't handle rotation, while ModelTransformationTag seems like a strong contender, as it allows for an affine transformation. I'm moving toward validating the use of the affine transform for this task.

**Confirming GeoTIFF Rotation**

I've confirmed that the affine transformation matrix, particularly through the ModelTransformationTag, is the mechanism for rotation within a GeoTIFF. QGIS demonstrably handles this, rendering rotated images directly from the metadata. I'm focusing now on clarifying the core concept: the TIFF stores the pixel grid, and the georeferencing dictates its spatial placement.

**Clarifying Rotation Details**

I'm now detailing practical implementation. I'll highlight the `ModelTransformationTag` and explain World Files (.tfw) as easier alternatives for comprehension, showcasing the structure with rotation parameters. I've validated QGIS's native handling. I'll refine the explanation, addressing the common confusion between resampling and dynamic projection to underscore the user's objective: avoiding resampling.

**Simplifying Matrix Concepts**

I'm working on explaining the rotation simply. The affine transformation is key, so I'm drafting an explanation centered on it. I initially considered a full matrix but found it too complex. Instead, I'm using the simpler GDAL geotransform array: (c, a, b, f, d, e), relating the pixel columns and rows to geographic coordinates through a, b, d and e, along with the origin.

**答案是肯定的。GeoTIFF 完全可以存储旋转角度信息，而且不需要改变图像本身的尺寸。**

你描述的需求（保持  $200\times 200$  像素，但带有旋转参数）正是 GeoTIFF 中 **仿射变换（Affine Transformation）** 的典型应用场景。

以下是详细的技术原理解析和操作说明：

### 1\. 核心原理：仿射变换矩阵

通常我们认为的 GeoTIFF 是“正南正北”的，这时候它只需要记录左上角坐标和像素分辨率。

但是，GeoTIFF 标准支持在文件头中写入一个 **ModelTransformationTag**（或者通过 World File `.tfw` 配合）。这个标签存储了一个仿射变换矩阵，用来描述像素坐标（行列号）与地理坐标（经纬度或投影坐标）之间的关系。

在这个矩阵中，通过设置非零的 **旋转/剪切参数**，就可以定义旋转角度。

**数学逻辑如下（使用 LaTeX 表示）：**

假设像素坐标为  $\left(P_{x},P_{y}\right)$ ，地理坐标为  $\left(G_{x},G_{y}\right)$ ，标准的转换公式为：

$$
G_{x}G_{y}​=A+B\cdot P_{x}+C\cdot P_{y}=D+E\cdot P_{x}+F\cdot P_{y}​
$$

*   **A, D:** 左上角的地理坐标（X 和 Y）。
    
*   **B, F:** 像素的宽和高（通常  $F$  为负数）。
    
*   **C, E:** **这两个参数控制旋转**。如果是正南正北的图像，这两个值为 0；如果有旋转（例如 13 度），这两个值就是非零的系数（基于  $\sin \left(13^{\circ }\right)$  和  $\cos \left(13^{\circ }\right)$  计算得出）。
    

### 2\. 对于你的具体需求

*   **存储方式：** GeoTIFF 文件内部的二进制数据依然只有  $200\times 200$  个像素。文件大小不会增加，不需要填充黑边（NoData）。
    
*   **QGIS 的表现：** 当你把这个文件拖入 QGIS 时：
    
    1.  QGIS 读取文件头中的仿射变换参数。
        
    2.  它识别出  $C$  和  $E$  参数不为 0。
        
    3.  QGIS 会在**渲染引擎层面**动态地将图像旋转 13 度显示在地图上。
        
    4.  **结果：** 你在屏幕上看到的是一个斜着的正方形，边缘整齐，没有锯齿状的黑边（除非你手动做了重采样）。
        

### 3\. 如何实现？（Python 示例）

如果你需要编程实现给一个普通的 TIFF 加上旋转信息，可以使用 Python 的 `rasterio` 或 `gdal` 库。

以下是一个使用 `rasterio` 的示例，展示如何定义这种旋转：

Python

    import rasterio
    from rasterio.transform import Affine
    import math
    
    # 假设参数
    width = 200
    height = 200
    rotation_angle = 13  # 旋转角度（度）
    pixel_size = 1.0     # 每个像素代表的地理距离
    top_left_x = 100000  # 左上角地理坐标 X
    top_left_y = 200000  # 左上角地理坐标 Y
    
    # 1. 计算旋转后的仿射变换矩阵
    # 这里使用的是 'rotation' 辅助函数，它会自动计算上文提到的 A,B,C,D,E,F 参数
    # 这里的 rotation 是逆时针还是顺时针取决于具体的库定义，通常正值是逆时针
    transform = Affine.translation(top_left_x, top_left_y) * \
                Affine.rotation(rotation_angle) * \
                Affine.scale(pixel_size, -pixel_size) # 负号因为图像坐标系Y轴向下
    
    # 2. 写入 GeoTIFF
    # 假设你有一个名为 data 的 numpy 数组 (200x200)
    with rasterio.open(
        'rotated_image.tif',
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='uint8', # 或者是你的数据类型
        crs='EPSG:3857', # 或者是你的投影代码
        transform=transform
    ) as dst:
        dst.write(data, 1)

### 总结

你不需要把图片扩充到  $220\times 220$ 。GeoTIFF 的元数据（Metadata）完全有能力告诉 QGIS：“这是一张  $200\times 200$  的图片，请把它的左上角放在这里，然后旋转 13 度显示。” QGIS 会完美支持这种操作。

* * *

**我可以为你提供更具体的帮助：** 如果你手头有这类的 TIFF 文件但不知道如何计算那 6 个仿射参数，或者想要通过 GDAL 命令行工具来快速设置旋转，请告诉我，我可以提供对应的脚本或命令。

---

我的提问：
-----

geotiff是否支持存储额外的nx2的坐标信息，用来记录mask的polygon？

---

Gemini 回答：
----------

显示思路

**Considering the Core Question**

I'm now zeroing in on the fundamental inquiry: Can GeoTIFFs accommodate supplementary coordinate data to define a mask polygon? My focus is on determining if GeoTIFFs have that capacity.

**Assessing Implementation Potential**

I'm now investigating potential solutions. It appears GeoTIFFs don't natively accommodate vector data like a polygon mask. However, I'm exploring the potential for storing mask coordinates as metadata, possibly using XML or GDAL tags. The key is to see if existing GIS tools like GDAL or QGIS readily support this approach, and how. So far, the vector approach the user wants seems the most important to consider.

**Evaluating Metadata Strategies**

I've been evaluating several strategies. While a direct "Vector Polygon Tag" doesn't exist in GeoTIFF, I've confirmed that the OGC GeoTIFF standard allows "Validity Masks" as raster bands. The most viable approach appears to be storing the polygon as WKT or GeoJSON within GDAL metadata (PAM). I'm also considering the implications of QGIS compatibility. Although not automatic, the data will still be present. It seems like the best approach.

**Refining The Response**

I'm now drafting the answer. It's becoming clear that a GeoTIFF is raster-focused, so storing vector polygons directly isn't its primary purpose. I'm clarifying that, and focusing on several possible solutions. I can write the polygon as WKT/GeoJSON in metadata, which keeps data in one place, but QGIS won't automatically mask the data. Alternatively, I can use an alpha band. I'm also looking into the GDAL "Footprint" concept.

**Clarifying Technical Nuances**

I'm now zeroing in on the best method. GDAL does support a `FOOTPRINT` tag in its metadata domain, which answers the core question. QGIS, however, ignores this for rendering unless an alpha band is used, but still considers the data as useful. Storing the polygon as WKT/GeoJSON in metadata works for data storage, but isn't a direct mask within QGIS. An XML packet (XMP) is also possible but isn't as clean. So I am structuring the response to reflect this, with an answer that is technically "yes" via metadata, but not natively or automatically supported. My methods will include the metadata approach, as well as the alpha band.

**Detailing Masking Techniques**

I'm now detailing several masking techniques. I'm focusing on the alpha band approach, as well as the "Footprint" metadata. I'll also add a comparison table for vector vs. raster masks, and a Python code example for writing Polygon WKT to TIFF metadata. The user's " $N\times 2$  coordinates" request is important, and I am emphasizing that simply storing this string in the TIFF won't create an automatic mask in QGIS, but will be available. I'll mention Virtual Rasters as a solution. I'm also ensuring the correct formatting and tone in the text.

这是一个非常深入且切中要害的问题。

简单直接的回答是：**GeoTIFF 本身没有标准的、专门用来存储“矢量裁剪多边形（Mask Polygon）”并被 GIS 软件自动执行的 Tag。**

但是，可以通过 **“非标准元数据（Metadata）”** 的方式把  $N\times 2$  的坐标存进去，或者使用 **GDAL 的辅助机制** 来实现。

以下是几种方案的详细对比，取决于你是想要“仅仅存储这些坐标作为记录”，还是“希望 QGIS 能利用这些坐标进行裁剪”。

* * *

### 方案 1：存入元数据（Metadata） —— 最符合你“存储坐标”的描述

你可以将  $N\times 2$  的坐标点转换成文本格式（如 **WKT** 或 **GeoJSON**），然后作为字符串存储在 GeoTIFF 的 **Metadata Tag** 中（例如 `GDAL_METADATA` 或 `TIFFTAG_IMAGEDESCRIPTION`）。

*   **原理：** GeoTIFF 允许存储任意的键值对（Key-Value Pair）数据。
    
*   **如何做：** 把 `POLYGON ((x1 y1, x2 y2, ...))` 这个字符串写进文件头。
    
*   **QGIS 的表现：**
    
    *   当你把文件拖进 QGIS，**QGIS 仍然会显示完整的正方形图像**。它不会自动去读这个 Tag 来裁剪图片。
        
    *   你需要手动在图层属性的“元数据”选项卡里查看，或者写 Python 脚本来提取这个多边形并应用。
        

**Python 写入示例：**

Python

    import rasterio
    
    # 假设你的 Mask 坐标
    mask_coords = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"
    
    with rasterio.open('image.tif', 'r+') as dst:
        # 将坐标存入名为 'MASK_POLYGON' 的自定义元数据中
        dst.update_tags(MASK_POLYGON=mask_coords)

* * *

### 方案 2：使用 VRT (Virtual Raster) —— 极其推荐的替代方案

如果你不想修改原始像素（保持  $200\times 200$ ），但又想让 QGIS 自动应用那个多边形作为 Mask，**VRT 是最佳解决方案**。

*   **原理：** 创建一个极小的文本文件（.vrt），它指向你的 GeoTIFF，并在 XML 结构中通过 `<Cutline>` 标签记录那  $N\times 2$  的坐标。
    
*   **QGIS 的表现：** 你把 **.vrt 文件**（而不是 .tif）拖进 QGIS。QGIS 会读取 VRT 中的 `<Cutline>`，实时地把多边形以外的区域变透明，而原始的 TIFF 数据完全不动。
    

**VRT 内容示意：**

XML

    <VRTDataset rasterXSize="200" rasterYSize="200">
      <VRTRasterBand dataType="Byte" band="1">
        <SimpleSource>
          <SourceFilename>source_image.tif</SourceFilename>
        </SimpleSource>
      </VRTRasterBand>
      <Cutline>POLYGON ((...))</Cutline>
    </VRTDataset>

* * *

### 方案 3：Alpha Band（Alpha 通道）—— 工业标准做法

虽然你说“不用存储边缘 mask 的信息”，但在 GIS 行业中，为了让软件“自动应用”透明效果，最标准的方法还是增加一个波段（Alpha Band）。

*   **原理：** 给 TIFF 增加第 4 个波段（如果是 RGB），在这个波段里，Mask 内的像素值为 255，Mask 外的为 0。
    
*   **缺点：** 确实如你所说，这增加了像素数据量，而不是仅仅存几十个坐标点。
    
*   **优点：** 任何软件（QGIS, ArcGIS, Photoshop）打开它都能自动识别透明区域。
    

* * *

### 总结与建议

| 方案 | 存储内容 | QGIS 是否自动裁剪? | 是否增加像素体积? | 推荐场景 |
| --- | --- | --- | --- | --- |
| 元数据存储 | WKT/GeoJSON 字符串 | 否(需脚本提取) | 否 | 仅作为数据归档/记录 |
| VRT 伴随文件 | <Cutline>矢量坐标 | 是(拖入VRT时) | 否 (仅增加几KB文本) | 最符合你的需求，需额外文件 |
| Alpha 通道 | 像素掩膜 | 是 | 是 | 追求最大兼容性时 |

导出到 Google 表格

**针对你的场景：** 如果你完全不想用额外的文件（如 VRT），也不想加 Alpha 通道，只希望单文件 GeoTIFF 既保持小体积又能存 Mask： **你可以存入元数据（方案 1），但在 QGIS 里无法自动生效。**

**下一步建议：** 如果你的工作流允许使用伴随文件，**我会建议你为你那个旋转的 GeoTIFF 生成一个对应的 `.vrt` 文件**。VRT 既包含了旋转的 GeoTransform 信息，也包含了 `<Cutline>` 矢量裁剪信息。这样在 QGIS 里打开 VRT，你将看到一个**旋转的、且被完美裁剪的**图像，而原始数据依然是那个干干净净的  $200\times 200$  正方形。

你想看看如何用 Python 生成这种带 Cutline 的 VRT 文件吗？

---

我的提问：
-----

使用rasterio的方案一实现，仅用来存储 mask对应的polygon，同时提供polygon坐标读取的代码

---

Gemini 回答：
----------

显示思路

**Defining the Core Challenge**

I'm focusing now on the fundamental challenge: efficiently storing an N x 2 polygon mask within a GeoTIFF using `rasterio`. The core goal is to correctly represent the polygon's shape and position within the GeoTIFF format using a suitable scheme, which is Scheme 1 in this context. I'm taking the initial request as my current point of analysis.

**Refining the Storage Method**

I'm now zeroing in on the storage aspect, keeping in mind this is solely for recording purposes within the GeoTIFF. My main requirement is to provide Python code that can write and read this polygon data. The task is to store an N x 2 polygon mask within a GeoTIFF using rasterio. The task is to use GeoTIFF tags (key-value pairs) in combination with the WKT format for serialization, specifically using Shapely for enhanced robustness, although string formatting may suffice for simpler scenarios.

**Prioritizing Simplicity in Implementation**

I've decided to streamline the polygon storage method for the example. While `shapely` is great, I'll stick to basic string formatting for the GeoTIFF example, while mentioning `shapely` as a safer standard alternative. I will now open the file in `'r+'` mode and utilize `dst.update_tags()` for updating the tags. For the custom key, I'm thinking of using `MASK_WKT` with the default namespace.

**Clarifying Coordinate System**

I'm now thinking about the coordinate system and I'll default to the understanding that provided coordinates should be georeferenced. I'm focusing on the WKT format for storing the polygon's coordinates, because it's standard: `POLYGON ((x y, x y, ...))`. I'm drafting two primary functions for writing and reading the polygon data to and from the GeoTIFF.

**Structuring the Python Code**

I'm starting to build out Python functions, focusing on using `rasterio`, `shapely.geometry`, and `shapely.wkt` for managing the WKT. My draft includes write and read functions using `rasterio.update_tags()`, and `shapely.wkt`. The example writes a polygon to a GeoTIFF using the tag 'CROP\_POLYGON'.

这是一个非常实用的方案。使用 `rasterio` 的 metadata（元数据）功能，我们可以把  $N\times 2$  的坐标点序列化为字符串（推荐使用标准的 **WKT** 格式）存储在 GeoTIFF 的标签中。

这样做的好处是数据跟随文件，且不破坏图像本身的结构。

以下是完整的 Python 代码实现，分为**写入**和**读取**两部分。为了方便处理几何坐标与文本的转换，我们通常会配合 `shapely` 库使用（GIS 处理的标准库）。

### 准备工作

你需要安装 `rasterio` 和 `shapely`：

Bash

    pip install rasterio shapely

### 完整代码实现

这段代码包含两个函数：`save_polygon_to_geotiff` 和 `read_polygon_from_geotiff`。为了演示效果，脚本开头会先生成一个空白的测试 GeoTIFF。

Python

    import rasterio
    from rasterio.transform import from_origin
    import numpy as np
    from shapely.geometry import Polygon
    import shapely.wkt
    
    # ==========================================
    # 1. 写入功能的实现
    # ==========================================
    def save_polygon_to_geotiff(tif_path, coords_list, tag_key='MASK_POLYGON'):
        """
        将 Nx2 的坐标列表转换为 WKT 字符串并存储到 GeoTIFF 的元数据中。
        
        Args:
            tif_path (str): GeoTIFF 文件路径
            coords_list (list): 形如 [(x1, y1), (x2, y2), ...] 的坐标列表
            tag_key (str): 存储在元数据中的键名
        """
        # 1. 确保坐标闭合（多边形起点和终点必须相同）
        if coords_list[0] != coords_list[-1]:
            coords_list.append(coords_list[0])
        
        # 2. 使用 Shapely 将坐标转换为标准的 WKT (Well-Known Text) 字符串
        # 例如: "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"
        poly = Polygon(coords_list)
        wkt_string = poly.wkt
        
        print(f"[-] 正在写入元数据，WKT 内容长度: {len(wkt_string)}")
    
        # 3. 以 'r+' (读写) 模式打开文件并更新 tags
        with rasterio.open(tif_path, 'r+') as src:
            # update_tags 接收关键字参数，所以我们需要动态解包字典
            metadata = {tag_key: wkt_string}
            src.update_tags(**metadata)
            
        print(f"[+] 成功将 Polygon 写入到标签 '{tag_key}' 中。")
    
    
    # ==========================================
    # 2. 读取功能的实现
    # ==========================================
    def read_polygon_from_geotiff(tif_path, tag_key='MASK_POLYGON'):
        """
        从 GeoTIFF 元数据中读取 WKT 字符串并还原为坐标列表。
        """
        with rasterio.open(tif_path, 'r') as src:
            # 获取所有标签
            tags = src.tags()
            
            if tag_key not in tags:
                print(f"[!] 未在文件中找到键名为 '{tag_key}' 的元数据。")
                return None
            
            wkt_string = tags[tag_key]
            print(f"[-] 读取到 WKT 字符串: {wkt_string[:50]}...")
            
            # 使用 Shapely 解析 WKT
            try:
                poly = shapely.wkt.loads(wkt_string)
                # 提取外部轮廓的坐标
                # list(poly.exterior.coords) 返回的是 [(x, y), ...]
                coords = list(poly.exterior.coords)
                return coords
            except Exception as e:
                print(f"[!] 解析 WKT 失败: {e}")
                return None
    
    # ==========================================
    # 3. 测试流程
    # ==========================================
    if __name__ == "__main__":
        filename = "test_with_mask.tif"
        
        # --- 步骤 A: 创建一个模拟的 GeoTIFF (200x200) ---
        # 这一步是为了让代码可运行，如果你已有文件，可跳过
        data = np.zeros((200, 200), dtype=rasterio.uint8)
        transform = from_origin(100, 100, 1, 1) # 简单模拟坐标系
        with rasterio.open(filename, 'w', driver='GTiff', 
                           height=200, width=200, count=1, dtype=data.dtype,
                           transform=transform) as dst:
            dst.write(data, 1)
    
        # --- 步骤 B: 定义你的 Mask Polygon (Nx2 坐标) ---
        # 假设这是该图像投影坐标系下的实际地理坐标
        my_mask_coords = [
            (110.5, 90.5), # x, y
            (150.0, 90.5),
            (150.0, 60.0),
            (120.0, 50.0),
            # 注意：这里不需要手动闭合，Shapely/代码逻辑会自动处理
        ]
    
        # --- 步骤 C: 写入 ---
        save_polygon_to_geotiff(filename, my_mask_coords)
    
        print("-" * 30)
    
        # --- 步骤 D: 读取验证 ---
        read_coords = read_polygon_from_geotiff(filename)
        
        if read_coords:
            print(f"[+] 成功读取并还原坐标 (前3个): {read_coords[:3]}")
            print(f"[+] 坐标点数量: {len(read_coords)}")

### 代码关键点解析

1.  **数据格式 (WKT)**：
    
    *   我选用了 **WKT (Well-Known Text)** 格式（如 `POLYGON ((x y, x y...))`）。
        
    *   **理由**：这是 GIS 领域的通用标准。比起存 JSON 字符串，WKT 更紧凑，且直接被 QGIS、ArcGIS、PostGIS 识别。如果你以后改变主意想让 QGIS 解析它，复制 WKT 字符串是最方便的。
        
2.  **`src.update_tags()`**：
    
    *   这是 `rasterio` 将键值对写入 GeoTIFF `GDAL_METADATA` 域的核心方法。这些数据是持久化存储在 .tif 文件头部的。
        
3.  **闭合性**：
    
    *   Polygon 标准要求首尾坐标一致。代码中 `save_polygon_to_geotiff` 加了一个简单的检查：如果传入的坐标没有闭合，自动把第一个点追加到末尾，防止生成无效的 Polygon。
        

### 如何在 QGIS 中手动查看这些数据？

当你把在这个文件拖入 QGIS 后：

1.  右键点击图层 -> **Properties (属性)**。
    
2.  点击 **Metadata (元数据)** 标签页。
    
3.  向下滚动找到 **Properties** 部分。
    
4.  你应该能看到一行类似 `MASK_POLYGON` 的条目，后面跟着你的坐标字符串。

---

