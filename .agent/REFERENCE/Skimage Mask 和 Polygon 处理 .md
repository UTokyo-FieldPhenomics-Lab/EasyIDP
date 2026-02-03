我的提问：
-----

skimage包是如何处理和记录 二值mask 和polygon坐标mask的？

---

Gemini 回答：
----------

在数字图像处理中，`scikit-image` (skimage) 主要将 mask 视为 **NumPy 数组**。不过，针对“二值掩码 (Binary Mask)”和“多边形坐标 (Polygon Coordinates)”这两者，它的处理逻辑和转换方式各有侧重。

以下是 `skimage` 处理这两类数据的核心机制：

* * *

1\. 二值掩码 (Binary Mask) 的处理
--------------------------

在 `skimage` 中，二值掩码被存储为 **二维布尔型或整型 NumPy 数组**。

*   **数据类型**：通常为 `bool`（`True` 表示目标，`False` 表示背景）或 `uint8`（`1` 或 `255` 表示目标）。
    
*   **记录方式**：它本质上是一个与原图尺寸相同的矩阵。
    
*   **核心工具**：
    
    *   `skimage.filters.threshold_*`：用于将灰度图转换为二值图。
        
    *   `skimage.morphology`：用于对二值掩码进行膨胀、腐蚀或开闭运算。
        

> **注意：** `skimage` 遵循  $Row\times Column$  (y, x) 的索引顺序，这在处理坐标时非常关键。

* * *

2\. 多边形坐标 (Polygon Coordinates) 的处理
-----------------------------------

多边形掩码通常记录为 **顶点的坐标序列**（例如：`[[r1, c1], [r2, c2], ...]`）。

*   **记录方式**：通常使用两个一维数组分别表示 **行 (r)** 和 **列 (c)** 的坐标。
    
*   **核心工具**：`skimage.draw.polygon`。
    

### 坐标与掩码的相互转换

这是最常见的操作场景，`skimage` 提供了非常直截了当的方法：

#### A. 从多边形坐标生成二值掩码

如果你有一组顶点坐标，想要生成对应的 Mask 矩阵：

Python

    from skimage import draw
    import numpy as np
    
    # 定义顶点 (row, col)
    r = np.array([10, 10, 80, 80])
    c = np.array([10, 80, 80, 10])
    
    # 创建空白画布
    mask = np.zeros((100, 100), dtype=np.uint8)
    
    # 填充多边形区域
    rr, cc = draw.polygon(r, c)
    mask[rr, cc] = 1

#### B. 从二值掩码提取边界坐标

如果你想把 Mask 转回坐标（用于矢量存储或可视化）：

*   使用 `skimage.measure.find_contours`：它使用 **Marching Squares 算法** 提取掩码的轮廓。
    
*   返回结果是一个包含坐标对 `(row, col)` 的列表。
    

* * *

3\. 关键差异对比
----------

| 特性 | 二值掩码 (Mask) | 多边形坐标 (Polygon) |
| --- | --- | --- |
| 存储形式 | 密集矩阵 (Pixel-wise) | 稀疏顶点 (Vertex-wise) |
| 空间占用 | 随图像分辨率增加而剧增 | 仅取决于形状的复杂度 |
| 适用场景 | 像素级分割、形态学运算 | 矢量标注、ROI 裁剪、存储导出 |
| 常用模块 | skimage.morphology | skimage.draw,skimage.measure |

导出到 Google 表格

* * *

4\. 进阶记录：标签图 (Label Image)
--------------------------

当一张图中有多个不同的目标时，`skimage` 会使用 **Label Image** 来记录。

*   **原理**：背景为 `0`，第一个物体像素全部设为 `1`，第二个设为 `2`，以此类推。
    
*   **函数**：`skimage.measure.label()` 可以将二值掩码中的连通域自动转换为这种标签记录格式。
    

**您是正在尝试将标注工具（如 LabelMe）产生的 JSON 坐标导入到 Python 中进行处理吗？** 如果需要，我可以为您演示如何编写一个从 JSON 顶点到 skimage 掩码的转换函数。

---

