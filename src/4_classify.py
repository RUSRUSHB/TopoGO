"""步骤 4: 对于判定是否真正属于线段分类的工作站模块。

此模块主要靠精密测算每一个标定提取出来斑块其内部的内切圆最高宽范围以及同时反向评估圈选包含整个部件轮廓的最短外接圆半径是否与预先规定的结构合理范围相互符合，由此判定这是否才是扭结本身构成有效线段的主体（属于“有效线条”或者其它如脏迹、毛边圆点等应遭排查抛弃的无关元素）。
"""

import cv2
import numpy as np


def is_line_segment(layer: np.ndarray) -> tuple[int, int, float]:
    """单独自查和解剖任意一层被独立摘出图件区域由此求其相关几何构造尺寸。

    首先依靠引入外部轮廓提取再叠带距离拓扑变换运算找到那隐藏深处的最多中心内切辐射圈距离数据，并且连带求解能完全紧包它外部体积所必需外圈大圆半径大小结果。

    Args:
        layer: 仅仅只有特定目标某组件保留数值其他位置均为清零状态的一份被二维摘切拿出的剥离平层面。

    Returns:
        包含以下重要判定数据参数返回一个包裹体系即 (所属具体组件标签数值, 中心内敛圆其最大可达到扩张界限, 把那全包裹所需的最小容套圈幅宽限度)。
    """
    label = layer.max()
    mask = layer.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return label, 0, 0

    contour = contours[0]

    # 利用到距离图像渐进梯度拓扑换算计算最大被遮住能容内部塞下的圆润圈限界幅度
    contour_img = np.zeros_like(mask)
    cv2.drawContours(contour_img, contours, -1, 255, thickness=cv2.FILLED)
    dist_transform = cv2.distanceTransform(
        contour_img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    _, max_val, _, _ = cv2.minMaxLoc(dist_transform)

    # 包含住该图形轮廓的最紧凑圈起范围用圆解算
    (x, y), min_enclosing_radius = cv2.minEnclosingCircle(contour)

    return label, int(max_val), min_enclosing_radius


def classify_segments(segmented_image: np.ndarray) -> dict[int, bool]:
    """总体检视并甄别这全部组件里是否哪一部分可以认同确属于线而那些纯属其非线杂质。

    一段被接受算作一条有效合理线条的其判定要求如下:
    1. 前边查出其内置宽度容许中心大圆度必须同提取出的平均中位数正常全宽差距上下偏离浮动不超过容忍允许阈度 (四成百分比如 40%) 波动界线。
    2. 而同时要把整个躯干强套进去这其中必需付出的那个圈径长则应当最少要有基础正常躯干三跨之多至少这三倍线宽规模才能证实其形态细长具备特征表现非小斑点或大水滴一类圆肿形存在。

    Args:
        segmented_image: 由早先 separate_labels() 那里专门给按层面剖开发配出来的呈现形状规模即 (总标签项目数量, 该片画面高, 该片画面阔) 一般的复杂三维阵集格式大组合。

    Returns:
        记录全盘每一组件归宿的评估成绩映射字典，左侧其归属标签指引向 -> 值包含着该线符合判定是否 True(合格认定) 或者 False(打回原形)。
    """
    classification = {}
    radii = np.zeros(len(segmented_image))
    min_enclosing_radii = np.zeros(len(segmented_image))

    for i, layer in enumerate(segmented_image):
        label, max_val, min_enclosing_radius = is_line_segment(layer)
        radii[i] = max_val
        min_enclosing_radii[i] = min_enclosing_radius
        classification[label] = (max_val, min_enclosing_radius)

    # 将明显尺寸规模特别偏头尾极端的反常规错误数据(分别斩除头上脚下方那各前后 10% 那些过高或者过偏异己)给去掉再找有效评估标准参考基础数据
    radii_sorted = np.sort(radii)
    percentile = max(1, int(len(radii_sorted) * 0.1))
    radii_filtered = radii_sorted[percentile:-percentile]

    if len(radii_filtered) < 1:
        return "图像提取里当前符合基础规模判定的总合格条数达不到所需最少于 1 条前提"

    # 把所筛得的最瘦那一个残留骨干部分就暂先给认作基底该图像全局参考线主体公认横评宽度值
    line_width = radii_filtered[0]

    for label, (radius, min_enc_radius) in classification.items():
        is_line = (
            abs(radius - line_width) < 0.4 * line_width
            and min_enc_radius >= 3 * line_width
        )
        classification[label] = is_line

    return classification
