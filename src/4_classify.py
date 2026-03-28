"""步骤 4: 组件分类与线段判定模块。

此模块通过分析每个连通分量的几何特征（如最大内切圆半径和最小外接圆半径），
判断其是否为构成扭结的有效线段，从而剔除噪声、毛刺、孤立斑点等干扰元素。
"""

import cv2
import numpy as np


def is_line_segment(layer: np.ndarray) -> tuple[int, int, float]:
    """计算单个连通分量的几何特征参数。

    通过轮廓提取和距离变换（Distance Transform）获取组件的最大内切圆半径，
    并计算其最小外接圆半径。

    Args:
        layer: 单个连通分量的二值掩码层。

    Returns:
        包含以下特征的元组: (组件标签值, 最大内切圆半径, 最小外接圆半径)。
    """
    label = layer.max()
    mask = layer.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return label, 0, 0

    contour = contours[0]

    # 使用距离变换计算最大内切圆半径
    contour_img = np.zeros_like(mask)
    cv2.drawContours(contour_img, contours, -1, 255, thickness=cv2.FILLED)
    dist_transform = cv2.distanceTransform(
        contour_img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    _, max_val, _, _ = cv2.minMaxLoc(dist_transform)

    # 计算最小外接圆半径
    (x, y), min_enclosing_radius = cv2.minEnclosingCircle(contour)

    return label, int(max_val), min_enclosing_radius


def classify_segments(segmented_image: np.ndarray) -> dict[int, bool]:
    """对所有分割出的组件进行分类，识别有效线段。

    判定标准如下:
    1. 组件的最大内切圆半径与全局估计线宽的偏差在 40% 以内。
    2. 最小外接圆半径至少为估计线宽的 3 倍（确保组件具有一定的细长特征，而非圆形斑点）。

    Args:
        segmented_image: 分层后的三维数组 (组件数, 高, 宽)。

    Returns:
        分类结果字典，Key 为标签 ID，Value 为布尔值（True 表示是线段，False 表示是干扰）。
    """
    classification = {}
    radii = np.zeros(len(segmented_image))
    min_enclosing_radii = np.zeros(len(segmented_image))

    for i, layer in enumerate(segmented_image):
        label, max_val, min_enclosing_radius = is_line_segment(layer)
        radii[i] = max_val
        min_enclosing_radii[i] = min_enclosing_radius
        classification[label] = (max_val, min_enclosing_radius)

    # 剔除极端值（上下各 10%），以获取更稳健的线宽估计参考
    radii_sorted = np.sort(radii)
    percentile = max(1, int(len(radii_sorted) * 0.1))
    radii_filtered = radii_sorted[percentile:-percentile]

    if len(radii_filtered) < 1:
        # 若样本不足，则返回空的分类字典
        return {}

    # 以过滤后的最小值作为全局线宽估计基础
    line_width = radii_filtered[0]

    for label, (radius, min_enc_radius) in classification.items():
        is_line = (
            abs(radius - line_width) < 0.4 * line_width
            and min_enc_radius >= 3 * line_width
        )
        classification[label] = is_line

    return classification

