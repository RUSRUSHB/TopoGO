"""步骤 1: 图像二值化处理。

使用固定阈值将输入的灰度图像转换为黑白二值图像。
"""

import cv2
import numpy as np


def binarize(img: np.ndarray, threshold: int = 127) -> np.ndarray:
    """使用固定阈值对输入图像进行二值化处理。

    Args:
        img: 输入的灰度图像数组。
        threshold: 用于区分前景和背景的灰度阈值 (默认值: 127)。

    Returns:
        二值化图像矩阵，包含 0 (黑) 和 255 (白)。
    """
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary_img

