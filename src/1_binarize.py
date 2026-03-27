"""步骤 1: 图像二值化处理。

采用一个固定的阈值参数，将输入的灰度特征图像转化为仅含黑白两色的二值化格式。
"""

import cv2
import numpy as np


def binarize(img: np.ndarray, threshold: int = 127) -> np.ndarray:
    """利用固定的灰度阈值使得一幅输入图片完成二值化处理。

    Args:
        img: 输入的初步灰度图数组。
        threshold: 区分前景和背景像素用的强度阈值界限 (默认值: 127)。

    Returns:
        包含 0 像素(黑) 和 255 像素(白) 的纯二值化图像矩阵。
    """
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary_img
