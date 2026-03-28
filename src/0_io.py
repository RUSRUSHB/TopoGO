"""步骤 0: 图像读写基础工具。

使用 OpenCV 提供简单的图像加载与保存功能。
"""

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """以灰度模式加载图像文件。

    Args:
        image_path: 图像文件的本地路径。

    Returns:
        灰度图像的 NumPy 数组。
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def save_image(image: np.ndarray, image_path: str) -> None:
    """将图像数组保存到指定磁盘路径。

    Args:
        image: 待保存的图像数组。
        image_path: 目标保存路径。
    """
    cv2.imwrite(image_path, image)

