"""步骤 0: 图像读写基础工具。

利用 OpenCV 库提供简单的图像文件加载和保存操作功能。
"""

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """以灰度模式加载本地路径所指向的图像文件。

    Args:
        image_path: 图片存放的路径字符串。

    Returns:
        作为 numpy 二维数组形式返回的灰度图像。
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def save_image(image: np.ndarray, image_path: str) -> None:
    """将图像数组写入到磁盘保存。

    Args:
        image: 等待被保存的输出图像数组。
        image_path: 要保存文件的目标存放路径。
    """
    cv2.imwrite(image_path, image)
