"""步骤 5b (辅助模块): Zhang-Suen 细化算法的 Python 原生实现。

本模块提供了 Zhang-Suen 细化（Thinning）算法的纯 Python 实现，用于将二值图像中的前景区域缩减为单像素宽度的骨架。
备注：在实际生产流程中，本项目优先推荐调用 `cv2.ximgproc.thinning()` 以获得更佳的性能表现。
本代码实现仅作为原理参考及纯 Python 环境下的备选方案。
"""

import numpy as np


def zhang_suen_thinning(image: np.ndarray) -> np.ndarray:
    """对二值图像应用 Zhang-Suen 细化算法，提取单像素宽度的骨架。

    Args:
        image: 输入的二值图像（0 为背景，1 为前景）。

    Returns:
        细化处理后的二值图像（uint8 格式）。
    """

    def neighbours(img, x, y):
        """按照 12 点钟方向顺时针提取 8-邻域的像素值。"""
        return [
            img[x - 1, y], img[x - 1, y + 1], img[x, y + 1],
            img[x + 1, y + 1], img[x + 1, y], img[x + 1, y - 1],
            img[x, y - 1], img[x - 1, y - 1],
        ]

    def transitions(n):
        """计算 8-邻域序列中从 0 到 1 的跳变次数。"""
        n_ext = n + n[0:1]
        return sum(
            (n1, n2) == (0, 1) for n1, n2 in zip(n_ext, n_ext[1:])
        )

    def thinning_iteration(img, _iter):
        """执行单次细化迭代（包含两个子迭代步骤）。"""
        marker = np.zeros_like(img)
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                p2, p3, p4, p5, p6, p7, p8, p9 = n = neighbours(img, i, j)
                if (
                    img[i, j] == 1
                    and 2 <= sum(n) <= 6
                    and transitions(n) == 1
                    and p2 * p4 * p6 == 0
                    and p4 * p6 * p8 == 0
                ):
                    marker[i, j] = 1
        img = np.logical_and(img, np.logical_not(marker))

        marker = np.zeros_like(img)
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                p2, p3, p4, p5, p6, p7, p8, p9 = n = neighbours(img, i, j)
                if (
                    img[i, j] == 1
                    and 2 <= sum(n) <= 6
                    and transitions(n) == 1
                    and p2 * p4 * p8 == 0
                    and p2 * p6 * p8 == 0
                ):
                    marker[i, j] = 1
        img = np.logical_and(img, np.logical_not(marker))
        return img

    prev_image = np.zeros_like(image)
    while True:
        curr_image = thinning_iteration(image.copy(), 0)
        if np.array_equal(curr_image, prev_image):
            break
        prev_image = curr_image
    return curr_image.astype(np.uint8)

