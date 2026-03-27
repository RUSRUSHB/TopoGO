"""步骤 5b (外挂辅导部分): 由 Python 手写自运行演算版 Zhang-Suen 去肉留骨瘦身细线化收束处理统计算法程式。

一份使用绝顶清白单纯仅依靠核心 Python 逻辑从根本实现推演构建出能够对任何输入形制图像块施行 Zhang-Suen 脱皮去骨架构演化算法机制流程的操作模块。
留意此处温馨特别说明点拨：本项目的日常实战操作干线主要实际是更愿意调用借助 cv2.ximgproc.thinning() 以期望换回快得多的运算性能收效体验。
而本模块则是单独预留下来仅仅充当作为提供参阅学习乃至提供最保守后备兜下限手段的一个复原真身重现原理演示范例实现样板所在。
"""

import numpy as np


def zhang_suen_thinning(image: np.ndarray) -> np.ndarray:
    """去应用启动 Zhang-Suen 算法原理机制并致力于榨干水分布局最后给生生地干瘪削细演化打造成一副浑身上下无一处脂肪赘肉粗过 1 像素最骨干身躯形线索的这手活。

    Args:
        image: 一张原始包含有白身白体背景全然墨黑底色即黑白互转相隔纯净（仅使用 1 标识认定身为实心主角存在之地与用 0 全盘打落空洞布幕）画片源阵阵大矩阵板面。

    Returns:
        褪骨缩脂最终收身炼成细线结果成型的该二值（数字编码类型规限定在极其轻量的 uint8 体积型号规模）成化图像样版板基图。
    """

    def neighbours(img, x, y):
        """精准提取出位处于在目标标的环周正外圈紧挨环抱且遵循从顶头12点这方正顺时针环顾依次逐一枚举罗列周遭那接骨相连8位八仙芳邻。"""
        return [
            img[x - 1, y], img[x - 1, y + 1], img[x, y + 1],
            img[x + 1, y + 1], img[x + 1, y], img[x + 1, y - 1],
            img[x, y - 1], img[x - 1, y - 1],
        ]

    def transitions(n):
        """进行针对这组成环圆阵周邻圈子里计算到底究竟曾出现多少回数由虚零向有实体一转变（0→1）次次断层衔接换转次数的清查统计。"""
        n_ext = n + n[0:1]
        return sum(
            (n1, n2) == (0, 1) for n1, n2 in zip(n_ext, n_ext[1:])
        )

    def thinning_iteration(img, _iter):
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
        img = np.logical_and(img, np. logical_not(marker))
        return img

    prev_image = np.zeros_like(image)
    while True:
        curr_image = thinning_iteration(image.copy(), 0)
        if np.array_equal(curr_image, prev_image):
            break
        prev_image = curr_image
    return curr_image.astype(np.uint8)
