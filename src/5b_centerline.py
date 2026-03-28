"""步骤 5b: 基于中心线分析法识别重叠交叉点的工作流。

属于 Pipeline B 方案：首先对各连通分量进行骨架化（Skeletonization）处理，将其缩减为单像素宽度的中心线。
接着定位中心线的端点，通过搜索端点间的邻近关系，判断是否存在被第三方组件遮挡导致的断裂，
从而识别出图像中隐藏的交叉点位置。
"""

import cv2
import numpy as np
from scipy.spatial import KDTree
import skimage.draw


def extract_centerline(binary_image: np.ndarray) -> np.ndarray:
    """提取二值图像中前景区域的单像素宽度骨架（中心线）。

    实现方法：计算离距离变换（Distance Transform），结合阈值分割与 skimage 的细化（Thinning）算法。

    Args:
        binary_image: 输入的单通道二值图像。

    Returns:
        提取出的单像素宽度骨架图（像素值为 0 或 255）。
    """
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 3)
    dist_transform = cv2.normalize(
        dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX
    )
    _, skeleton = cv2.threshold(
        dist_transform, 0.2 * np.max(dist_transform), 1.0, cv2.THRESH_BINARY
    )
    skeleton = (skeleton * 255).astype(np.uint8)
    skeleton = cv2.ximgproc.thinning(skeleton)
    return skeleton


def extract_centerlines_for_labels(labels: np.ndarray) -> np.ndarray:
    """为标签图像中的每个连通分量提取其中心线骨架。

    Args:
        labels: 带有连通分量标签的图像数组。

    Returns:
        包含各组件骨架的图像数组，骨架像素保留其原始标签值，其余为 0。
    """
    unique_labels = np.unique(labels)
    centerlines = np.zeros_like(labels, dtype=np.uint8)
    background_label = labels[5, 5]

    for label in unique_labels:
        if label == background_label:
            continue
        mask = (labels == label).astype(np.uint8) * 255
        centerline = extract_centerline(mask)
        centerlines[centerline > 0] = label

    return centerlines


def get_endpoints(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """识别骨架图中的端点（即 8-邻域内仅有一个相邻像素的点）。

    Args:
        skeleton: 单像素宽度的骨架图像。

    Returns:
        包含端点坐标 (y, x) 的列表。
    """
    endpoints = []
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x] > 0:
                neighborhood = skeleton[y - 1 : y + 2, x - 1 : x + 2]
                if np.sum(neighborhood) == 2:  # 包含点自身及 1 个邻居
                    endpoints.append((y, x))
    return endpoints


def dilate_line(
    image: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], width: int = 3
) -> np.ndarray:
    """在布尔图像中绘制连接两点的粗线段。

    Args:
        image: 目标布尔图像。
        pt1: 起始点坐标 (y, x)。
        pt2: 终止点坐标 (y, x)。
        width: 线段扩充宽度。

    Returns:
        修改后的布尔图像。
    """
    rr, cc = skimage.draw.line(pt1[0], pt1[1], pt2[0], pt2[1])
    for dr in range(-width // 2, width // 2 + 1):
        for dc in range(-width // 2, width // 2 + 1):
            r, c = rr + dr, cc + dc
            valid = (r >= 0) & (r < image.shape[0]) & (c >= 0) & (c < image.shape[1])
            image[r[valid], c[valid]] = True
    return image


def map_labels_to_consecutive_integers(
    crossings: list[list[int]],
) -> tuple[list[list[int]], dict[int, int]]:
    """将交叉点记录中的原始标签映射为从 0 开始的连续整数。

    Args:
        crossings: 交叉点信息列表，每个元素为 [上方标签, 下方标签A, 下方标签B]。

    Returns:
        包含 (映射后的交叉点列表, 原始标签到新编号的映射字典) 的元组。
    """
    unique_labels = set()
    for crossing in crossings:
        unique_labels.update(crossing)

    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    mapped = [
        [label_mapping[l1], label_mapping[l2], label_mapping[l3]]
        for l1, l2, l3 in crossings
    ]

    return mapped, label_mapping


def examine_crossings(
    centerlines: np.ndarray, is_visualize: bool = False
) -> list[list[int]]:
    """通过端点邻近性分析来检测并识别图像中的交叉点。

    基本逻辑如下：
    1. 遍历所有骨架端点，寻找属于不同标签且距离较近的端点对。
    2. 在两端点间绘制测试线段并进行膨胀。
    3. 检查该线段跨越了哪些其他标签的组件。
    4. 若刚好跨越了一个独立的第三方标签，则认为该处存在一个交叉点（上方连通，下方被截断）。

    Args:
        centerlines: 包含各组件中心线（及其原始标签）的图像数组。
        is_visualize: 是否开启可视化调试窗口。

    Returns:
        交叉点信息列表，格式为 [[上方组件标签, 下方组件标签1, 下方组件标签2], ...]。
    """
    endpoints = {}
    unique_labels = np.unique(centerlines)
    background_label = centerlines[5, 5]

    for label in unique_labels:
        if label == background_label:
            continue
        mask = centerlines == label
        endpoints[label] = get_endpoints(mask)

    all_endpoints = [
        (y, x, label)
        for label, points in endpoints.items()
        for y, x in points
    ]

    if not all_endpoints:
        return []

    # 可视化准备
    if is_visualize:
        import matplotlib.pyplot as plt
        vis_img = centerlines.copy()
        _, vis_img = cv2.threshold(vis_img, 0, 255, cv2.THRESH_BINARY)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        for y, x, label in all_endpoints:
            cv2.circle(vis_img, (x, y), 2, (0, 0, 255), -1)
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_img)
        plt.axis("off")
        plt.title("Endpoints Visualization")
        plt.show()

    # 构建 KD-tree 进行最近邻查询
    tree = KDTree([(y, x) for y, x, _ in all_endpoints])
    crossings = []
    matched_endpoints = set()

    for y, x, label in all_endpoints:
        distances, indices = tree.query((y, x), k=6)
        for idx in indices[1:]:  # 跳过自身
            ny, nx, nlabel = all_endpoints[idx]
            if nlabel == label:
                continue

            if (y, x) in matched_endpoints or (ny, nx) in matched_endpoints:
                continue

            # 在两个端点之间绘制测试路径
            test_image = np.zeros_like(centerlines, dtype=bool)
            test_image = dilate_line(test_image, (y, x), (ny, nx), width=3)

            # 统计测试路径跨越的标签
            intersecting_labels = set(centerlines[test_image])
            intersecting_labels.discard(background_label)
            intersecting_labels.discard(label)
            intersecting_labels.discard(nlabel)

            # 有效交叉点判定：端点对之间恰好被一个第三方标签阻断
            if len(intersecting_labels) == 1:
                crossing = [list(intersecting_labels)[0], label, nlabel]
                crossing[1], crossing[2] = sorted([label, nlabel])

                if crossing not in crossings:
                    crossings.append(crossing)
                    matched_endpoints.update([(y, x), (ny, nx)])

                    if is_visualize:
                        cv2.circle(vis_img, (x, y), 2, (0, 255, 0), -1)
                        cv2.circle(vis_img, (nx, ny), 2, (0, 255, 0), -1)
                        cv2.line(vis_img, (x, y), (nx, ny), (0, 255, 0), 1)

    if is_visualize:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_img)
        plt.axis("off")
        plt.title("Crossings Visualization")
        plt.show()

    return crossings

