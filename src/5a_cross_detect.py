"""步骤 5a: 基于滑动窗口的交叉点检测。

Pipeline A 方案：通过在标签图像上移动滑动窗口，检测多个标签汇聚的区域。
分析窗口内标签的分布及其在边缘的排列顺序，识别出交叉点位置，并判定哪条线段位于上方（遮挡者），
哪两条线段位于下方（被遮挡者）。
"""

import cv2
import numpy as np


def is_crossing(window: np.ndarray, window_size: int) -> bool:
    """判断当前窗口区域是否包含符合特征的交叉点。

    一个标准的交叉点应包含三个不同的前景标签，并在窗口边缘呈现特定的排列顺序
    （通常表现为一条完整线段横跨窗口，另两条线段被截断）。

    Args:
        window: 当前滑动窗口内的局部图像块。
        window_size: 滑动窗口的边长。

    Returns:
        若窗口内特征符合交叉点判定标准则返回 True。
    """
    unique_labels, label_counts = np.unique(window, return_counts=True)
    background_label = unique_labels[-1]
    foreground_labels = unique_labels[unique_labels != background_label]

    # 前景线段数量必须恰好为 3
    if len(foreground_labels) != 3:
        return False

    # 每个标签必须占据足够的像素数量，以减少噪声干扰
    for label in unique_labels:
        if np.sum(window == label) < 5:
            return False

    # 背景区域应占据合理的比例
    if np.sum(window == np.max(window)) < 20:
        return False

    # 提取窗口边缘的标签序列（逆时针遍历）
    edge_labels = np.concatenate([
        window[0:-1, 0],
        window[-1, :],
        window[-2:0:-1, -1],
        window[0, -1:0:-1],
    ])
    edge_labels_shift = np.roll(edge_labels, len(edge_labels) // 2)

    # 忽略序列中的背景标签
    edge_labels = edge_labels[edge_labels != background_label]
    edge_labels_shift = edge_labels_shift[edge_labels_shift != background_label]

    # 合并连续相同的标签（去重）
    def merge_consecutive(arr):
        i = 0
        while i < len(arr) - 1:
            if arr[i] == arr[i + 1]:
                arr = np.delete(arr, i)
            else:
                i += 1
        return arr

    edge_labels = merge_consecutive(edge_labels)
    edge_labels_shift = merge_consecutive(edge_labels_shift)

    unique_labels = np.unique(edge_labels)
    edge_labels = edge_labels.tolist()
    edge_labels_shift = edge_labels_shift.tolist()

    # 寻找位于上方的标签（该标签在边缘序列中应多次出现或跨越相对边缘）
    up_label = -1
    for label in unique_labels:
        if edge_labels.count(label) == 3 or edge_labels_shift.count(label) == 3:
            up_label = label
            break
        if edge_labels.count(label) == 2 and edge_labels_shift.count(label) == 2:
            up_label = label
            break

    if up_label == -1:
        return False

    [down_label_1, down_label_2] = [x for x in unique_labels if x != up_label]

    # 验证边缘排列是否符合交叉交替特征
    l1, l2, l3 = up_label, down_label_1, down_label_2
    possible_5 = [
        [l1, l3, l1, l2, l1], [l1, l2, l1, l3, l1],
        [l3, l1, l2, l1, l3], [l2, l1, l3, l1, l2],
    ]
    possible_4 = [
        [l1, l3, l1, l2], [l1, l2, l1, l3],
        [l3, l1, l2, l1], [l2, l1, l3, l1],
    ]

    is_valid = False
    if len(edge_labels) == 5:
        is_valid = edge_labels in possible_5
    elif len(edge_labels) == 4:
        is_valid = edge_labels in possible_4

    if not is_valid:
        return False

    # 检查窗口内的连通分量是否足够靠近
    for i in range(len(foreground_labels)):
        for j in range(i + 1, len(foreground_labels)):
            coords_i = np.argwhere(window == foreground_labels[i])
            coords_j = np.argwhere(window == foreground_labels[j])
            min_dist = np.min(
                np.linalg.norm(coords_i[:, None] - coords_j, axis=-1)
            )
            if min_dist > 50:
                return False

    return True


def find_crossings(
    labels: np.ndarray, window_size: int = 5
) -> list[tuple[int, int, tuple]]:
    """扫描整个标签图像，寻找并记录所有潜在的交叉点位置。

    Args:
        labels: 二维标签图像。
        window_size: 检测窗口的大小。

    Returns:
        交叉点列表，包含 (x坐标, y坐标, 涉及的标签元组)。
    """
    height, width = labels.shape
    half_window = window_size // 2
    crossings = []

    for y in range(half_window, height - half_window, 1):
        for x in range(half_window, width - half_window, 1):
            window = labels[
                y - half_window : y + half_window + 1,
                x - half_window : x + half_window + 1,
            ]
            if is_crossing(window, window_size):
                unique = tuple(
                    sorted(np.unique(window[window != np.max(window)]))
                )
                crossings.append((x, y, unique))

    return crossings


def determine_lines(
    window: np.ndarray,
) -> tuple[list[int], list[int]]:
    """在确认的交叉点窗口内，判定各线段的叠放顺序。

    通常认为未被遮断、且在窗口内面积占比最大的线段位于上方。

    Args:
        window: 包含交叉点的局部窗口图像。

    Returns:
        包含 (上方线标签列表, 下方线标签列表) 的元组。
    """
    background_label = np.max(window)

    top_edge = window[0, :]
    left_edge = window[:, 0]
    right_edge = window[:, -1]
    bottom_edge = window[-1, :]

    edge_labels = np.concatenate([top_edge, left_edge, right_edge, bottom_edge])
    unique_edge_labels, edge_label_counts = np.unique(
        edge_labels, return_counts=True
    )

    edge_dict = {
        label: count
        for label, count in zip(unique_edge_labels, edge_label_counts)
        if label != background_label
    }

    if not edge_dict:
        return [], []

    # 统计窗口内每个前景标签的像素面积
    area_dict = {
        label: np.sum(window == label) for label in edge_dict.keys()
    }

    # 面积最大者通常为覆盖在最上方的完整线段
    top_label = max(area_dict, key=area_dict.get)
    top_line = [top_label]
    bottom_line = [l for l in edge_dict.keys() if l != top_label]

    return top_line, bottom_line


def process_image(
    labels: np.ndarray, window_size: int = 5
) -> list[tuple[int, int, list, list]]:
    """完整的交叉点检测与分析流程：扫描全图、定位节点并判定上下层关系。

    Args:
        labels: 归一化后的标签图像。
        window_size: 滑动检测窗口的大小。

    Returns:
        格式化的交叉点详情列表，内容包含 (x, y, 上方线标签, 下方线标签列表)。
    """
    crossings = find_crossings(labels, window_size)
    seen_combinations = set()
    unique_crossings = []

    for crossing in crossings:
        x, y, unique_labels = crossing
        if unique_labels not in seen_combinations:
            seen_combinations.add(unique_labels)
            half = window_size // 2
            window = labels[
                y - half : y + half + 1,
                x - half : x + half + 1,
            ]
            top_line, bottom_line = determine_lines(window)

            if len(top_line) != 1 or len(bottom_line) != 2:
                continue
            unique_crossings.append((x, y, top_line, bottom_line))

    return unique_crossings
