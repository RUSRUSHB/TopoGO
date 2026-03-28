"""步骤 2: 连通分量检测算法。

基于两遍扫描法 (Two-pass) 和并查集 (Union-Find) 原理，提取二值图像中的前景连通分量。
"""

import numpy as np
import cv2


def connected_components(img: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """检测二值图像中的连通区域并进行标记。

    使用两遍扫描算法，并结合路径压缩和按秩合并优化的并查集。
    注意：仅像素值为 255 的点被视为前景。

    Args:
        img: 黑白二值图像 (0 为背景, 255 为前景)。

    Returns:
        包含以下元素的元组: (labels 标记数组, component_sizes 连通分量大小字典)。
        labels 矩阵中的标签从 1 开始分配。
    """
    height, width = img.shape
    labels = np.zeros((height, width), dtype=int)
    current_label = 1

    def find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 路径压缩
            x = parent[x]
        return x

    def union(parent, rank, x, y):
        root_x = find(parent, x)
        root_y = find(parent, y)
        if root_x != root_y:
            if rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            elif rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            else:
                parent[root_y] = root_x
                rank[root_x] += 1

    parent = {}
    rank = {}

    # 第一轮遍历: 初步标记并维护并查集中的等价关系
    for y in range(height):
        for x in range(width):
            if img[y, x] == 255:
                neighbors = []
                if x > 0 and labels[y, x - 1] > 0:
                    neighbors.append(labels[y, x - 1])
                if y > 0 and labels[y - 1, x] > 0:
                    neighbors.append(labels[y - 1, x])

                if not neighbors:
                    labels[y, x] = current_label
                    parent[current_label] = current_label
                    rank[current_label] = 0
                    current_label += 1
                else:
                    smallest_label = min(neighbors)
                    labels[y, x] = smallest_label
                    for neighbor in neighbors:
                        union(parent, rank, smallest_label, neighbor)

    # 第二轮遍历: 解析并查集，使所有连通像素点共享同一个根标签
    for y in range(height):
        for x in range(width):
            if labels[y, x] > 0:
                labels[y, x] = find(parent, labels[y, x])

    # 统计每个连通分量的像素数量（大小）
    component_sizes = {}
    for label in np.unique(labels):
        if label == 0:
            continue
        component_sizes[label] = np.sum(labels == label)

    # 重新映射标签，确保它们是 1, 2, 3... 连续的整数映射
    new_labels = np.zeros_like(labels)
    label_mapping = {old: new for new, old in enumerate(np.unique(labels))}
    for old_label, new_label in label_mapping.items():
        new_labels[labels == old_label] = new_label

    new_component_sizes = {
        label_mapping[old]: size for old, size in component_sizes.items()
    }

    return new_labels, new_component_sizes


def visualize_components(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """为检测到的连通分量随机着色，用于可视化。

    Args:
        img: 原始灰度图像（仅用于获取尺寸）。
        labels: 连通分量标签数组。

    Returns:
        着色后的 RGB 图像数组。
    """
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    colors = np.random.randint(0, 255, size=(num_labels, 3))
    colors[0] = [0, 0, 0]  # 背景始终设为黑色

    output_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i, label in enumerate(unique_labels):
        if label == 0:
            continue
        output_img[labels == label] = colors[i]

    return output_img


def segment_image(binary_img: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray]:
    """对二值图像进行分割，检测连通区域。

    Args:
        binary_img: 输入的二值化图像。

    Returns:
        包含 (labels 标签数组, 连通分量大小字典, 可视化彩色图) 的元组。
    """
    labels, component_sizes = connected_components(binary_img)
    output_img = visualize_components(binary_img, labels)
    return labels, component_sizes, output_img

