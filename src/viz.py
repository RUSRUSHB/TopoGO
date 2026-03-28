"""图像可视化辅助工具模块。

提供了一系列用于展示标签图像、连通分量、随机着色以及交叉点检测结果的工具函数。
支持在图像上自动标注标签 ID、绘制边界框（Bounding Box）以及展示骨架分析结果。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random


def visualize(img: np.ndarray, title: str, colorful: bool = False) -> None:
    """使用 matplotlib 展示图像。

    Args:
        img: 待展示的图像数组。
        title: 图像窗口的标题。
        colorful: 是否以彩色方式显示（True 则使用默认映射，False 则使用灰度映射）。
    """
    plt.figure()
    if colorful:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.show()


def show_each_components(separated_layers: np.ndarray) -> None:
    """逐个展示分割后的连通分量层。

    Args:
        separated_layers: 三维数组，形状为 (通道数/标签数, 高, 宽)，每层代表一个独立的连通分量。
    """
    num_labels, height, width = separated_layers.shape
    for i in range(num_labels):
        layer = separated_layers[i]
        plt.imshow(layer, cmap="gray")
        plt.title(f"Component {i}")
        plt.show()


def _is_surrounded_by_same_label(
    labels: np.ndarray, x: int, y: int, label: int, radius: int = 6
) -> bool:
    """检查指定坐标点周边的邻域是否全部属于同一个标签。

    此函数用于辅助寻找合适的标注位置，确保文字标注不会跨越到其他组件区域。

    Args:
        labels: 标签图像数组。
        x: 目标点横坐标。
        y: 目标点纵坐标。
        label: 期望的标签值。
        radius: 检查半径（邻域范围）。

    Returns:
        若邻域内所有像素点均具有相同的标签值则返回 True，否则返回 False。
    """
    height, width = labels.shape
    for dx in range(-radius, radius):
        for dy in range(-radius, radius):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if labels[ny, nx] != label:
                    return False
    return True


def visualize_labels_with_random_color_and_text(labels: np.ndarray) -> None:
    """为标签图像进行随机着色并标注其 ID。

    该函数会为每个唯一的标签分配一个随机颜色，并在每个组件的中心（通过寻找符合全包围条件的点）标注其 ID 数字。

    Args:
        labels: 输入的二维标签图像数组。
    """
    unique_labels = np.unique(labels)
    color_dict = {label: np.random.rand(3) for label in unique_labels}

    height, width = labels.shape
    colored_image = np.ones((height, width, 3))

    for label in unique_labels:
        mask = labels == label
        colored_image[mask] = color_dict[label]

    plt.figure(figsize=(10, 10))
    plt.imshow(colored_image)

    for label in unique_labels:
        mask = labels == label
        if np.any(mask):
            yx_pairs = np.argwhere(mask)
            for y, x in yx_pairs:
                if _is_surrounded_by_same_label(labels, x, y, label):
                    plt.text(
                        x, y, str(label),
                        color="white", fontsize=12,
                        ha="center", va="center",
                    )
                    break

    patches = [
        mpatches.Patch(color=color_dict[label], label=f"Label {label}")
        for label in unique_labels
    ]
    plt.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    plt.axis("off")
    plt.show()


def visualize_labels_with_crossings(
    labels: np.ndarray,
    crossings: list,
    window_size: int = 5,
) -> None:
    """展示标签图像及检测到的交叉点位置。

    Args:
        labels: 输入的二维标签图像数组。
        crossings: 交叉点信息列表，每个元素包含定位坐标及涉及的标签。
        window_size: 绘制交叉点标志框时使用的窗口大小。
    """
    unique_labels = np.unique(labels)
    height, width = labels.shape
    color_map = {}

    for label in unique_labels:
        if label == unique_labels[-1]:
            color_map[label] = (0, 0, 0)  # 背景设为黑色
        else:
            color_map[label] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    for label in unique_labels:
        color_image[labels == label] = color_map[label]

    # 在图像上添加文本标注
    for label in unique_labels:
        positions = np.argwhere(labels == label)
        if len(positions) > 0:
            y, x = positions[random.randint(0, len(positions) - 1)]
            if np.all(
                labels[
                    max(y - 1, 0) : min(y + 2, height),
                    max(x - 1, 0) : min(x + 2, width),
                ]
                == label
            ):
                cv2.putText(
                    color_image, str(label), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1,
                )

    # 绘制交叉点检测框
    half_window = window_size // 2
    for crossing in crossings:
        x, y, _, _ = crossing
        top_left = (x - half_window, y - half_window)
        bottom_right = (x + half_window, y + half_window)
        cv2.rectangle(color_image, top_left, bottom_right, (0, 255, 0), 2)

    plt.imshow(color_image)
    plt.title("Crossings with Sliding Window Visualization")
    plt.show()

