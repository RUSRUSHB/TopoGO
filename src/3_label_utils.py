"""步骤 3: 标签处理与实用工具函数集。

包含在图像分割完成后，对组件标签进行排序、分层提取、非线性组件过滤以及交叉点信息格式化的辅助函数。
"""

import numpy as np


def make_bg_biggest(labels: np.ndarray) -> np.ndarray:
    """确保背景标签（通常以左上角像素为准）在所有标签中具有最大值。

    由于后续处理管线约定背景为最大值，此函数用于维护该一致性。

    Args:
        labels: 二维标签图像数组。

    Returns:
        背景标签已置换为最大值的标签数组。
    """
    max_label = np.max(labels)
    background_label = labels[0, 0]
    temp_label = max_label + 1

    if background_label != max_label:
        labels[labels == max_label] = temp_label
        labels[labels == background_label] = max_label
        labels[labels == temp_label] = background_label

    return labels


def arrange_labels(input_array: np.ndarray) -> np.ndarray:
    """重新映射标签，使其从 1 开始连续递增，并确保背景标签为最大值。

    Args:
        input_array: 包含不连续或随机标签的原始数组。

    Returns:
        标签已归一化且背景为最大值的数组。
    """
    max_label = np.max(input_array)
    input_array[input_array == 0] = max_label + 1

    unique_labels = np.unique(input_array)
    arranged_labels = np.arange(1, len(unique_labels) + 1)

    label_mapping = dict(zip(unique_labels, arranged_labels))
    arranged_array = np.vectorize(label_mapping.get)(input_array)
    arranged_array = make_bg_biggest(arranged_array)

    return arranged_array


def separate_labels(input_array: np.ndarray) -> np.ndarray:
    """将标签图像拆分为多个单标签层（One-hot 风格的分层）。

    Args:
        input_array: 二维标签图像数组。

    Returns:
        三维数组，形状为 (标签总数, 高, 宽)，每一层仅包含一个特定标签的像素。
    """
    unique_labels = np.unique(input_array)
    height, width = input_array.shape
    num_labels = len(unique_labels)
    separated_layers = np.zeros(
        (num_labels, height, width), dtype=input_array.dtype
    )

    for i, label in enumerate(unique_labels):
        layer = np.where(input_array == label, label, 0)
        separated_layers[i] = layer

    return separated_layers


def unify_non_line_segments(
    labels: np.ndarray, classification: dict[int, bool]
) -> np.ndarray:
    """将识别为“非线条”的组件合并到背景中。

    Args:
        labels: 二维标签图像数组。
        classification: 标签分类字典，Key 为标签 ID，Value 为布尔值（True 表示是线条）。

    Returns:
        过滤掉干扰组件后的标签数组。
    """
    background_label = labels[0, 0]

    for label, is_line in classification.items():
        if not is_line:
            labels[labels == label] = background_label

    return labels


def extract_labels_from_crossings(crossings: list) -> np.ndarray:
    """从交叉点检测结果中提取涉及的标签 ID。

    将检测到的复杂交叉结构转换为标准的 [上方标签, 下方标签1, 下方标签2] 矩阵格式。

    Args:
        crossings: 包含 (x, y, 上方线标签, 下方线标签列表) 的原始检测列表。

    Returns:
        形状为 (N, 3) 的整数数组。
    """
    crossings_arr = np.zeros((len(crossings), 3), dtype=int)
    for i, crossing in enumerate(crossings):
        x, y, top_line, bottom_line = crossing
        crossings_arr[i, 0] = top_line[0]
        crossings_arr[i, 1] = bottom_line[0]
        crossings_arr[i, 2] = bottom_line[1]
    return crossings_arr


def unify_crossings(crossings_arr: np.ndarray) -> np.ndarray:
    """将交叉点记录中的标签重新映射为从 0 开始的连续整数，并按上方线编号进行排序。

    Args:
        crossings_arr: 原始交叉点标签关系矩阵 (N, 3)。

    Returns:
        映射并排序后的规范化交叉关系矩阵。
    """
    unified = crossings_arr.copy()
    unique_labels = np.unique(crossings_arr)
    label_mapping = dict(zip(unique_labels, np.arange(len(unique_labels))))

    for i in range(len(crossings_arr)):
        for j in range(3):
            unified[i, j] = label_mapping[crossings_arr[i, j]]

    # 按第一列（上方交叉线）进行升序排序
    unified = unified[np.argsort(unified[:, 0])]
    return unified
