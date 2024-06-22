import numpy as np


def make_bg_biggest(labels):
    max_label = np.max(labels)
    background_label = labels[0, 0]
    tem_labels = max_label + 1

    if background_label != max_label:
        labels[labels == max_label] = tem_labels
        labels[labels == background_label] = max_label
        labels[labels == tem_labels] = background_label

    return labels


def arrange_labels(input_array):
    """
    整理标签数组, 将零替换为最大的标签, 并按照步长为1的等差数列重新排列其他标签,然后把背景label调到最大
    """
    # 将零替换为最大的标签
    max_label = np.max(input_array)
    input_array[input_array == 0] = max_label+1

    # 按照步长为1的等差数列重新排列其他标签
    unique_labels = np.unique(input_array)
    arranged_labels = np.arange(1, len(unique_labels) + 1)

    # 创建标签映射字典
    label_mapping = dict(zip(unique_labels, arranged_labels))

    # 对输入数组应用标签映射
    arranged_array = np.vectorize(label_mapping.get)(input_array)

    arranged_array = make_bg_biggest(arranged_array)

    return arranged_array


def separate_labels(input_array):
    """
    将输入数组中的每个唯一标签分离到一个单独的图层中。
    """
    # 获取所有唯一的标签
    unique_labels = np.unique(input_array)

    # 初始化一个3维数组
    height, width = input_array.shape
    num_labels = len(unique_labels)
    separated_layers = np.zeros(
        (num_labels, height, width), dtype=input_array.dtype)

    # 填充每个图层
    for i, label in enumerate(unique_labels):
        # 创建每个标签的图层
        layer = np.where(input_array == label, label, 0)
        separated_layers[i] = layer

    return separated_layers


def unify_non_line_segments(labels, classification):
    """
    将不是线段的标签统一为最后一个标签（背景标签）。
    """
    # 获取最后一个标签（背景标签）
    background_label = labels[0, 0]

    # 遍历每个标签
    for label, is_line_segment in classification.items():
        if not is_line_segment:
            # 将不是线段的标签统一为背景标签
            labels[labels == label] = background_label

    return labels


def extractLabelsFromCross_detect(crossings):
    """
    提取出top和bottom从crossings
    """
    crossings_arr = np.zeros((len(crossings), 3), dtype=int)
    for i, crossing in enumerate(crossings):
        x, y, top_line, bottom_line = crossing
        crossings_arr[i, 0] = top_line[0]
        crossings_arr[i, 1] = bottom_line[0]
        crossings_arr[i, 2] = bottom_line[1]
    return crossings_arr


def unifyCrossings(crossings_arr):
    """
    格式化crossings，使其每一个子数组的第一个数成为从0开始的整数，保持映射不变
    input：
    array([[ 2,  1,  8],
       [ 8,  2,1],
       [ 1,  2,  8]])
    Output:
    array([[ 0,  2,  1],
       [ 1,  0,  2],
       [ 2,  0,  1]])
    """
    unified_crossings = crossings_arr
    unique_labels = np.unique(crossings_arr)
    label_mapping = dict(zip(unique_labels, np.arange(len(unique_labels))))
    for i in range(len(crossings_arr)):
        for j in range(3):
            unified_crossings[i, j] = label_mapping[crossings_arr[i, j]]

    # print("unified_crossings:", unified_crossings)
    # 再给unified_crossings排序，使得第一列是从0开始的整数，保持映射不变
    unified_crossings = unified_crossings[np.argsort(unified_crossings[:, 0])]

    return unified_crossings


# if __name__ == '__main__':
#     # 示例使用
#     classification = {1: True, 2: False,
#                       3: True, 4: True}  # 示例 classification 字典
#     labels = np.array([
#         [1, 1, 0, 0],
#         [1, 2, 0, 0],
#         [0, 0, 3, 3],
#         [0, 0, 3, 4]
#     ])  # 示例 labels 数组

#     # 统一不是线段的标签
#     labels_updated = unify_non_line_segments(labels, classification)
#     print(labels_updated)
