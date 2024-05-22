import cv2
import numpy as np
import matplotlib.pyplot as plt


def is_crossing(window, window_size):
    unique_labels, label_counts = np.unique(window, return_counts=True)
    background_label = unique_labels[-1]
    foreground_labels = unique_labels[unique_labels != background_label]

    # 检查是否有足够数量的前景标签
    if len(foreground_labels) <= 2:
        return False

    # 找到数量最少的两个前景标签
    min_count_indices = np.argsort(label_counts)[:2]  # 最小数量的两个标签
    min_count_labels = unique_labels[min_count_indices]
    min_count_values = label_counts[min_count_indices]
    min_count = window_size * 1.2

    # 检查最少的两个前景标签的数量是否满足要求
    if min(min_count_values) < min_count:
        return False

    return True


def find_crossings(labels, window_size=5):
    height, width = labels.shape
    half_window = window_size // 2
    crossings = []

    for y in range(half_window, height - half_window, int(half_window/6)):
        for x in range(half_window, width - half_window, int(half_window/6)):
            window = labels[y - half_window:y + half_window +
                            1, x - half_window:x + half_window + 1]
            if is_crossing(window, window_size):
                unique_labels = tuple(
                    sorted(np.unique(window[window != np.max(window)])))
                crossings.append((x, y, unique_labels))

    return crossings


def determine_lines(window):
    height, width = window.shape
    background_label = np.max(window)

    # 获取窗口边界上的标签
    top_edge_labels = window[0, :]
    left_edge_labels = window[:, 0]
    right_edge_labels = window[:, -1]
    bottom_edge_labels = window[-1, :]

    edge_labels = np.concatenate(
        [top_edge_labels, left_edge_labels, right_edge_labels, bottom_edge_labels])
    unique_edge_labels, edge_label_counts = np.unique(
        edge_labels, return_counts=True)

    # 排除背景标签
    edge_labels_dict = {label: count for label, count in zip(
        unique_edge_labels, edge_label_counts) if label != background_label}

    if not edge_labels_dict:
        return [], []  # 没有找到前景标签

    # 计算每个标签的面积
    area_dict = {}
    for label in edge_labels_dict.keys():
        area_dict[label] = np.sum(window == label)

    # 找到面积最大的标签
    max_area_label = max(area_dict, key=area_dict.get)

    # 找到边界上像素最多的标签
    max_edge_label = max(edge_labels_dict, key=edge_labels_dict.get)

    # 将面积最大的标签作为topLine
    top_line_labels = [max_area_label]

    # 如果面积最大的标签和边界上像素最多的标签不同，将后者也加入topLine
    if max_area_label != max_edge_label:
        top_line_labels.append(max_edge_label)

    # 其余的标签作为bottomLine
    bottom_line_labels = [
        label for label in edge_labels_dict.keys() if label not in top_line_labels]

    return top_line_labels, bottom_line_labels


def process_image(labels, window_size=5):
    crossings = find_crossings(labels, window_size)
    seen_combinations = set()
    unique_crossings = []
    for crossing in crossings:
        x, y, unique_labels = crossing
        if unique_labels not in seen_combinations:
            seen_combinations.add(unique_labels)
            window = labels[y - window_size//2:y + window_size //
                            2 + 1, x - window_size//2:x + window_size//2 + 1]
            top_line, bottom_line = determine_lines(window)
            unique_crossings.append((x, y, top_line, bottom_line))
            print(
                f"Crossing at ({x}, {y}): Top line label(s): {top_line}, Bottom line label(s): {bottom_line}")

    return unique_crossings
