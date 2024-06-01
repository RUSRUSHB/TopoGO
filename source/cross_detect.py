import cv2
import numpy as np
import matplotlib.pyplot as plt


def is_crossing(window, window_size):
    unique_labels, label_counts = np.unique(window, return_counts=True)
    background_label = unique_labels[-1]
    foreground_labels = unique_labels[unique_labels != background_label]
    # 检查是否有正确数量的前景标签
    if len(foreground_labels) !=3:
        return False
    # 所有标签的面积都必须大于一个值
    for label in unique_labels:
        if np.sum(window == label) < 30:
            return False
    # 最大标签的面积必须大于一个值
    if np.sum(window == np.max(window)) < 300:
        return False




    # 创建围绕窗口的一个数组
    # edge_labels = np.concatenate([window[0:-1, 0], window[-1, :], window[-2:0:-1, -1], window[0, -1:0:-1]])
    # edge_labels_shift = np.roll(edge_labels, len(edge_labels) // 2)
    # # 去除背景标签




    # edge_labels = edge_labels[edge_labels != background_label]
    # edge_labels_shift = edge_labels_shift[edge_labels_shift != background_label]

    # # 兼并相同的标签，保留交替信息
    # # 遍历检查，如果当前标签和上一个标签一样，就删掉当前这一个
    
    # i = 0
    # while i < len(edge_labels) - 1:
    #     if edge_labels[i] == edge_labels[i + 1]:
    #         edge_labels = np.delete(edge_labels, i)
    #     else:
    #         i += 1
    # i = 0
    # while i < len(edge_labels_shift) - 1:
    #     if edge_labels_shift[i] == edge_labels_shift[i + 1]:
    #         edge_labels_shift = np.delete(edge_labels_shift, i)
    #     else:
    #         i += 1
    # # print(f"edge_labels: {edge_labels}")
    # # print(f"edge_labels_shift: {edge_labels_shift}")
    # # 如果第一个标签和最后一个标签一样，就删掉最后一个
    # if edge_labels[0] == edge_labels[-1]:
    #     edge_labels = edge_labels[:-1]


    # unique_labels = np.unique(edge_labels)
    # # tolist
    # edge_labels = edge_labels.tolist()
    # edge_labels_shift = edge_labels_shift.tolist()
    # # print(edge_labels)
    # # print(f"unique_labels: {unique_labels}")
    # up_label, down_label_1, down_label_2 = -1, -1, -1
    # # 查看edge_labels和edge_labels_shift中是否有出现了三次的标签
    # for label in unique_labels:
    #     if edge_labels.count(label) == 3 or edge_labels_shift.count(label) == 3:
    #         up_label = label
    #         break
    #     if edge_labels.count(label) == 2 and edge_labels_shift.count(label) == 2:
    #         up_label = label
    #         break

    # if up_label == -1:
    #     # print("No crossing")
    #     return False

    # [down_label_1, down_label_2] = [x for x in unique_labels if x != up_label]
    
    # l1, l2, l3 = up_label, down_label_1, down_label_2
    # # # 数学分析可得所有可能的交替顺序
    # # possible_labels_5 = [[l1,l3,l1,l2,l1], [l1,l2,l1,l3,l1], [l3,l1,l2,l1,l3], [l2,l1,l3,l1,l2]]
    # # possible_labels_4 = [[l1,l3,l1,l2], [l1,l2,l1,l3], [l3,l1,l2,l1], [l2,l1,l3,l1]]
    # # # print("Lines: ", l1, l2, l3)
    # # # 长度为5的话，为13121,12131,31213,21312
    # # # edge_labels = edge_labels.tolist()
    # # is_true_alternating = False
    # # if len(edge_labels) == 5:
    # #     # print('Length 5')
    # #     for label in possible_labels_5:
    # #         if edge_labels == label:
    # #             # print('True label 5')
    # #             is_true_alternating = True
    # #     # print('False label 5')
    # # # 长度为4的话，为1312,1213,3121,2131
    # # elif len(edge_labels) == 4:
    # #     # print('Length 4')
    # #     for label in possible_labels_4:
    # #         if edge_labels == label:
    # #             # print('True label 4')
    # #             is_true_alternating = True
        
    # #     # print('False label 4')
    # possible_labels = [[l1,l2,l1,l3],[l1,l3,l1,l2],[l2,l1,l3,l1],[l3,l1,l2,l1]]
    # for label in possible_labels:
    #     if edge_labels == label:
    #         is_true_alternating = True
    # if not is_true_alternating:
    #     return False
    # # print('pass alternating')

    # # 不同前景标签的相互最小距离必须小于5
    # for i in range(len(foreground_labels)):
    #     for j in range(i+1, len(foreground_labels)):
    #         label1 = foreground_labels[i]
    #         label2 = foreground_labels[j]
    #         label1_coords = np.argwhere(window == label1)
    #         label2_coords = np.argwhere(window == label2)
    #         min_dist = np.min(np.linalg.norm(label1_coords[:, None] - label2_coords, axis=-1))
    #         if min_dist > 20:
    #             # print(f"min_dist: {min_dist}")
    #             return False

    return True


def find_crossings(labels, window_size=5):
    height, width = labels.shape
    half_window = window_size // 2
    crossings = []

    # for y in range(half_window, height - half_window, int(half_window/6)):
    #     for x in range(half_window, width - half_window, int(half_window/6)):
    
    for y in range(half_window, height - half_window, 1):
        for x in range(half_window, width - half_window, 1):
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
    # if max_area_label != max_edge_label:
    #     top_line_labels.append(max_edge_label)

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
            # only one topLine and two buttomLine, examine this
            if (len(top_line) != 1 or len(bottom_line) != 2):
                continue
            unique_crossings.append((x, y, top_line, bottom_line))
            # print(
                # f"Crossing at ({x}, {y}): Top line label(s): {top_line}, Bottom line label(s): {bottom_line}")

    # return a newthing call crossings_arr, crossings_arr is a 2d numpy array, first is top_line, second and third is bottom_line
    # crossings_arr = np.zeros((len(unique_crossings), 3), dtype=int)
    # for i, crossing in enumerate(unique_crossings):
    #     x, y, top_line, bottom_line = crossing
    #     crossings_arr[i, 0] = top_line[0]
    #     crossings_arr[i, 1] = bottom_line[0]
    #     crossings_arr[i, 2] = bottom_line[1]

    return unique_crossings  # , crossings_arr
