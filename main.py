from source.binarize import binarize
from source.connect_component_detect import segment_image
from source.io import load_image, save_image
from source.is_line import is_line_segment, classify_segments
from source.util import *
from source.alex import alex_polynomial, straighten

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import KDTree
import skimage.draw  # 这里导入skimage.draw而不是skimage.morphology

def visualize_labels_with_random_color_and_text(labels):
    unique_labels = np.unique(labels)
    color_dict = {label: np.random.rand(3,) for label in unique_labels}
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
            for (y, x) in yx_pairs:
                if is_surrounded_by_same_label(labels, x, y, label):
                    plt.text(x, y, str(label), color='black', fontsize=20, ha='center', va='center')
                    break

    patches = [mpatches.Patch(color=color_dict[label], label=f'Label {label}') for label in unique_labels]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.axis('off')
    plt.show()

def is_surrounded_by_same_label(labels, x, y, label):
    height, width = labels.shape
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                if labels[ny, nx] != label:
                    return False
    return True

def extract_centerline(binary_image):
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 3)  # 调整距离变换的半径
    dist_transform = cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    _, skeleton = cv2.threshold(dist_transform, 0.2 * np.max(dist_transform), 1.0, cv2.THRESH_BINARY)  # 调整阈值
    skeleton = (skeleton * 255).astype(np.uint8)
    skeleton = cv2.ximgproc.thinning(skeleton)
    return skeleton

def extract_centerlines_for_labels(labels):
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

def get_endpoints(skeleton):
    endpoints = []
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x] > 0:
                neighborhood = skeleton[y-1:y+2, x-1:x+2]
                if np.sum(neighborhood) == 2:
                    endpoints.append((y, x))
    return endpoints

def dilate_line(image, pt1, pt2, width=3):
    rr, cc = skimage.draw.line(pt1[0], pt1[1], pt2[0], pt2[1])
    for dr in range(-width // 2, width // 2 + 1):
        for dc in range(-width // 2, width // 2 + 1):
            r, c = rr + dr, cc + dc
            valid = (r >= 0) & (r < image.shape[0]) & (c >= 0) & (c < image.shape[1])
            image[r[valid], c[valid]] = True
    

    return image


def map_labels_to_consecutive_integers(crossings):
    # 收集所有的标签
    unique_labels = set()
    for crossing in crossings:
        unique_labels.update(crossing)

    # 将标签映射到连续的整数
    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    # 使用映射转换crossings列表中的标签
    mapped_crossings = [[label_mapping[label1], label_mapping[label2], label_mapping[label3]] for label1, label2, label3 in crossings]

    return mapped_crossings, label_mapping

def examine_crossings(centerlines, is_visualize=False):
    """
    检查中心线图像中的crossings。
    crossings是指两个不同标签的端点之间存在且仅存在一个其他标签的连线。
    
    参数:
    centerlines - 标记了不同中心线的图像
    is_visualize - 是否可视化中间结果，默认为False
    
    返回:
    crossings列表，格式为[label1, label2, label3]，
    其中label1是连线上存在的其他标签，label2和label3分别是两个端点对应的标签
    """
    # 创建一个字典来存储每个标签的端点
    endpoints = {}
    # 获取中心线图像中的所有唯一标签
    unique_labels = np.unique(centerlines)
    background_label = centerlines[5, 5]
    
    # 对于每一个标签，找到其端点并存储在字典中
    for label in unique_labels:
        if label == background_label:
            continue  # 跳过背景标签
        mask = (centerlines == label)
        endpoints[label] = get_endpoints(mask)

    # 将所有端点转换为一个包含(y, x, label)的列表
    all_endpoints = [(y, x, label) for label, points in endpoints.items() for y, x in points]
    
    if is_visualize:
        # 可视化：使用cv在centerlines图像上绘制所有端点
        centerlines_with_endpoints = centerlines.copy()
        _, centerlines_with_endpoints = cv2.threshold(centerlines_with_endpoints, 0, 255, cv2.THRESH_BINARY)
        centerlines_with_endpoints = cv2.cvtColor(centerlines_with_endpoints, cv2.COLOR_GRAY2BGR)
        for y, x, label in all_endpoints:
            cv2.circle(centerlines_with_endpoints, (x, y), 2, (0, 0, 255), -1)
        plt.figure(figsize=(10, 10))
        plt.imshow(centerlines_with_endpoints)
        plt.axis('off')
        plt.title("Endpoints Visualization")
        plt.show()

    if not all_endpoints:
        return []

    # 创建KD树以便快速查询最近邻
    tree = KDTree([(y, x) for y, x, label in all_endpoints])
    crossings = []
    # 记录已匹配的端点
    matched_endpoints = set()



    # 对于每一个端点，找到最近的一个其他端点
    for y, x, label in all_endpoints:
        distances, indices = tree.query((y, x), k=6)  # 调参：k越大，认为可容忍目标离当前标签越近
        for idx in indices[1:]:  # 跳过第一个索引，因为它是点自身
            ny, nx, nlabel = all_endpoints[idx]
            if nlabel == label:
                continue  # 跳过相同标签的端点

            # 如果这个endpoint已经成功匹配，则跳过
            if (y, x) in matched_endpoints or (ny, nx) in matched_endpoints:
                continue

            # 检查(y, x)和(ny, nx)之间的线段
            test_image = np.zeros_like(centerlines, dtype=bool)
            test_image = dilate_line(test_image, (y, x), (ny, nx), width=3)

            # 可视化：显示测试线段
            # if is_visualize:
            #     test_image_show = test_image.astype(np.uint8) * 255
            #     test_image_show = cv2.cvtColor(test_image_show, cv2.COLOR_GRAY2BGR)
            #     cv2.circle(test_image_show, (x, y), 2, (0, 255, 0), -1)
            #     cv2.circle(test_image_show, (nx, ny), 2, (0, 255, 0), -1)
            #     cv2.line(test_image_show, (x, y), (nx, ny), (0, 255, 0), 1)
            #     plt.figure(figsize=(10, 10))
            #     plt.imshow(test_image_show)
            #     plt.axis('off')
            #     plt.title("Test Line Visualization")
            #     plt.show()

            # 计算膨胀线区域中的唯一标签
            intersecting_labels = set(centerlines[test_image])
            intersecting_labels.discard(background_label)  # 移除背景标签
            intersecting_labels.discard(label) #移除自己
            intersecting_labels.discard(nlabel) #移除另一个端点的标签

            # 如果在膨胀线区域中仅有一个其他标签，则认为这是一个crossing
            if len(intersecting_labels) == 1:
                crossing = [list(intersecting_labels)[0], label, nlabel]
                # 将label2和label3排序，不要改动label1
                crossing[1], crossing[2] = sorted([label, nlabel])

                if crossing not in crossings:  # 如果排好序的crossing不在crossings中，则添加
                    crossings.append(crossing)
                    # 将匹配过的端点加入到匹配集合中
                    matched_endpoints.update([(y, x), (ny, nx)])

                    if is_visualize:
                        cv2.circle(centerlines_with_endpoints, (x, y), 2, (0, 255, 0), -1)
                        cv2.circle(centerlines_with_endpoints, (nx, ny), 2, (0, 255, 0), -1)
                        cv2.line(centerlines_with_endpoints, (x, y), (nx, ny), (0, 255, 0), 1)
                        cv2.putText(centerlines_with_endpoints, str(crossing[0]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(centerlines_with_endpoints, str(crossing[1]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(centerlines_with_endpoints, str(crossing[2]), (nx, ny), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if is_visualize:
        plt.figure(figsize=(10, 10))
        plt.imshow(centerlines_with_endpoints)
        plt.axis('off')
        plt.title("Crossings Visualization")
        plt.show()

    return crossings


def run_with_all_visualization(img):
    # 读入图像并进行二值化
    bi_img = binarize(img)
    
    # 分割图像并获取标签
    labels, component_sizes, output_img = segment_image(bi_img)
    labels = arrange_labels(labels)
    
    # 可视化分割后的标签图像
    visualize_labels_with_random_color_and_text(labels)

    # 分割标签，进行线段分类和非线段统一
    separated_labels = separate_labels(labels)
    classification = classify_segments(separated_labels)
    unifyLabels = unify_non_line_segments(labels, classification)

    # 提取中心线
    centerline_labels = extract_centerlines_for_labels(unifyLabels)
    centerline_labels = centerline_labels[1:-1, 1:-1]

    # 可视化中心线
    visualize_labels_with_random_color_and_text(centerline_labels)
    
    # 检查交叉点
    crossings = examine_crossings(centerline_labels, True)

    # 将标签映射为连续整数
    mapped_crossings, label_mapping = map_labels_to_consecutive_integers(crossings)

    # 输出映射后的交叉点信息
    print(f'mapped_crossings: {mapped_crossings}')
    print(f'number of crossings: {len(mapped_crossings)}')

    # 进一步处理
    print(np.array(mapped_crossings))

    # 可视化各个阶段的结果（需要取消注释）
    # 可视化二值化后的图像
    plt.imshow(bi_img, cmap='gray')
    plt.title('Binarized Image')
    plt.axis('off')
    plt.show()

    # 可视化分割后的标签图像（含有随机颜色和文本）
    visualize_labels_with_random_color_and_text(labels)

    # 可视化分割后的标签图像（含有标签连接线）
    plt.imshow(output_img)
    plt.title('Segmented Image with Connected Components')
    plt.axis('off')
    plt.show()

    # 可视化中心线
    visualize_labels_with_random_color_and_text(centerline_labels)

# 主程序
if __name__ == '__main__':
    
    img = load_image("img/rolfsen_all/10_164.png")

    bi_img = binarize(img)
    labels, component_sizes, output_img = segment_image(bi_img)
    labels = arrange_labels(labels)
    print(f'labels: {labels}')
    print(f'component_sizes: {component_sizes}')
    visualize_labels_with_random_color_and_text(labels)

    separate_labels = separate_labels(labels)
    classification = classify_segments(separate_labels)
    unifyLabels = unify_non_line_segments(labels, classification)

    # 提取中心线
    centerline_labels = extract_centerlines_for_labels(unifyLabels)
    centerline_labels = centerline_labels[1:-1, 1:-1]

    # 可视化中心线
    visualize_labels_with_random_color_and_text(centerline_labels)
    
    # 检查crossings
    crossings = examine_crossings(centerline_labels, True)
    print(f'crossings: {crossings}')

    mapped_crossings, label_mapping = map_labels_to_consecutive_integers(crossings)

    print(f'mapped_crossings: {mapped_crossings}')
    print(f'number of crossings: {len(mapped_crossings)}')

    # 进一步处理
    print(np.array(mapped_crossings))
    straightened_data = straighten(np.array(mapped_crossings))
    print(f'The polynomial is: {alex_polynomial(straightened_data)}')
