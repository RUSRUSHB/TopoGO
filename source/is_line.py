import cv2
import numpy as np
import matplotlib.pyplot as plt


def separate_labels(input_array):
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
        layer = np.where(input_array == label, 255, 0)  # 将标签像素设为255，其它像素设为0
        separated_layers[i] = layer

    return separated_layers


def is_line_segment(layer):
    label = layer.max()
    mask = layer.astype(np.uint8)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return label, 0, 0  # 没有找到任何轮廓

    contour = contours[0]

    # 距离变换
    contour_img = np.zeros_like(mask)
    cv2.drawContours(contour_img, contours, -1, (255), thickness=cv2.FILLED)
    dist_transform = cv2.distanceTransform(contour_img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, max_val, _, _ = cv2.minMaxLoc(dist_transform)

    # 计算最小外接圆
    (x, y), min_enclosing_radius = cv2.minEnclosingCircle(contour)

    return label, int(max_val), min_enclosing_radius


def classify_segments(segmented_image):
    classification = {}
    radii = np.zeros(len(segmented_image))
    min_enclosing_radii = np.zeros(len(segmented_image))

    # Calculate radii for each segment
    for i, layer in enumerate(segmented_image):
        label, max_val, min_enclosing_radius = is_line_segment(layer)
        radii[i] = max_val
        min_enclosing_radii[i] = min_enclosing_radius
        classification[label] = (max_val, min_enclosing_radius)
    # print(f'radii: {radii}')
    # Sort radii and remove outliers
    radii_sorted = np.sort(radii)
    percentile = int(len(radii_sorted) * 0.1)
    if percentile == 0:
        percentile = 1
    radii_filtered = radii_sorted[percentile:-percentile]

    # Check if there are enough segments
    if len(radii_filtered) < 1:
        return "The number of line segments is less than 1"

    # Determine line width
    line_width = radii_filtered[0]
    # print(f'line_width: {line_width}')

    # Classify segments based on line width and minimum enclosing radius
    for label, (radius, min_enclosing_radius) in classification.items():
        if (abs(radius - line_width) < 0.4 * line_width) and (min_enclosing_radius >= 3 * line_width):
            # if (abs(radius - line_width) < 0.4 * line_width) and (min_enclosing_radius >= 3 * line_width):
            classification[label] = True
        else:
            classification[label] = False

    return classification
