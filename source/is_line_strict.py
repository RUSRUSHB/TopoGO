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
    # 确保掩码是单通道的8位无符号整数图像
    label = layer.max()
    mask = layer.astype(np.uint8)

    # 查找轮廓
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空白图像来绘制轮廓
    contour_img = np.zeros_like(mask)

    # 绘制轮廓
    cv2.drawContours(contour_img, contours, -1, (255), thickness=cv2.FILLED)

    # 距离变换
    dist_transform = cv2.distanceTransform(
        contour_img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # 查找距离变换结果中的最大值和对应位置
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)

    # 查找轮廓
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return label, 0  # 没有找到任何轮廓

    contour = contours[0]
    plotTheCircle = False
    if (plotTheCircle):
        # plot the contour with circle in a new image
        img = np.zeros_like(mask)
        cv2.drawContours(img, [contour], -1, 255, 1)
        cv2.circle(img, max_loc, int(max_val), 255, 1)
        plt.imshow(img, cmap='gray')
        plt.show()

    return label, int(max_val)


def classify_segments(segmented_image):

    classification = {}
    radii = np.zeros(len(segmented_image))

    # Calculate radii for each segment
    for i, layer in enumerate(segmented_image):
        label, radii[i] = is_line_segment(layer)
        classification[label] = radii[i]
    # print(f'radii: {radii}')
    # Sort radii and remove outliers
    radii_sorted = np.sort(radii)
    percentile = int(len(radii_sorted) * 0.1)
    if (percentile == 0):
        percentile = 1
    radii_filtered = radii_sorted[percentile:-percentile]

    # Check if there are enough segments
    if len(radii_filtered) < 1:
        return "The number of line segments is less than 1"

    # Determine line width
    line_width = radii_filtered[0]
    # print(f'line_width: {line_width}')

    # Classify segments based on line width
    for label, radius in classification.items():
        if abs(radius - line_width) < 0.4 * line_width:
            classification[label] = True
        else:
            classification[label] = False

    return classification
