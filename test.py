from source.binarize import binarize
from source.visualize import visualize
from source.connect_component_detect import segment_image
from source.io import load_image, save_image
from source.is_line import is_line_segment, classify_segments
from source.visualize_new import *
from source.util import *
from source.cross_detect import *
from source.alex import *

###
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import KDTree

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
                    plt.text(x, y, str(label), color='white',
                             fontsize=12, ha='center', va='center')
                    break
    patches = [mpatches.Patch(
        color=color_dict[label], label=f'Label {label}') for label in unique_labels]
    plt.legend(handles=patches, bbox_to_anchor=(
        1.05, 1), loc='upper left', borderaxespad=0.)
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
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    dist_transform = cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    _, skeleton = cv2.threshold(dist_transform, 0.4, 1.0, cv2.THRESH_BINARY)
    skeleton = (skeleton * 255).astype(np.uint8)
    skeleton = cv2.ximgproc.thinning(skeleton)
    return skeleton

def extract_centerlines_for_labels(labels):
    unique_labels = np.unique(labels)
    centerlines = np.zeros_like(labels, dtype=np.uint8)
    background_label = labels[0, 0]
    for label in unique_labels:
        if label == background_label:
            continue
        mask = (labels == label).astype(np.uint8) * 255
        centerline = extract_centerline(mask)
        centerlines[centerline > 0] = label
    return centerlines

def remove_border_pixels(image, border_width=1):
    return image[border_width:-border_width, border_width:-border_width]

def get_endpoints(skeleton):
    endpoints = []
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x] > 0:
                neighborhood = skeleton[y-1:y+2, x-1:x+2]
                if np.sum(neighborhood) == 2:
                    endpoints.append((y, x))
    print(endpoints)
    return endpoints

def find_crossings(labels, window_size):
    height, width = labels.shape
    crossings = []

    for y in range(0, height, window_size):
        for x in range(0, width, window_size):
            window = labels[y:y + window_size, x:x + window_size]
            if is_crossing(window, window_size):
                crossings.append((y, x))
    
    return crossings

def is_crossing(window, window_size):
    unique_labels, label_counts = np.unique(window, return_counts=True)
    background_label = unique_labels[-1]
    foreground_labels = unique_labels[unique_labels != background_label]

    if len(foreground_labels) <= 2:
        return False

    min_count_indices = np.argsort(label_counts)[:2]
    min_count_labels = unique_labels[min_count_indices]
    min_count_values = label_counts[min_count_indices]
    min_count = window_size * 1

    if min(min_count_values) < min_count:
        return False

    max_count_index = np.argmax(label_counts)
    max_count_label = unique_labels[max_count_index]
    max_count_value = label_counts[max_count_index]

    if max_count_value < min_count*5:
        return False

    top_edge_labels = window[0, :]
    left_edge_labels = window[:, 0]
    right_edge_labels = window[:, -1]
    bottom_edge_labels = window[-1, :]

    edge_labels = np.concatenate([
        top_edge_labels,
        right_edge_labels[1:-1],
        bottom_edge_labels[::-1],
        left_edge_labels[-2:0:-1]
    ])

    unique_edge_labels = np.unique(edge_labels)

    label_connectivity = {}
    for label in unique_edge_labels:
        if label == background_label:
            continue
        edge_mask = (edge_labels == label).astype(np.uint8)
        num_labels, labels_im = cv2.connectedComponents(edge_mask.reshape(-1, 1))
        label_connectivity[label] = num_labels - 1

    disconnected_labels = [label for label, count in label_connectivity.items() if count > 1]
    connected_labels = [label for label, count in label_connectivity.items() if count == 1]

    if len(disconnected_labels) != 1 or len(connected_labels) != 2:
        return False

    if len(disconnected_labels) > 0:
        up_label = disconnected_labels[0]
        down_labels = [x for x in unique_labels if x != up_label and x != background_label]
        if len(down_labels) < 2:
            return False
        down_label_1, down_label_2 = down_labels[:2]
    else:
        return False

    return True

img = load_image("img/rolfsen_all/4_3.png")
windowSize = 30

# visualize(img, 'Original Image')

bi_img = binarize(img)

labels, component_sizes, output_img = segment_image(bi_img)
labels = arrange_labels(labels)
print(f'labels: {labels}')
print(f'component_sizes: {component_sizes}')
visualize_labels_with_random_color_and_text(labels)

# save_image(output_img, 'output/3_1_segment.png')

separate_labels = separate_labels(labels)
# show_each_components(separate_labels)

classification = classify_segments(separate_labels)
unifyLabels = unify_non_line_segments(labels, classification)

###
unifyLabels = extract_centerlines_for_labels(unifyLabels)
labels = remove_border_pixels(unifyLabels)
# 示例使用

# 提取所有标签的中心线
centerlines = extract_centerlines_for_labels(labels)

# 移除边界像素
centerlines = remove_border_pixels(centerlines)
visualize_labels_with_random_color_and_text(centerlines)
# 获得端点

endpoints = get_endpoints(centerlines)

# 获得交叉点
# 如果