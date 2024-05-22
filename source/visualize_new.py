import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random


def visualize(img, title, *args):
    # 必选参数：[]
    # 可变参数列表：[彩色]
    plt.figure()

    if len(args) == 0:
        plt.imshow(img, cmap='gray')
    elif len(args) == 1:
        if args[0] == 'colorful':
            plt.imshow(img)
            plt.title(args[0])
    else:
        pass
    plt.title(title)
    plt.show()


def show_each_components(separated_layers):
    """
    将输入数组中的每个唯一标签显示为单独的图像。
    """

    # 遍历并显示每个图层
    num_labels, height, width = separated_layers.shape
    for i in range(num_labels):
        layer = separated_layers[i]
        plt.imshow(layer, cmap='gray')
        plt.title(f'Component {i}')
        plt.show()


def is_surrounded_by_same_label(labels, x, y, label):
    # 检查 (x, y) 周围的像素是否都是同样的标签
    height, width = labels.shape
    for dx in range(-6, 6):
        for dy in range(-6, 6):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if labels[ny, nx] != label:
                    return False
    return True


def visualize_labels_with_random_color_and_text(labels):
    # 获取唯一的标签
    unique_labels = np.unique(labels)

    # 创建一个颜色字典，将每个标签映射到一个随机颜色
    color_dict = {label: np.random.rand(3,) for label in unique_labels}

    # 创建一个RGB图像，初始化为全白色
    height, width = labels.shape
    colored_image = np.ones((height, width, 3))

    # 为每个标签分配颜色
    for label in unique_labels:
        mask = labels == label
        colored_image[mask] = color_dict[label]

    # 使用plt显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(colored_image)

    # 在图像上标记标签
    for label in unique_labels:
        mask = labels == label
        if np.any(mask):
            # 找到标签区域内的任意一个像素位置
            yx_pairs = np.argwhere(mask)
            for (y, x) in yx_pairs:
                if is_surrounded_by_same_label(labels, x, y, label):
                    # 在该位置放置标签文本
                    plt.text(x, y, str(label), color='white',
                             fontsize=12, ha='center', va='center')
                    break

    # 创建图例
    patches = [mpatches.Patch(
        color=color_dict[label], label=f'Label {label}') for label in unique_labels]
    plt.legend(handles=patches, bbox_to_anchor=(
        1.05, 1), loc='upper left', borderaxespad=0.)

    plt.axis('off')  # 关闭坐标轴显示
    plt.show()


def visualize_labels_with_random_color_and_text_and_crossingWindow(labels, crossings, window_size=5):
    # 获取标签的唯一值
    unique_labels = np.unique(labels)
    height, width = labels.shape
    color_map = {}

    # 随机生成每个标签的颜色
    for label in unique_labels:
        if label == unique_labels[-1]:
            color_map[label] = (0, 0, 0)  # 背景颜色为黑色
        else:
            color_map[label] = (random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255))

    # 创建彩色图像
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 根据标签着色
    for label in unique_labels:
        color_image[labels == label] = color_map[label]

    # 在图像上标注标签文本
    for label in unique_labels:
        # 获取标签的位置
        positions = np.argwhere(labels == label)
        if len(positions) > 0:
            # 随机选择一个位置进行标注
            y, x = positions[random.randint(0, len(positions) - 1)]
            # 判断周围是否都是同样的颜色
            if np.all(labels[max(y - 1, 0):min(y + 2, height), max(x - 1, 0):min(x + 2, width)] == label):
                cv2.putText(color_image, str(label), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # 在图像上绘制交叉窗口
    half_window = window_size // 2
    for crossing in crossings:
        x, y, _, _ = crossing
        top_left = (x - half_window, y - half_window)
        bottom_right = (x + half_window, y + half_window)
        cv2.rectangle(color_image, top_left, bottom_right, (0, 255, 0), 2)

    # 显示图像
    plt.imshow(color_image)
    plt.title('Labels with Random Colors, Text, and Crossing Windows')
    plt.show()


if __name__ == '__main__':
    labels = np.array([[0, 0, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0],
                       [2, 2, 0, 0, 0, 3],
                       [2, 2, 0, 0, 0, 3],
                       [2, 2, 0, 0, 0, 3],
                       [0, 0, 3, 3, 3, 0]])
    visualize_labels_with_random_color_and_text(labels)
