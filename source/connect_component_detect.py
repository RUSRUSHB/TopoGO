import numpy as np
import cv2
import matplotlib.pyplot as plt

def connected_components(img):
    height, width = img.shape
    labels = np.zeros((height, width), dtype=int)
    current_label = 1

    # 查找根节点函数
    def find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 路径压缩
            x = parent[x]
        return x

    # 合并集合函数
    def union(parent, rank, x, y):
        root_x = find(parent, x)
        root_y = find(parent, y)
        if root_x != root_y:
            if rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            elif rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            else:
                parent[root_y] = root_x
                rank[root_x] += 1

    parent = {}
    rank = {}

    # 遍历图像中的每个像素
    for y in range(height):
        for x in range(width):
            if img[y, x] == 255:  # 只处理白色像素
                neighbors = []
                if x > 0 and labels[y, x - 1] > 0:  # 左邻居
                    neighbors.append(labels[y, x - 1])
                if y > 0 and labels[y - 1, x] > 0:  # 上邻居
                    neighbors.append(labels[y - 1, x])

                if not neighbors:  # 如果没有已标记的邻居
                    labels[y, x] = current_label
                    parent[current_label] = current_label
                    rank[current_label] = 0
                    current_label += 1
                else:  # 如果有已标记的邻居
                    smallest_label = min(neighbors)
                    labels[y, x] = smallest_label
                    for neighbor in neighbors:
                        union(parent, rank, smallest_label, neighbor)

    # 第二次遍历，将所有像素的标签更新为根标签
    for y in range(height):
        for x in range(width):
            if labels[y, x] > 0:
                labels[y, x] = find(parent, labels[y, x])

    # 计算每个连通组件的大小
    component_sizes = {}
    for label in np.unique(labels):
        if label == 0:
            continue
        component_sizes[label] = np.sum(labels == label)

    # 重新排序标签
    new_labels = np.zeros_like(labels)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(np.unique(labels))}
    for old_label, new_label in label_mapping.items():
        new_labels[labels == old_label] = new_label
    
    # 更新component_sizes的标签
    new_component_sizes = {label_mapping[old_label]: size for old_label, size in component_sizes.items()}
    # 现在这些标签是从1开始的了
    return new_labels, new_component_sizes

def visualize_components(img, labels):
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    colors = np.random.randint(0, 255, size=(num_labels, 3))
    colors[0] = [0, 0, 0]  # 保持背景为黑色

    output_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i, label in enumerate(unique_labels):
        if label == 0:
            continue
        output_img[labels == label] = colors[i]

    return output_img

def segment_image(binary_img):
    labels, component_sizes = connected_components(binary_img)
    output_img = visualize_components(binary_img, labels)
    return labels, component_sizes, output_img

# 示例使用
if __name__ == '__main__':
    # 读取图像并二值化
    img = cv2.imread('input/Trefoil.png', cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 处理图像
    labels, component_sizes, output_img = segment_image(binary_img)
    # 储存图像output_img
    cv2.imwrite('output/connected_components.png', output_img)
    # 显示结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(binary_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output_img)
    plt.title('Connected Components')
    plt.axis('off')

    plt.show()
