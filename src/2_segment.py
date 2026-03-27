"""步骤 2: 连通组件划分/检测算法。

基于双遍历（Two-pass）以及并查集原理（Union-Find），在二值化图里提取出各个连通的前景组件标签。
"""

import numpy as np
import cv2


def connected_components(img: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """在当前已二值化的图像里找到连通区域，为其贴上组件标签。

    使用了结合【路径压缩】及【按秩合并】优化的并查集查找两遍算法(Two-pass algorithm)。
    请注意：仅有像素值为白色（数值 255）的点才被认定属于有效前景计算之内。

    Args:
        img: 黑白二值化图像形参 (0 代表背景背景, 255 意为前景像素)。

    Returns:
        包含以下元素的元组形式 (labels连通标签数组, component_sizes各元素体积字典)。
        其中 labels 是从数值 '1' 起头向后依次分配整数作为独立标签的矩阵。
    """
    height, width = img.shape
    labels = np.zeros((height, width), dtype=int)
    current_label = 1

    def find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 触发查找树状结构的路径快速收缩
            x = parent[x]
        return x

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

    # 第一轮逐行遍历操作: 先初步分配局部区域的临时子标签编号、并将连接规则更新入并查集内部
    for y in range(height):
        for x in range(width):
            if img[y, x] == 255:
                neighbors = []
                if x > 0 and labels[y, x - 1] > 0:
                    neighbors.append(labels[y, x - 1])
                if y > 0 and labels[y - 1, x] > 0:
                    neighbors.append(labels[y - 1, x])

                if not neighbors:
                    labels[y, x] = current_label
                    parent[current_label] = current_label
                    rank[current_label] = 0
                    current_label += 1
                else:
                    smallest_label = min(neighbors)
                    labels[y, x] = smallest_label
                    for neighbor in neighbors:
                        union(parent, rank, smallest_label, neighbor)

    # 第二次全面遍历: 正式解析连通树，使得所有的连通组块对应到最初那个最高根标签上
    for y in range(height):
        for x in range(width):
            if labels[y, x] > 0:
                labels[y, x] = find(parent, labels[y, x])

    # 根据最新的标签情况来计算属于同一组件范围包含的总块个数（尺寸）
    component_sizes = {}
    for label in np.unique(labels):
        if label == 0:
            continue
        component_sizes[label] = np.sum(labels == label)

    # 最后再统一映射重排标签数列，使其保证像 1,2,3,4... 等完全连续的不中断序列形式
    new_labels = np.zeros_like(labels)
    label_mapping = {old: new for new, old in enumerate(np.unique(labels))}
    for old_label, new_label in label_mapping.items():
        new_labels[labels == old_label] = new_label

    new_component_sizes = {
        label_mapping[old]: size for old, size in component_sizes.items()
    }

    return new_labels, new_component_sizes


def visualize_components(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """给那些标定好连通区域的不同组件上涂色彩，用作展示可视化效果使用。

    Args:
        img: 处理初的高宽灰阶图底板 (在这里被拿来对比获取形状规格即可)。
        labels: 经 connected_components() 解析产出的各像素连通结果身份码数组。

    Returns:
        随机染上互斥差异显著彩色的最终 RGB 图像数组矩阵。
    """
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    colors = np.random.randint(0, 255, size=(num_labels, 3))
    colors[0] = [0, 0, 0]  # 让处于大片空缺背景下的色块始终维持黑色不变

    output_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i, label in enumerate(unique_labels):
        if label == 0:
            continue
        output_img[labels == label] = colors[i]

    return output_img


def segment_image(binary_img: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray]:
    """对二值图像本身展开划分并分割为相互带记号连通的分块组件集合。

    Args:
        binary_img: 只有0即255的源头二值图文件。

    Returns:
        返回具有三属性的一个集合元组结构，由 (连通区域数字记号群组, 对应其物理体积记录词典, 带颜色的可展示版本图片) 拼接而成。
    """
    labels, component_sizes = connected_components(binary_img)
    output_img = visualize_components(binary_img, labels)
    return labels, component_sizes, output_img
