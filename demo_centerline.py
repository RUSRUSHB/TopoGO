"""管线 B 演示: 基于中心线法的扭结检测 (单张图片)。

通过提取中心线(骨架)的方法逐步处理单张扭结图片，并展示用于分析的中间结果。
"""

import numpy as np
import importlib

# 动态导入模块，因为文件名是以数字开头的
io = importlib.import_module("src.0_io")
load_image = io.load_image

binmod = importlib.import_module("src.1_binarize")
binarize = binmod.binarize

seg = importlib.import_module("src.2_segment")
segment_image = seg.segment_image

label_utils = importlib.import_module("src.3_label_utils")
arrange_labels = label_utils.arrange_labels
separate_labels = label_utils.separate_labels
unify_non_line_segments = label_utils.unify_non_line_segments

classify = importlib.import_module("src.4_classify")
classify_segments = classify.classify_segments

center = importlib.import_module("src.5b_centerline")
extract_centerlines_for_labels = center.extract_centerlines_for_labels
examine_crossings = center.examine_crossings
map_labels_to_consecutive_integers = center.map_labels_to_consecutive_integers

alex = importlib.import_module("src.6_alexander")
alex_polynomial = alex.alex_polynomial
straighten = alex.straighten

viz = importlib.import_module("src.viz")
visualize_labels_with_random_color_and_text = viz.visualize_labels_with_random_color_and_text

def run_with_all_visualization(img: np.ndarray) -> None:
    """运行完整的管线 B，并在每个步骤中显示可视化效果。

    Args:
        img: 输入的灰度图像。
    """
    import matplotlib.pyplot as plt

    # 步骤 1: 二值化处理
    bi_img = binarize(img)

    # 步骤 2: 将图像分割为连通组件
    labels, component_sizes, output_img = segment_image(bi_img)
    labels = arrange_labels(labels)
    visualize_labels_with_random_color_and_text(labels)

    # 步骤 3-4: 提取分类与重组组件
    separated = separate_labels(labels)
    classification = classify_segments(separated)
    unified_labels = unify_non_line_segments(labels, classification)

    # 步骤 5b: 提取各个线段组件的中心线
    centerline_labels = extract_centerlines_for_labels(unified_labels)
    centerline_labels = centerline_labels[1:-1, 1:-1]
    visualize_labels_with_random_color_and_text(centerline_labels)

    # 步骤 5b: 进行交叉点检测
    crossings = examine_crossings(centerline_labels, is_visualize=True)
    mapped_crossings, label_mapping = map_labels_to_consecutive_integers(crossings)

    print(f"mapped_crossings: {mapped_crossings}")
    print(f"number of crossings: {len(mapped_crossings)}")
    print(np.array(mapped_crossings))

    # 显示各个处理阶段的可视化结果
    plt.imshow(bi_img, cmap="gray")
    plt.title("Binarized Image")
    plt.axis("off")
    plt.show()

    visualize_labels_with_random_color_and_text(labels)

    plt.imshow(output_img)
    plt.title("Segmented Image with Connected Components")
    plt.axis("off")
    plt.show()

    visualize_labels_with_random_color_and_text(centerline_labels)

if __name__ == "__main__":
    # --- 配置项 ---
    # 图片路径可以根据需要更改
    IMAGE_PATH = "img/rolfsen_all/9_32.png"

    img = load_image(IMAGE_PATH)

    # 步骤 1: 二值化处理
    bi_img = binarize(img)

    # 步骤 2: 将图像分割为连通组件
    labels, component_sizes, output_img = segment_image(bi_img)
    labels = arrange_labels(labels)
    print(f"labels: {labels}")
    print(f"component_sizes: {component_sizes}")
    visualize_labels_with_random_color_and_text(labels)

    # 步骤 3-4: 提取分类以及合并标签
    separated = separate_labels(labels)
    classification = classify_segments(separated)
    unified_labels = unify_non_line_segments(labels, classification)

    # 步骤 5b: 获取组件中心线
    centerline_labels = extract_centerlines_for_labels(unified_labels)
    centerline_labels = centerline_labels[1:-1, 1:-1]
    visualize_labels_with_random_color_and_text(centerline_labels)

    # 步骤 5b: 核查并检查交叉点位置
    crossings = examine_crossings(centerline_labels, is_visualize=True)
    print(f"crossings: {crossings}")

    mapped_crossings, label_mapping = map_labels_to_consecutive_integers(crossings)
    print(f"mapped_crossings: {mapped_crossings}")
    print(f"number of crossings: {len(mapped_crossings)}")

    # 步骤 6: 计算相应的亚历山大多项式
    straightened_data = straighten(np.array(mapped_crossings))
    result, _ = alex_polynomial(straightened_data)
    print(f"亚历山大多项式是: {result}")
