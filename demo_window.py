"""管线 A 演示: 基于滑动窗口法的扭结检测 (单张图片)。

通过滑动窗口法逐步处理单张扭结图片，并展示用于分析的中间结果。
"""

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
extract_labels_from_crossings = label_utils.extract_labels_from_crossings
unify_crossings = label_utils.unify_crossings

classify = importlib.import_module("src.4_classify")
classify_segments = classify.classify_segments

cross = importlib.import_module("src.5a_cross_detect")
process_image = cross.process_image

alex = importlib.import_module("src.6_alexander")
alex_polynomial = alex.alex_polynomial
straighten = alex.straighten

viz = importlib.import_module("src.viz")
visualize_labels_with_random_color_and_text = viz.visualize_labels_with_random_color_and_text
visualize_labels_with_crossings = viz.visualize_labels_with_crossings
show_each_components = viz.show_each_components

# --- 配置项 ---
IMAGE_PATH = "img/rolfsen_all/9_32.png"
WINDOW_SIZE = 16

# 步骤 0: 加载图像
img = load_image(IMAGE_PATH)

# 步骤 1: 二值化处理
bi_img = binarize(img)

# 步骤 2: 将图像分割为连通组件
labels, component_sizes, output_img = segment_image(bi_img)

# 步骤 3: 整理标签 (使其成为连续的整数，背景标记为最大值)
labels = arrange_labels(labels)
print(f"labels: {labels}")
print(f"component_sizes: {component_sizes}")
visualize_labels_with_random_color_and_text(labels)

# 步骤 3b: 分离并展示各个组件
separated = separate_labels(labels)
show_each_components(separated)

# 步骤 4: 对组件进行分类，判断其是否为线段
classification = classify_segments(separated)
unified_labels = unify_non_line_segments(labels, classification)
visualize_labels_with_random_color_and_text(unified_labels)

# 步骤 5a: 通过滑动窗口检测交叉点
crossings = process_image(unified_labels, WINDOW_SIZE)
visualize_labels_with_crossings(unified_labels, crossings, WINDOW_SIZE)

print(f"classification: {classification}")
print(f"the crossings are {crossings}")

# 步骤 5a (继续): 提取并统一交叉标签的数据结构
crossings_arr = extract_labels_from_crossings(crossings)
print(f"crossings_arr: {crossings_arr}")

unified_cross = unify_crossings(crossings_arr)
print(f"unified crossings: {unified_cross}")

# 步骤 6: 计算亚历山大多项式(Alexander polynomial)
straightened_data = straighten(unified_cross)
result, _ = alex_polynomial(straightened_data)
print(f"亚历山大多项式是: {result}")
