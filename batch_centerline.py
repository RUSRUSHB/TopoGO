"""管线 B 批量运行器: 使用中心线法处理所有扭结图片。

遍历目标文件夹中的所有图像文件，并依次使用中心线（骨架）的方法计算获取亚历山大多项式。
"""

import os
import re
import time
import numpy as np
import importlib

# 动态导入模块以规避数字开头的文件名引用限制
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

# --- 参数配置项 ---
# 使用图片输入文件夹的路径
INPUT_DIR = "img/extended_rolfsen_all_no_wrong"

def sort_key(filename: str) -> list[int]:
    """提取文件名中的数字编号，以便对文件进行准确排序。"""
    return list(map(int, re.findall(r"\d+", filename)))

def main():
    file_list = sorted(os.listdir(INPUT_DIR), key=sort_key)
    start_time = time.time()
    success = 0

    # 循环遍历目录执行管线批跑过程
    for filename in file_list:
        try:
            img = load_image(os.path.join(INPUT_DIR, filename))
            bi_img = binarize(img)
            labels = arrange_labels(segment_image(bi_img)[0])

            separated = separate_labels(labels)
            classification = classify_segments(separated)
            unified_labels = unify_non_line_segments(labels, classification)

            centerline_labels = extract_centerlines_for_labels(unified_labels)
            centerline_labels = centerline_labels[1:-1, 1:-1]

            crossings = examine_crossings(centerline_labels)
            mapped_crossings = map_labels_to_consecutive_integers(crossings)[0]

            straightened_data = straighten(np.array(mapped_crossings))
            result, _ = alex_polynomial(straightened_data)
            print(f"{filename} 的多项式结果是: {result}")
            success += 1

        except Exception as e:
            print(f"处理 {filename} 时捕获到异常信息: {e}")

    print(f"\n顺利处理了 {success} 张图像。")
    print(f"总计运算耗时: {time.time() - start_time:.1f} 秒。")

if __name__ == "__main__":
    main()
