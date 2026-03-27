"""管线 A 批量运行器: 使用滑动窗口法处理所有扭结图片。

遍历目标文件夹中的所有图像文件，并依次计算其对应的亚历山大多项式。
若某一个设定的滑动窗口大小未成功跑通，会尝试备选窗口尺寸重试。
"""

import os
import re
import time
import importlib

# 动态导入模块
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

# --- 参数配置项 ---
# 使用图片输入文件夹的路径
INPUT_DIR = "img/rolfsen_all_no_wrong"
# 作为备选的尝试滑动窗口尺寸
WINDOW_SIZES = [27, 30, 24, 20, 16]

def sort_key(filename: str) -> list[int]:
    """提取文件名中的数字编号，以便用于对文件进行准确排序。"""
    return list(map(int, re.findall(r"\d+", filename)))

def main():
    file_list = sorted(os.listdir(INPUT_DIR), key=sort_key)
    start_time = time.time()
    success = 0

    # 循环遍历整个文件夹内的数据集
    for filename in file_list:
        window_size = WINDOW_SIZES[0]
        attempts = 1

        while True:
            try:
                img = load_image(os.path.join(INPUT_DIR, filename))
                bi_img = binarize(img)

                labels, _, _ = segment_image(bi_img)
                labels = arrange_labels(labels)

                # 将各管线处理步骤整合，一次性计算结果输出
                result, _ = alex_polynomial(
                    straighten(
                        unify_crossings(
                            extract_labels_from_crossings(
                                process_image(
                                    unify_non_line_segments(
                                        labels,
                                        classify_segments(
                                            separate_labels(labels)
                                        ),
                                    ),
                                    window_size,
                                )
                            )
                        )
                    )
                )

                if attempts > 1:
                    print(f"{filename} 的多项式结果是: {result}。 (重试了 {attempts} 次)")
                else:
                    print(f"{filename} 的多项式结果是: {result}")
                success += 1
                break

            except Exception as e:
                # 当报错的时候，尝试改用下一个窗口尺寸值；如果都耗尽仍然失败则彻底跳过
                if window_size != WINDOW_SIZES[-1]:
                    attempts += 1
                    window_size = WINDOW_SIZES[
                        WINDOW_SIZES.index(window_size) + 1
                    ]
                else:
                    print(f"{filename}: 所有备选窗口尺寸都尝试过，最终处理失败。")
                    break

    print(f"\n顺利处理了 {success} 张图像。")
    print(f"总计运算耗时: {time.time() - start_time:.1f} 秒。")

if __name__ == "__main__":
    main()
