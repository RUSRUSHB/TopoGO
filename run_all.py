from source.binarize import binarize
from source.visualize import visualize
from source.connect_component_detect import segment_image
from source.io import load_image, save_image
from source.is_line import is_line_segment, classify_segments
from source.visualize_new import *
from source.util import *
from source.cross_detect import *
from source.alex import *
import time
# read all the image in the folder "img/rolfsen_all" then do the knot detection then store the alex and knot detection result in the folder "output"
import os
import re
# do the error handling for the case when error occurs
# 按数字顺序读文件
# 获取文件夹中所有文件的文件名并按字母顺序排序
file_list = sorted(os.listdir("img/rolfsen_all_no_wrong"))

start_time = time.time()
def sort_key(filename):
    # 提取文件名中的数字部分用于排序
    return list(map(int, re.findall(r'\d+', filename)))


file_list = sorted(os.listdir("img/rolfsen_all_no_wrong"), key=sort_key)
success = 0
# 处理文件夹中的所有图像
windowSizes = [27, 30, 24, 20, 16] # 尝试的windowSize备选值
# windowSizes = [27]
for filename in file_list:
    windowSize = windowSizes[0]  # 初始值为27
    attempts = 1  # 尝试次数

    while True:  # 循环直到不再遇到异常或者没有更多备选的windowSize
        try:
            img = load_image("img/rolfsen_all_no_wrong/" + filename)
            # visualize(img, 'Original Image')

            bi_img = binarize(img)

            labels, component_sizes, output_img = segment_image(bi_img)
            labels = arrange_labels(labels)
            # visualize_labels_with_random_color_and_text(labels)

            # separate_labels =
            # print(f"start separate_labels for {filename}")

            # classification =
            # print(f"start classify_segments for {filename}")
            # unifyLabels =
            # print(f"start unify_non_line_segments for {filename}")

            # crossings =
            # # print(f"start process_image for {filename}")
            # # visualize_labels_with_random_color_and_text_and_crossingWindow(
            # #     unifyLabels, crossings, windowSize)

            # crossings_arr =
            # print(f"start extractLabelsFromCross_detect for {filename}")
            # unifyCrossings =
            # print(f"start unifyCrossings for {filename}")
            # straightened_data =
            # print(f"start straighten for {filename}")
            result = alex_polynomial(straighten(unifyCrossings(extractLabelsFromCross_detect(process_image(unify_non_line_segments(
                labels, classify_segments(separate_labels(labels))), windowSize)))))
            if attempts > 1:
                print(
                    f"{filename} is {result}. ({attempts} attempts)")
            else:
                print(f"{filename} is {result})")
            attempts = 1
            success += 1
            # print(
            #     f"the knot detection result for {filename} has been saved in the folder output")
            # print(
            #     f"the alex polynomial result for {filename} has been saved in the folder output")
            break  # 成功处理该文件，退出循环
        except Exception as e:
            # print(f"Error processing {filename} with windowSize {windowSize}: {e}")
            # 如果遇到异常，尝试下一个windowSize，如果已经没有备选值了，则退出循环
            if windowSize != windowSizes[-1]:
                attempts+=1
                windowSize = windowSizes[windowSizes.index(windowSize) + 1]
            else:
                print(f'{filename}: All attempts failed')
                # save this picture to the error folder
                # save_image(bi_img, f'output/error/advance_cross_4/{filename}')
                break

print(f"Successfully processed {success} images.")
print(f"Total time: {time.time() - start_time} seconds.")