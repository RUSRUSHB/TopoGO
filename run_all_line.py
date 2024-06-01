from source.binarize import binarize
from source.connect_component_detect import segment_image
from source.io import load_image, save_image
from source.is_line import classify_segments
from source.util import *
from source.alex import alex_polynomial, straighten
from main import *

import time
# read all the image in the folder "img/rolfsen_all" then do the knot detection then store the alex and knot detection result in the folder "output"
import os
import re
# do the error handling for the case when error occurs
# 按数字顺序读文件
# 获取文件夹中所有文件的文件名并按字母顺序排序
file_list = sorted(os.listdir("img/extended_rolfsen_all_no_wrong"))

def sort_key(filename):
    # 提取文件名中的数字部分用于排序
    return list(map(int, re.findall(r'\d+', filename)))

# def fast_action(filename):
#     #只进行操作，尽可能快速
#     labels, _, _ = arrange_labels(segment_image(binarize(load_image("img/extended_rolfsen_all_no_wrong/" + filename))))
#     straightened_data = straighten(np.array(map_labels_to_consecutive_integers(examine_crossings(extract_centerlines_for_labels(unify_non_line_segments(labels, classify_segments(separate_labels(arrange_labels(segment_image(binarize(load_image("img/extended_rolfsen_all_no_wrong/" + filename))))[0]))))[1:-1, 1:-1]))[0]))
#     print(f'{filename} is: {alex_polynomial(straightened_data)}')
#     success += 1
#     pass
def fast_actions(filename):
    labels = arrange_labels(segment_image(binarize(load_image("img/extended_rolfsen_all_no_wrong/" + filename)))[0])
    # print(f'{filename} is: {alex_polynomial(straighten(np.array(map_labels_to_consecutive_integers(examine_crossings(extract_centerlines_for_labels(unify_non_line_segments(labels, classify_segments(separate_labels(labels))))[1:-1, 1:-1]))[0])))}')

    alex_polynomial(straighten(np.array(map_labels_to_consecutive_integers(examine_crossings(extract_centerlines_for_labels(unify_non_line_segments(labels, classify_segments(separate_labels(labels))))[1:-1, 1:-1]))[0])))
    pass

file_list = sorted(os.listdir("img/extended_rolfsen_all_no_wrong"), key=sort_key)
success = 0
start_time = time.time()

# 处理文件夹中的所有图像
for filename in file_list:
    
    try:
        # fast_actions(filename)
        img = load_image("img/extended_rolfsen_all_no_wrong/" + filename)
        
        bi_img = binarize(img)
        labels = arrange_labels(segment_image(bi_img)[0])

        separated_labels = separate_labels(labels)
        classification = classify_segments(separated_labels)
        unifyLabels = unify_non_line_segments(labels, classification)

        centerline_labels = extract_centerlines_for_labels(unifyLabels)
        centerline_labels = centerline_labels[1:-1, 1:-1]

        crossings = examine_crossings(centerline_labels)

        mapped_crossings = map_labels_to_consecutive_integers(crossings)[0]

        straightened_data = straighten(np.array(mapped_crossings))
        print(f'{filename} is: {alex_polynomial(straightened_data)[0]}')
        success += 1
    
    except Exception as e:
        print(f"Error processing {filename}")
        # visualize_labels_with_random_color_and_text(labels)
        # visualize_labels_with_random_color_and_text(centerline_labels)
        # run_with_all_visualization(img)


print(f"Successfully processed {success} images.")
print(f"Total time: {time.time() - start_time} seconds.")