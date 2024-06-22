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

img = load_image("img/rolfsen_all/9_32.png")

windowSizes = [27, 30, 24, 20, 16] # 尝试的windowSize备选值
# windowSizes = [27]
windowSize = windowSizes[0]  # 初始值为27
attempts = 1  # 尝试次数

while True:  # 循环直到不再遇到异常或者没有更多备选的windowSize
    try:
        # visualize(img, 'Original Image')

        bi_img = binarize(img)

        labels, component_sizes, output_img = segment_image(bi_img)
        labels = arrange_labels(labels)


        result = alex_polynomial(straighten(unifyCrossings(extractLabelsFromCross_detect(process_image(unify_non_line_segments(
            labels, classify_segments(separate_labels(labels))), windowSize)))))
        
        
        if attempts > 1:
            print(
                f"is {result}. ({attempts} attempts)")
        else:
            print(f"is {result})")
        attempts = 1

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
            print(f'All attempts failed')
            # save this picture to the error folder
            # save_image(bi_img, f'output/error/advance_cross_4/{filename}')
            break
