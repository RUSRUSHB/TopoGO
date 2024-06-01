from source.binarize import binarize
from source.visualize import visualize
from source.connect_component_detect import segment_image
from source.io import load_image, save_image
from source.is_line import is_line_segment, classify_segments
from source.visualize_new import *
from source.util import *
from source.cross_detect import *
from source.alex import *

# read gif image
img = load_image("img/rolfsen_all/10_163.png")
windowSize = 27

# visualize(img, 'Original Image')

bi_img = binarize(img)

labels, component_sizes, output_img = segment_image(bi_img)
labels = arrange_labels(labels)
print(f'labels: {labels}')
print(f'component_sizes: {component_sizes}')
visualize_labels_with_random_color_and_text(labels)

# save_image(output_img, 'output/3_1_segment.png')

separate_labels = separate_labels(labels)
# show_each_components(separate_labels)

classification = classify_segments(separate_labels)
unifyLabels = unify_non_line_segments(labels, classification)
# visualize_labels_with_random_color_and_text(unifyLabels)
crossings = process_image(unifyLabels, windowSize)
visualize_labels_with_random_color_and_text_and_crossingWindow(
    unifyLabels, crossings, windowSize)

print(f'classification: {classification}')
print(f"the crossings are {crossings}")
crossings_arr = extractLabelsFromCross_detect(crossings)
print(f"crossings_arr: {crossings_arr}")
unifyCrossings = unifyCrossings(crossings_arr)
print(f"crossings_arr: {unifyCrossings}")


straightened_data = straighten(unifyCrossings)
alex_polynomial(straightened_data)