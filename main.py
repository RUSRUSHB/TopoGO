from source.binarize import binarize
from source.visualize import visualize
from source.connect_component_detect import segment_image
from source.io import load_image, save_image
from source.is_line import is_line_segment, classify_segments
from source.visualize_new import *
from source.util import *
from source.cross_detect import *


# read gif image
img = load_image("img/k3.png")  # 内径40，外径60
# visualize(img, 'Original Image')

bi_img = binarize(img)

labels, component_sizes, output_img = segment_image(bi_img)
labels = arrange_labels(labels)
print(f'labels: {labels}')
print(f'component_sizes: {component_sizes}')
# visualize_labels_with_random_color_and_text(labels)

# save_image(output_img, 'output/3_1_segment.png')

separate_labels = separate_labels(labels)
# show_each_components(separate_labels)

classification = classify_segments(separate_labels)
unifyLabels = unify_non_line_segments(labels, classification)
# visualize_labels_with_random_color_and_text(unifyLabels)
visualize_labels_with_random_color_and_text_and_crossingWindow(
    unifyLabels, process_image(unifyLabels, 70), 70)

print(f'classification: {classification}')
print(f"the crossings are {process_image(unifyLabels,70)}")
