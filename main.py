from source.binarize import binarize
from source.visualize import visualize
from source.connect_component_detect import segment_image
from source.io import load_image, save_image

# read gif image
img = load_image("img/rolfsen/3_1.png")
# visualize(img, 'Original Image')

bi_img = binarize(img)
# visualize(bi_img, 'Binarized Image')

labels, component_sizes, output_img=segment_image(bi_img)
# visualize(output_img, 'Segmented Image', 'colorful')
save_image(output_img, 'output/3_1_segment.png')

print(f'labels: {labels}')
print(f'component_sizes: {component_sizes}')