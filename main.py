from source.binarize import binarize
from source.visualize import visualize
from source.connect_component_detect import segment_image
import matplotlib.pyplot as plt

img = binarize("input/Trefoil.png")
visualize(img, 'Binarized Image')

labels, component_sizes, output_img=segment_image(img)
visualize(output_img, 'Segmented Image', 'colorful')

