import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_image(img, title):
    # plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    # plt.axis('off')
    # plt.show()
    return
