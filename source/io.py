import cv2
import numpy as np

def load_image(image_path):
    # 读取图像为灰度图
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def save_image(image, image_path):
    # 保存图像
    cv2.imwrite(image_path, image)
    return