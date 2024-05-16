import cv2

def binarize(img_path):

    parameters = [127]
    # 参数列表：[阈值]

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, parameters[0], 255, cv2.THRESH_BINARY)

    # binary_img = cv2.bitwise_not(binary_img)# 黑底白线

    return binary_img

