import cv2

def binarize(img_path):

    parameters = [127]
    # 参数列表：[阈值]

    img = cv2.imread(img_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, parameters[0], 255, cv2.THRESH_BINARY)
    
    gray = cv2.bitwise_not(gray)# 黑底白线

    return gray

binarize("input/Trefoil.png")