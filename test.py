import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
#
in_img = cv2.imread("input/two_squares.png")

# 转成灰度图
gray_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)

# 二值化
_, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

# 黑色为背景，白色为边界线
binary_img = cv2.bitwise_not(thresh_img)

# 显示图片

# plt.figure(figsize=(10, 5))
#
# plt.subplot(121)
# plt.imshow(cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
#
# plt.subplot(122)
# plt.imshow(binary_img, cmap='gray')
# plt.title('Binary Image')
#
# plt.show()

# 联通区域标记
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=4, ltype=cv2.CV_32S)
# 4联通模式。返回连通区域数目，标记图，状态信息，中心点信息

print('Number of labels:', num_labels)
print('Labels:', labels)
print('Stats:', stats)
print('Centroids:', centroids)
