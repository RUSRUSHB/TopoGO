import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("input/Trefoil.png")

# threshold the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# invert black and white
gray = cv2.bitwise_not(gray)



# show the image
cv2.imshow("gray", gray)
cv2.waitKey(0)


# connected component analysis
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity=4)

print(f'num_labels: {num_labels}')
plt.figure()
plt.imshow(labels)
plt.show()

print(f'max of labels: {np.max(labels)}')

print(f'stats: {stats}')
print(f'centroids: {centroids}')


