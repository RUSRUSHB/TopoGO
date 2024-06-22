import numpy as np

def zhang_suen_thinning(image):
    # Zhang-Suen细化算法的迭代过程
    def thinning_iteration(image, iter):
        marker = np.zeros_like(image)
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                p2, p3, p4, p5, p6, p7, p8, p9 = n = neighbours(image, i, j)
                if (image[i, j] == 1 and                  # 中心像素为1
                    2 <= sum(n) <= 6 and                   # 中心像素的8邻域有2-6个非零像素
                    transitions(n) == 1 and               # 中心像素的8邻域的非零像素到0像素的过渡次数为1
                    p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0):  # p2, p4, p6, p8不全为非零像素
                    marker[i, j] = 1
        image = np.logical_and(image, np.logical_not(marker))
        
        marker = np.zeros_like(image)
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                p2, p3, p4, p5, p6, p7, p8, p9 = n = neighbours(image, i, j)
                if (image[i, j] == 1 and
                    2 <= sum(n) <= 6 and
                    transitions(n) == 1 and
                    p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0):
                    marker[i, j] = 1
        image = np.logical_and(image, np.logical_not(marker))
        return image

    # 计算像素邻域
    def neighbours(image, x, y):
        return [image[x-1, y], image[x-1, y+1], image[x, y+1], image[x+1, y+1], 
                image[x+1, y], image[x+1, y-1], image[x, y-1], image[x-1, y-1]]

    # 计算从0到1的过渡次数
    def transitions(neighbours):
        n = neighbours + neighbours[0:1]
        return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

    # 进行多次迭代直到不再有变化
    prev_image = np.zeros_like(image)
    while True:
        curr_image = thinning_iteration(image.copy(), 0)
        if np.array_equal(curr_image, prev_image):
            break
        prev_image = curr_image
    return curr_image.astype(np.uint8)
