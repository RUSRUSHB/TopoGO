import numpy as np
import matplotlib.pyplot as plt

def flood_fill(image, seed, target_color, windows_loc):
    height, width = image.shape
    target_value = image[seed[0], seed[1]]
    if target_value == target_color:
        return image
    
    queue = [seed]
    while queue:

        # TODO: stop if run into a window

        y, x = queue.pop(0)
        image[y, x] = target_color
        
        neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
        for ny, nx in neighbors:
            if 0 <= ny < height and 0 <= nx < width and image[ny, nx] == target_value:
                queue.append((ny, nx))
        # plt.imshow(image)
        # plt.show()
        # plt.pause(0.001)

    return image

# image = np.random.randint(2, size=(50,50))
image = np.zeros([100,100])

# 选择种子像素和目标颜色
seed = (0, 0)
target_color = 3

flood_fill(image, seed, target_color)