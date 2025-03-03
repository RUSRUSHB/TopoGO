import matplotlib.pyplot as plt
import numpy as np


def visualize(img, title, *args):
    # 必选参数：[]
    # 可变参数列表：[彩色]
    plt.figure()

    if len(args) == 0:
        plt.imshow(img, cmap='gray')
    elif len(args) == 1:
        if args[0] == 'colorful':
            plt.imshow(img)
            plt.title(args[0])
    else:
        pass
    plt.title(title)
    plt.show()


def show_each_components(labels):
    # 分别展现每一个组件
    for i in range(labels.max() + 1):
        component = np.zeros_like(labels)
        component[labels == i] = 255
        visualize(component, f'Component {i}')
    # 注意到第一个组件是轮廓线，最后一个组件是背景
    pass
