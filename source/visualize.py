import matplotlib.pyplot as plt

def visualize(img, title, *args):
    # 必选参数：[]
    # 可变参数列表：[彩色]
    plt.figure()

    match len(args):
        case 0:
            plt.imshow(img, cmap='gray')

        case 1:
            match args[0]:
                case 'colorful':
                    plt.imshow(img)
                    plt.title(args[0])
                case _:
                    pass
        case _:
            pass
    plt.title(title)
    plt.show()
