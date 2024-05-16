import numpy as np

def window_scan(labels):
    parameters = [50]  # 窗口大小
    height, width = labels.shape
    window_size = parameters[0]  # 窗口大小

    # 遍历整个图像
    for y in range(0, height-window_size+1):
        for x in range(0, width-window_size+1):
            # 获取当前窗口的子图像
            window = labels[y:y+window_size, x:x+window_size]
            
            # 在这里进行对窗口内子图像的处理
            # 例如，可以计算窗口内像素的平均值、标准差等统计信息，或者应用其他的图像处理操作
            
            # 示例：计算窗口内像素的平均值
            window_mean = np.mean(window)
            
            # 打印结果（这里只是示例）
            print(f"窗口 ({y},{x}) 内的标签平均值: {window_mean}")

# 示例使用
labels = np.random.randint(0, 11, size=(100, 100))  # 随机生成标签图像
# labels = np.ones((100, 100))  # 全部为1的标签图像
window_scan(labels)
