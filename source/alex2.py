import numpy as np
import sympy as sp

'''
window_scan返回的数组维度为(n, 4):
c_i=[crossing, upline, downlines[0], downlines[1]]
每个crossing有三条线，两条在下面，一条在上面
相对应地，每一条线都对应着两个crossing，两次都在下面
任务1：理顺downlines，使得每一个line只会成为一次downlines[0]、一次downlines[1]

'''


def create_snake_array(n):
    # 第一部分：从 [0, 0] 到 [n-1, 0]
    part1 = np.column_stack((np.arange(n), np.zeros(n, dtype=int)))

    # 第二部分：从 [n-1, 1] 到 [n-1, n-1]
    part2 = np.column_stack((np.full(n - 1, n - 1), np.arange(1, n)))

    # 第三部分：从 [n-1, n-1] 到 [n-1, 0]
    part3 = np.column_stack((np.full(n - 1, n - 1), np.arange(n - 1, 0, -1)))

    # 第四部分：从 [n-1, 0] 到 [1, 0]
    part4 = np.column_stack((np.arange(n - 1, 0, -1), np.zeros(n - 1, dtype=int)))

    # 合并所有部分
    snake_array = np.vstack((part1, part2, part3, part4))

    return snake_array


def extract_boundary_counterclockwise(matrix):
    # 确保输入是一个二维数组
    if len(matrix.shape) != 2:
        raise ValueError("Input must be a 2D array")

    rows, cols = matrix.shape
    if rows == 1:
        return matrix[0, :].tolist()
    if cols == 1:
        return matrix[:, 0].tolist()

    # 提取边界元素
    boundary_elements = []

    # 上边界（从左到右）
    boundary_elements.extend(matrix[0, :])

    # 右边界（从上到下，不包括第一个元素）
    boundary_elements.extend(matrix[1:, -1])

    # 下边界（从右到左，不包括第一个和最后一个元素）
    if rows > 1:
        boundary_elements.extend(matrix[-1, -2::-1])

    # 左边界（从下到上，不包括第一个和最后一个元素）
    if cols > 1:
        boundary_elements.extend(matrix[-2:0:-1, 0])

    return boundary_elements


def window_area_reassign(windows):
    windows_size = windows.shape[0]
    snake_array = create_snake_array(windows_size)

    around_loc = np


def straighten(raw_data: np.array) -> np.array:
    # 提取第一列
    raw_uplines = raw_data[:, 0]

    # 提取除第一列外的所有列
    raw_downlines = raw_data[:, 1:]

    print("Uplines:\n", raw_uplines)
    print("Downlines:\n", raw_downlines)

    # 1. 从第一个crossing开始
    crossing_list = [0]  # 从原始数据的第一个crossing开始
    start_line_list = [raw_downlines[0, 0]]  # 从第一个crossing的第一个downline开始
    crossing_num = len(raw_data)
    target = raw_downlines[0, 0]

    for i in range(crossing_num):
        for j in range(crossing_num):
            if j != i and target in raw_downlines[j] and j not in crossing_list:
                # 如果这个crossing的downlines中有target，且这个crossing没有被访问过

                crossing_list.append(j)
                target = [x for x in raw_downlines[j] if x != target][0]
                # 从这个crossing的downlines中找到不是target（也就是另外一个）的那个downline
                # [0]是因为它返回的是一个列表，而不是一个数字。而这个列表长度必为1
                start_line_list.append(target)

                break

    print("Start Line list:", start_line_list)
    print("Crossing list:", crossing_list)

    straight_data = np.zeros([crossing_num, 4], dtype=int)
    # 按照crossing_list的顺序，将raw_data的数据按照start_line_list的顺序填入
    for i in range(crossing_num):
        straight_data[i, 0] = crossing_list[i]
        straight_data[i, 1] = raw_uplines[crossing_list[i]]
        straight_data[i, 2] = start_line_list[i]
        straight_data[i, 3] = [x for x in raw_downlines[crossing_list[i]] if x != straight_data[i, 2]][0]

    print("Straight data:\nCrossing|Upline|Downline_out|Downline_in\n", straight_data)
    return straight_data


def alex_polynomial(straight_data: np.array) -> np.array:
    pass


# 样例：
if __name__ == "__main__":
    # t = sp.symbols('t')
    # matrix = sp.Matrix([[1-t,0,0,0,-1,t],[-1,t,0,0,1-t,0],[0,1-t,0,-1,t,0],[0,-1,t,1-t,0,0],[-1,0,1-t,0,0,t],[0,0,t,-1,0,1-t]])
    # matrix.row_del(0)
    # matrix.col_del(0)
    # print(matrix)
    # print(sp.det(matrix))

    # raw_data = np.array([[1,3,2], [3,1,2], [2,3,1]])-1
    # 3_1 trefoil
    # ANS: 1-t+t^2

    # raw_data = np.array([[1,2,0], [0,2,5],[3,4,1],[2,1,3],[4,5,3],[5,4,0]])
    # 6_1
    # ANS: 2-5*t+2*t^2

    # raw_data = np.array([[3,6,1],[6,3,4],[1,6,5],[5,1,2],[2,5,4],[4,2,3]])-1
    # # 知乎中的例子
    # # ANS: -2t^3+5t^2-2t

    # straightened_data = straighten(raw_data)
    # alex_polynomial(straightened_data)
    pass
