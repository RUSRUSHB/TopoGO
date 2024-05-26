import numpy as np
import sympy as sp
'''
window_scan返回的数组维度为(n, 3):
c_i=[upline, downlines[0], downlines[1]]
每个crossing有三条线，两条在下面，一条在上面
相对应地，每一条线都对应着两个crossing，两次都在下面
任务1：理顺downlines，使得每一个line只会成为一次downlines[0]、一次downlines[1]

'''

def straighten_fixed(picture: np.array, crossing_data: np.array, windows_loc: np.array, window_size)->np.array:
    colored_picture = np.zeros_like(picture)
    colored_picture[picture == max(picture.flatten())] = 255
    # 背景上色为255

    raw_uplines = crossing_data[:, 0]
    raw_downlines = crossing_data[:, 1:]

    print("Uplines:\n", raw_uplines)
    print("Downlines:\n", raw_downlines)

    crossing_num = len(raw_data)
    windows_list = picture[windows_loc[0]:windows_loc[0]+window_size, windows_loc[1]:windows_loc[1]+window_size]

    start_line_list = [raw_downlines[0, 0]]# 从第一个crossing的第一个downline开始
    target = raw_downlines[0, 0]

    # window中label为target的像素点的坐标，选取第一个点作为seed
    for i in range(crossing_num):
        seed = np.where(windows_list == target)[0][0]
        # 染色算法，向四联通方向扩展
        colored_picture

    pass

def straighten(raw_data: np.array)->np.array:
    
    # 提取第一列
    raw_uplines = raw_data[:, 0]

    # 提取除第一列外的所有列
    raw_downlines = raw_data[:, 1:]

    print("Uplines:\n", raw_uplines)
    print("Downlines:\n", raw_downlines)

    # 1. 从第一个crossing开始
    crossing_list = [0]  # 从原始数据的第一个crossing开始
    start_line_list = [raw_downlines[0, 0]]# 从第一个crossing的第一个downline开始
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

def alex_polynomial(straight_data: np.array)->np.array:
    # upline: 1-t; downline_out: t; downline_in: -1
    
    t = sp.symbols('t')
    # 生成矩阵
    n = len(straight_data)
    uplines = straight_data[:, 1]
    downlines_out = straight_data[:, 2]
    downlines_in = straight_data[:, 3]
    matrix = sp.zeros(n)
    for i in range(n):
        matrix[i, uplines[i]] = 1-t
        matrix[i, downlines_out[i]] = t
        matrix[i, downlines_in[i]] = -1
        # 压线是1-t，被压前是-1，被压后是t
    # 计算行列式
    # print(matrix)
    # 一行行打印矩阵
    for i in range(n):
        print(matrix[i, :])
    # 移除第一行和第一列
    matrix.row_del(0)
    matrix.col_del(0)
    # print('Matrix after removing the first row and the first column:')
    # for i in range(n-1):
    #     print(matrix[i, :])
    det = sp.det(matrix)
    print("Det:\n", det)
    lowest_term = det.as_ordered_terms()[-1] # 最低次项
    # 获得最低次项的系数
    print(f'Lowest term: {lowest_term}')
    reduce_det = sp.div(det, lowest_term/abs(lowest_term.coeff(t)))[0]
    print("Reduced Det:\n", reduce_det)
    pass

# 样例：
if __name__ == "__main__":

    # raw_data = np.array([[1,3,2], [3,1,2], [2,3,1]])-1
    # raw_data = np.array([[0,2,3],[1,1,0],[2,3,0],[3,2,1]])
    # 3_1 trefoil
    # ANS: 1-t+t^2
    
    # raw_data = np.array([[1,2,0], [0,2,5],[3,4,1],[2,1,3],[4,5,3],[5,4,0]])
    # 6_1
    # ANS: 2-5*t+2*t^2

    raw_data = np.array([[3,6,1],[6,3,4],[1,6,5],[5,1,2],[2,5,4],[4,2,3]])-1
    # 知乎中的例子
    # ANS: -2t^3+5t^2-2t

    # raw_data = np.array([[0,0,1],[0,1,2],[0,2,3],[0,3,4],[0,0,4]])

    straightened_data = straighten(raw_data)
    alex_polynomial(straightened_data)
    pass
