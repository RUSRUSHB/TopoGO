"""步骤 5a: 通过基于滑动窗口移动的覆盖分析法探查图像交叉处重叠状况。

管线 A（Pipeline A）流派解决策略步骤：以类似探查游标的浮窗框架结构扫过标签结果底片的全部地方进行查找。
其通过检查一旦察觉当三个以上颜色线段发生聚头碰到一处时该如何判等，明确并找出哪个具体代号颜色代表正自上跨越穿行压顶(成为高线处位置)，而在底下潜过的两个被切断隐藏头尾的颜色代表各自是啥。
"""

import cv2
import numpy as np


def is_crossing(window: np.ndarray, window_size: int) -> bool:
    """查探和判定框定内的小区块视野里是否发生了符合要求的合格交叉重叠模式特征。

    一个确切真实的相交点它必然需要有着绝对完整凑满且仅仅限为三根前置线的主体参与存在且均匀有序地交替代排贴在这个方形小观测游窗画面的其边界边缘线位置 (必定有一处表现成了横穿画面两侧，其他两股则各自断为两处独有一次暴露)。

    Args:
        window: 直接由宏大的底板主画面裁切带下并保留在内部的二维局部小剖面图块。
        window_size: 其限定使用的那一个搜查窗口尺寸宽窄。

    Returns:
        假使该窗内情况已经构成了符合严格限制规定的真正节点区域时就会传出 True。
    """
    unique_labels, label_counts = np.unique(window, return_counts=True)
    background_label = unique_labels[-1]
    foreground_labels = unique_labels[unique_labels != background_label]

    # 一定要有确实数目恰好只等于是三的不同组成线条出现
    if len(foreground_labels) != 3:
        return False

    # 所以现身的部分都务必保有可做认定判断之底线占据数量 (不然就属误碰噪块)
    for label in unique_labels:
        if np.sum(window == label) < 5:
            return False

    # 周遭其后底板背景的体量应当保证满足相对足够宽广占比
    if np.sum(window == np.max(window)) < 20:
        return False

    # 以逆时针次序环场逐剥提取窗口这一大圈方形地块四处边缘暴露在外露脸显出身份的所有那些颜色的色块代号
    edge_labels = np.concatenate([
        window[0:-1, 0],
        window[-1, :],
        window[-2:0:-1, -1],
        window[0, -1:0:-1],
    ])
    edge_labels_shift = np.roll(edge_labels, len(edge_labels) // 2)

    # 去掉在边缘处无所不在占据庞大比重的空处背景元素代数成分
    edge_labels = edge_labels[edge_labels != background_label]
    edge_labels_shift = edge_labels_shift[edge_labels_shift != background_label]

    # 将完全就是并排在一起处于不中断纯粹由于线粗带来重复多占那类邻近完全一样数据实施整合归并和简化
    def merge_consecutive(arr):
        i = 0
        while i < len(arr) - 1:
            if arr[i] == arr[i + 1]:
                arr = np.delete(arr, i)
            else:
                i += 1
        return arr

    edge_labels = merge_consecutive(edge_labels)
    edge_labels_shift = merge_consecutive(edge_labels_shift)

    unique_labels = np.unique(edge_labels)
    edge_labels = edge_labels.tolist()
    edge_labels_shift = edge_labels_shift.tolist()

    # 设法辨认到底是谁才是高高盘在上面的顶端线位置 (它在总共整个大范围一趟边界统计下竟然高达3次冒头或是退一万步算作起头结尾2次边缘均占有的主角身份)
    up_label = -1
    for label in unique_labels:
        if edge_labels.count(label) == 3 or edge_labels_shift.count(label) == 3:
            up_label = label
            break
        if edge_labels.count(label) == 2 and edge_labels_shift.count(label) == 2:
            up_label = label
            break

    if up_label == -1:
        return False

    [down_label_1, down_label_2] = [x for x in unique_labels if x != up_label]

    # 最后再多核对下这种冒头和暴露排列先后顺序有没有构成交替这一个要求情况发生
    l1, l2, l3 = up_label, down_label_1, down_label_2
    possible_5 = [
        [l1, l3, l1, l2, l1], [l1, l2, l1, l3, l1],
        [l3, l1, l2, l1, l3], [l2, l1, l3, l1, l2],
    ]
    possible_4 = [
        [l1, l3, l1, l2], [l1, l2, l1, l3],
        [l3, l1, l2, l1], [l2, l1, l3, l1],
    ]

    is_valid = False
    if len(edge_labels) == 5:
        is_valid = edge_labels in possible_5
    elif len(edge_labels) == 4:
        is_valid = edge_labels in possible_4

    if not is_valid:
        return False

    # 核验在局域区段大家这些属于不同类别实实在在前沿的各路线条之间其近距离是否确凿存在碰头
    for i in range(len(foreground_labels)):
        for j in range(i + 1, len(foreground_labels)):
            coords_i = np.argwhere(window == foreground_labels[i])
            coords_j = np.argwhere(window == foreground_labels[j])
            min_dist = np.min(
                np.linalg.norm(coords_i[:, None] - coords_j, axis=-1)
            )
            if min_dist > 50:
                return False

    return True


def find_crossings(
    labels: np.ndarray, window_size: int = 5
) -> list[tuple[int, int, tuple]]:
    """以探照搜查方法巡遍整张被标签划分过的画卷查到且标记出来全局哪里蕴含存在过节点区域的所有确切地方。

    Args:
        labels: 二维度的主观区域画面全阵。
        window_size: 执行侦查浮块的该边长尺寸数字大小。

    Returns:
        送出一份打包完带有包含位置 (横向 x 座标, 纵向 y 核心座标, 被涉及牵连所有三根色彩代码线其各自身份号组汇聚包) 并成组陈列出去。
    """
    height, width = labels.shape
    half_window = window_size // 2
    crossings = []

    for y in range(half_window, height - half_window, 1):
        for x in range(half_window, width - half_window, 1):
            window = labels[
                y - half_window : y + half_window + 1,
                x - half_window : x + half_window + 1,
            ]
            if is_crossing(window, window_size):
                unique = tuple(
                    sorted(np.unique(window[window != np.max(window)]))
                )
                crossings.append((x, y, unique))

    return crossings


def determine_lines(
    window: np.ndarray,
) -> tuple[list[int], list[int]]:
    """仔细判断这个已经被锁定包含了有效扭结重叠发生地的独立窗口中谁占到便宜成为骑在上面之线位并且确认哪两个沦为下垫穿越底边地位的路线身份。

    判断理论基调即为如果单条颜色能够在该局部拥有绝对充盈巨大之体量自然可判决由于不受其它覆盖从而它占了那最毫无阻碍面积最大即成为那所谓的上方压线主力（而另一条被分割断在下面只能断成少得多体量的归其被压入底栏成分中）。

    Args:
        window: 包含了特定交触地点局部影像块的阵图。

    Returns:
        一组传令双向结果结构的元组即 (判断结果里算作顶部位置的序列线阵, 和底下埋葬两处部分所在的序列线段集合身份编号表册)。
    """
    background_label = np.max(window)

    top_edge = window[0, :]
    left_edge = window[:, 0]
    right_edge = window[:, -1]
    bottom_edge = window[-1, :]

    edge_labels = np.concatenate([top_edge, left_edge, right_edge, bottom_edge])
    unique_edge_labels, edge_label_counts = np.unique(
        edge_labels, return_counts=True
    )

    edge_dict = {
        label: count
        for label, count in zip(unique_edge_labels, edge_label_counts)
        if label != background_label
    }

    if not edge_dict:
        return [], []

    # 算出这视窗框架之中所有每一类带有颜色的图块分别究竟能霸占夺走几多份额的体积比表面积数值
    area_dict = {
        label: np.sum(window == label) for label in edge_dict.keys()
    }

    # 因为没有经过被断裂所以体量份额最大必然等于头顶上那不可阻隔连续越过的这处完整线条自身身份
    top_label = max(area_dict, key=area_dict.get)
    top_line = [top_label]
    bottom_line = [l for l in edge_dict.keys() if l != top_label]

    return top_line, bottom_line


def process_image(
    labels: np.ndarray, window_size: int = 5
) -> list[tuple[int, int, list, list]]:
    """提供一项整套一体把涵盖寻找，裁断位置加上分析解构上下面方位角色分配等工作内容一次打通管用的全程全域图像探雷扫描检举操作业务链流程汇总实现。

    Args:
        labels: 剔除了与处理无关废弃色块之后完整大张图卷标记数组阵式基块板图。
        window_size: 用于当框扫平滑使用的主导寻查移动镜窗之比例型号尺寸情况。

    Returns:
        将所有能不重不漏找到包含独立出现一次完整汇聚碰头结构各要素打包构成为：
        (相交中轴x, 相交中轴y, 表皮最高上过线表代码单合集, 断裂在背后的下过线身份双重集)，
        一份由这些独特每一位参与相接节点情况信息总成之后呈现的大清单汇总回报簿录目录。
    """
    crossings = find_crossings(labels, window_size)
    seen_combinations = set()
    unique_crossings = []

    for crossing in crossings:
        x, y, unique_labels = crossing
        if unique_labels not in seen_combinations:
            seen_combinations.add(unique_labels)
            half = window_size // 2
            window = labels[
                y - half : y + half + 1,
                x - half : x + half + 1,
            ]
            top_line, bottom_line = determine_lines(window)

            if len(top_line) != 1 or len(bottom_line) != 2:
                continue
            unique_crossings.append((x, y, top_line, bottom_line))

    return unique_crossings
