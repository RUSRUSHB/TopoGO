"""步骤 3: 标签调整与重组操作的工具集合。

包含在分割步骤完毕后针对图像标签开展排序整理、各图层拆分提取，以及后期非线条部分合并的各项辅助函数。
"""

import numpy as np


def make_bg_biggest(labels: np.ndarray) -> np.ndarray:
    """交换标签阵列，保证背景部分(即左上角)总是被指定成了所有数字里最大那一个值。

    由于背景应当等于最大数值标签，这条准则在贯穿处理管线的上下文都有使用约定。

    Args:
        labels: 二维图面的原始分区标签数组。

    Returns:
        由于背景所处位置改变从而让它等于最大号码数值后的标签数组返回结构。
    """
    max_label = np.max(labels)
    background_label = labels[0, 0]
    temp_label = max_label + 1

    if background_label != max_label:
        labels[labels == max_label] = temp_label
        labels[labels == background_label] = max_label
        labels[labels == temp_label] = background_label

    return labels


def arrange_labels(input_array: np.ndarray) -> np.ndarray:
    """重新调整整个标签系列使其变成从数字 1 开始逐个连续递增且背景一定是被排到最后成为最大值的数列。

    Args:
        input_array: 由之前初步图像提取所分出来带有多组离散组件标号数组。

    Returns:
        包含已经全部连续替换至 1, 2, ..., N 等形式 (其中背景值为最终这一个 N ) 的对应重编版数组。
    """
    max_label = np.max(input_array)
    input_array[input_array == 0] = max_label + 1

    unique_labels = np.unique(input_array)
    arranged_labels = np.arange(1, len(unique_labels) + 1)

    label_mapping = dict(zip(unique_labels, arranged_labels))
    arranged_array = np.vectorize(label_mapping.get)(input_array)
    arranged_array = make_bg_biggest(arranged_array)

    return arranged_array


def separate_labels(input_array: np.ndarray) -> np.ndarray:
    """各自抽取每一个互不重叠唯一的标签至属于它自己的一套图层阵列上形成层切片效果。

    Args:
        input_array: 输入过来的原始二维平面的带成分划分数值的组件矩阵。

    Returns:
        构建出一个带有此特级多维切片数组，其三维特征将包含厚度（对应图层数目就是所有的那些独立独特组件数）、外加长宽形态；其中切出来仅包含像素属于这个分层对应单组件的数据值。
    """
    unique_labels = np.unique(input_array)
    height, width = input_array.shape
    num_labels = len(unique_labels)
    separated_layers = np.zeros(
        (num_labels, height, width), dtype=input_array.dtype
    )

    for i, label in enumerate(unique_labels):
        layer = np.where(input_array == label, label, 0)
        separated_layers[i] = layer

    return separated_layers


def unify_non_line_segments(
    labels: np.ndarray, classification: dict[int, bool]
) -> np.ndarray:
    """对那些实质上压根不是线条的斑点成分通通做一次清洗并跟底色背景混合为一体。

    Args:
        labels: 二维尺寸的全部数字标签总合数组矩阵。
        classification: 装载所有成分标号其对应 真(这是线段) 或 假(并非线形结构) 信息的映射关联字典集。

    Returns:
        用当前原本底色的代表号取代原本属于各色奇怪离散块所在的位置并作为替换完毕的新返回矩阵内容。
    """
    background_label = labels[0, 0]

    for label, is_line in classification.items():
        if not is_line:
            labels[labels == label] = background_label

    return labels


def extract_labels_from_crossings(crossings: list) -> np.ndarray:
    """快速剥离并在交叉碰撞探测分析报告中只获取各处交汇压叠线路的关键号码标记。

    接收 cross_detect 那个负责判定上跨与被覆盖部分的汇总返回体，进一步格式化抽取转换成为每一行均是标准模式[最上层线, 底下经过线段1, 底下另外方向线段2]这种样式 (N 乘 3 的整数版) 的记录大表。

    Args:
        crossings: 前置步骤获取到的每一项包括了 (x, y, 顶端上跨线号, 和被掩盖两下压部分线段数组) 排列格式的复杂组合项目大列表。

    Returns:
        规范整理出来并且仅存由具体那哪几号交叉线路汇聚情况纯粹构成的列二维大清单矩阵。
    """
    crossings_arr = np.zeros((len(crossings), 3), dtype=int)
    for i, crossing in enumerate(crossings):
        x, y, top_line, bottom_line = crossing
        crossings_arr[i, 0] = top_line[0]
        crossings_arr[i, 1] = bottom_line[0]
        crossings_arr[i, 2] = bottom_line[1]
    return crossings_arr


def unify_crossings(crossings_arr: np.ndarray) -> np.ndarray:
    """再次整体地打乱然后顺序重新按 0 算起向后派发重塑一套新整数形式以映射所有的各接触压叠线条，
    其结果完毕之后，最后必定还会沿着首列那一批处在高处（上跨）编号顺序对其完成整体逐行重整理。

    Args:
        crossings_arr: 即前头所获取下来的只装那些上上下下线数字组态对应交叉关系的二维清整资料。

    Returns:
        被整改规范映射且经过针对头部所在编号递增升序整理完之后最新版的那一套集合大名单。
    """
    unified = crossings_arr.copy()
    unique_labels = np.unique(crossings_arr)
    label_mapping = dict(zip(unique_labels, np.arange(len(unique_labels))))

    for i in range(len(crossings_arr)):
        for j in range(3):
            unified[i, j] = label_mapping[crossings_arr[i, j]]

    # 对处在第一列表面交叉的那条线路按照从小向大进行自然梳理排序。
    unified = unified[np.argsort(unified[:, 0])]
    return unified
