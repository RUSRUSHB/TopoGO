"""步骤 6: 亚历山大多项式（Alexander Polynomial）计算。

本模块接收前期提取的交叉点信息，通过构建 Alexander 矩阵并计算其次行列式，
最终求得该扭结的拓扑不变量——亚历山大多项式。
算法流程包括：交叉点信息的规范化（Straighten）、矩阵构建以及多项式的标准化项化简。
"""

import numpy as np
import sympy as sp


def straighten(raw_data: np.ndarray) -> np.ndarray:
    """对原始交叉点数据进行规范化处理，使其符合 Alexander 矩阵构建所需的顺序。

    通过追踪线段的连通关系，确保每个交叉点处的线段角色（上方跨越、下方移入、下方移出）
    能够形成一个连续且一致的循环序列。

    Args:
        raw_data: 原始交叉点数据矩阵 (N, 3)，格式为 [上方线标签, 下方线 A, 下方线 B]。

    Returns:
        规范化后的数据矩阵 (N, 4)，格式为 [交叉点索引, 上方线标签, 移出线标签, 移入线标签]。
    """
    raw_uplines = raw_data[:, 0]
    raw_downlines = raw_data[:, 1:]

    crossing_num = len(raw_data)
    crossing_list = [0]
    start_line_list = [raw_downlines[0, 0]]
    target = raw_downlines[0, 0]

    for i in range(crossing_num):
        for j in range(crossing_num):
            if target in raw_downlines[j] and j not in crossing_list:
                crossing_list.append(j)
                target = [x for x in raw_downlines[j] if x != target][0]
                start_line_list.append(target)
                break

    straight_data = np.zeros([crossing_num, 4], dtype=int)
    for i in range(crossing_num):
        ci = crossing_list[i]
        straight_data[i, 0] = ci
        straight_data[i, 1] = raw_uplines[ci]
        straight_data[i, 2] = start_line_list[i]
        others = [x for x in raw_downlines[ci] if x != straight_data[i, 2]]
        straight_data[i, 3] = others[0]

    return straight_data


def reduce_polynomial(poly: sp.Expr) -> sp.Expr:
    """对求得的多项式进行归一化处理。

    移除多项式中除不尽的最低阶变量因式 t^k，并调整系数符号，
    使常数项（或首项）系数为正，以符合亚历山大多项式的标准表达形式。

    Args:
        poly: 原始行列式计算出的多项式表达式。

    Returns:
        归一化后的亚历山大多项式。
    """
    t = sp.Symbol("t")

    const_term = poly.as_coefficients_dict().get(t**0, 0)

    if const_term != 0:
        return poly if const_term > 0 else -poly
    else:
        terms = poly.as_ordered_terms()
        if not terms:
            return poly
        lowest_term = terms[-1]
        lowest_coeff, lowest_exp = lowest_term.as_coeff_exponent(t)

        if lowest_coeff < 0:
            poly = -poly

        lowest_monom = t**lowest_exp
        poly = sp.div(poly, lowest_monom)[0]
        return poly


def alex_polynomial(
    straight_data: np.ndarray,
) -> tuple[sp.Expr, sp.Expr]:
    """根据规范化后的交叉点数据，计算亚历山大多项式。

    构建 Alexander 矩阵的规则：
    - 上方跨越线位置填充 1 - t
    - 下方移出线位置填充 t
    - 下方移入线位置填充 -1

    依据拓扑学定义，删除矩阵的第一行和第一列后，计算其行列式（Minor），
    最后进行多项式化简。

    Args:
        straight_data: 规范化后的交叉点关系矩阵 (N, 4)。

    Returns:
        包含 (归一化多项式, 原始行列式结果) 的元组。
    """
    t = sp.symbols("t")
    n = len(straight_data)

    uplines = straight_data[:, 1]
    downlines_out = straight_data[:, 2]
    downlines_in = straight_data[:, 3]

    # 将下游线标签映射为行索引
    mapped_lines = np.arange(n)
    mapping = dict(zip(downlines_out, mapped_lines))

    uplines = np.array([mapping[x] for x in uplines])
    downlines_out = np.array([mapping[x] for x in downlines_out])
    downlines_in = np.array([mapping[x] for x in downlines_in])

    # 构建 Alexander 矩阵
    matrix = sp.zeros(n)
    for i in range(n):
        matrix[i, uplines[i]] = 1 - t
        matrix[i, downlines_out[i]] = t
        matrix[i, downlines_in[i]] = -1

    # 根据标准定义，移除首行首列以计算次行列式
    matrix_minor = matrix.copy()
    matrix_minor.row_del(0)
    matrix_minor.col_del(0)

    det = sp.det(matrix_minor)
    reduced_det = reduce_polynomial(det)

    return reduced_det, det


if __name__ == "__main__":
    # 示例: 6_1 扭结
    # 预期结果: 2 - 5*t + 2*t^2
    raw_data = np.array([
        [1, 2, 0], [0, 2, 5], [3, 4, 1],
        [2, 1, 3], [4, 5, 3], [5, 4, 0],
    ])
    straightened_data = straighten(raw_data)
    reduced, raw = alex_polynomial(straightened_data)
    print(f"Alexander polynomial: {reduced}")

