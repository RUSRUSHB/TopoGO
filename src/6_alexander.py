"""步骤 6: 终极神兵目标产物解构推到解脱计算求出属于亚历山大多项式这一终极大关。

此步接纳收束吸纳由前期交叉信息汇总而来的打下夯实数据根基然后以此求解转化诞生反映所处目标扭结本质神髓特征其内在特有的拓扑量纲之最高无量法宝——也即那个被数学界传颂的名称为“亚历山大多项式”的最终方程式表述结果。
所有曾经过收容进来的那份繁密交叉图版表册数据将在第一步必先要历经一道被命名“顺顺理顺”（也就是使其排个规矩确保能致使从头追溯一路走来到尾每根连线都绝对恰好经历参与且仅仅仅只经历一回身为“下沉受压从这处出来”，和仅且唯一充当一回被分配称作“此下面向内钻身陷其下”两者双重身份考验轮回之过程）；等理顺以后此多项式才能水到渠成般利用借助对于这经过裁剪之后变薄缩减型了的那个著名亚历山大大局势演变矩阵（Alexander matrix）再行做推演提取求其内部根本行列数值行列式最终破解获得真谛答案之。
"""

import numpy as np
import sympy as sp


def straighten(raw_data: np.ndarray) -> np.ndarray:
    """去理清、排转顺畅所有扭结跨度数据给整合成一种能贯通首尾不矛盾连续一气可无碍流通查阅前后相随关系跟脚之秩序严明通衢一致排列表制。

    开端时由这头源起初始第一起交叉作为着手起源下手地，向着断流指着朝下一路紧追踪顺那些被压埋地处接茬追踪不断一直找续，以此产发衍生建立构造形成一项可包管绝无遗漏跟混乱且令致随便哪段局部断位片段线条最终都会毫无二致仅限于恰恰暴露且有一次是作为呈现它乃下穿钻出这一面目形态且必定还有仅只单回充当着即将潜下入阵之角色使命二种对等义务双担身分的这么一副极度有顺畅纪律保证的阵列清点表格式。

    Args:
        raw_data: 原本长相形似有着 (全部交叉节点总计数量N乘上 3 类线情况列)这般大体貌规格的一类二维横行陈列表格其行里并排登记收纳的即如 [头上有大高跨过顶线, 最底下两边之线其 1, 最底下两边之线其 2] 像如此这样这三样玩意。

    Returns:
        改造加工重获新生并升级配置变为囊括且具有起 (行数等同交节点总合数量N, 但扩宽扩展成其包括 4 要素一并长短列数维系尺寸) 此等样貌规模并且在各自行格行间已排好了记录书写的内容包括犹如形同是 [属于其该交叉节所被编定座次号码, 上首大跨顶盖过越线, 有一条处于底层下面将窜走往外伸延之线段身份, 与另一位深潜于这底下身形正指向将扎内向内陷进去延伸方向走势之段落身分标示] 四者合并打包而就阵式形态呈现出来之一番好物结果。
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
    """清理规整、裁剪精炼修去累加浮夸从而达成对于一个杂乱繁杂亚历山大大结构复杂多项巨长程式的统筹标准化一缩减至理出最平实原本核心真实基础之清爽形态表达。

    把那些明显作为共有附带可剥离公倍之拖泥带水累赘最低起阶初次变量t相通通都除尽抽抽掉并在其中连带着微调统一协调矫正调整它符号让整体能维持和保住令该程式处于常数阶层起底那个那端常数所在(若缺则指代为首起那项)必须务必要恒久稳定呈现维系为一个光明昭昭且必然是大过0的正数符号作领引前言这种庄严表观态。

    Args:
        poly: 其所面对着正待需实施化解操刀待理的尚未进行精修前那份包裹牵挂涵盖诸多交缠变量代数t因子的原生未化整零数式原本。

    Returns:
        完成整肃打磨梳裁修饰过后纯净而化归无妄简约真章最终敲定完成好归一精简态格式的最终规范版成品代数表示方程式图板展现物。
    """
    t = sp.Symbol("t")

    const_term = poly.as_coefficients_dict().get(t**0, 0)

    if const_term != 0:
        return poly if const_term > 0 else -poly
    else:
        terms = poly.as_ordered_terms()
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
    """凭恃仗用依傍取自从打被先前一轮已弄得理顺打散开齐过有条不紊顺序摆布罗列成阵过那些明堂端正所有交交叉叉重重跌跌路网底牌家底材料，藉此直接去演化排阵推翻演算求这获取名为亚历山大大神其最高境界象征的方程式精要奥秘。

    依规据典正式隆重推演铺布张开这座称做属于亚历山大大盘大势之大阵矩阵式：
      - 于居高俯视覆盖占居越顶之上行长线那一专属位置栏目列内均统一赋给注入那作为有带有其代数参数含义之代数量级因数 (即 1 - t ；代表其在这一特定局段扮演着其既要经历遮挡却又独立一面的这般本色贡献) 
      - 与那充任陷到深坑从深底下往出逃生外窜走之钻出低贱路线栏落列名之下均不约而同统一塞贴着这标志有它这一流窜角色符号标记参数之代表常数量度因数 t 之挂冠
      - 再还有与另外那个则充任是埋伏钻首内入埋首其中入阵身份职责所待底下逃生路线入线位一栏一并全部被赋予签押带有这个常数因子刻印记分因其位即此代表性常数因量 -1 的徽记

    这之后，即可发动阵势并起势大局推衍算出位于这座大阵将剔掉割让最初那极开始之排首起先一行一列，收拢成型变身为体量缩为（节点N个减 1）横乘（N数减 1）竖这等规模式之子矩阵残部之阵局精华提取精华部分其内在根源最根本实质代表了总体阵胆特质精神其蕴核心那个内在精华指标数字即也就是被名作阵眼行列根本式值。

    Args:
        straight_data: 自那边先头队伍里由 straighten() 专职手下做完苦工规整清算过后出单交托过来的属于有着 (多达N数之总共有N条交叉条数其长×足足 4大要素列齐之宽) 格式长相那般规范端方齐整排好有秩资料板子大阵大样表底档本。

    Returns:
        包含交接了属于两样珍罕重大之成果成就合成元生大包裹一组包裹集合打包发送回报回来。其里面具体则就是带有两项这分装：一是如期圆满得手这件经已剔裁冗余整理妥帖后最干脆洗练完满不枝不蔓精简最终化了亚历山大所表征最内核结果；二是与之附列而同样一并寄上这虽尚留浑浊而其未精未加工这依然呈现未提纯裁剪之毛糙原原本本粗狂状态那只行列式真身上一份原态胚表影本身子影拓原本之版，如此二物合并捆缚呈奏给您)。
    """
    t = sp.symbols("t")
    n = len(straight_data)

    uplines = straight_data[:, 1]
    downlines_out = straight_data[:, 2]
    downlines_in = straight_data[:, 3]

    # Map downline-out values to row indices
    mapped_lines = np.arange(n)
    mapping = dict(zip(downlines_out, mapped_lines))

    uplines = np.array([mapping[x] for x in uplines])
    downlines_out = np.array([mapping[x] for x in downlines_out])
    downlines_in = np.array([mapping[x] for x in downlines_in])

    # Build the Alexander matrix
    matrix = sp.zeros(n)
    for i in range(n):
        matrix[i, uplines[i]] = 1 - t
        matrix[i, downlines_out[i]] = t
        matrix[i, downlines_in[i]] = -1

    # Remove first row and column (standard Alexander polynomial convention)
    matrix.row_del(0)
    matrix.col_del(0)

    det = sp.det(matrix)
    reduced_det = reduce_polynomial(det)

    return reduced_det, det


if __name__ == "__main__":
    # Example: 6_1 knot
    # Expected: 2 - 5*t + 2*t^2
    raw_data = np.array([
        [1, 2, 0], [0, 2, 5], [3, 4, 1],
        [2, 1, 3], [4, 5, 3], [5, 4, 0],
    ])
    straightened_data = straighten(raw_data)
    reduced, raw = alex_polynomial(straightened_data)
    print(f"Alexander polynomial: {reduced}")
