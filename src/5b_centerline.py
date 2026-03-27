"""步骤 5b: 基于提取主心骨中心线分析法勘定向重叠结点位置的工作流。

属于管线 B (Pipeline B) 的一种解决手段法门：首先把各段已分离带编号形状粗壮的躯体剥离开来并通过细分抽取缩减变为仅有一个像素宽度的主中心线 (骨架结构化)，接着定位识别其各向尽头部位的极点端点，随后再去顺藤摸瓜勘察是否存在跨过第三类编号地表后将两个同色系尽头断联拼接在一块儿相通的情况以此从而将整个隐秘于遮蔽底下交叉位置发掘查实出来。
"""

import cv2
import numpy as np
from scipy.spatial import KDTree
import skimage.draw


def extract_centerline(binary_image: np.ndarray) -> np.ndarray:
    """去将原本丰满占据面积的一个白色大形状躯体抽丝剥茧还原为其单薄最里层的骨架中轴线走向。

    做法是应用测算离边缘界限的纵深空间距离转换配合进行数值阈值界限切割及后续用上专门用于进行影像学身骨架剥离化变薄的技术函数处理收束搞定。

    Args:
        binary_image: 那副具有白区块占据和黑暗空白背景的只有一个纯色彩频道的双色彩二极相图。

    Returns:
        最终蜕下其余边肉而仅仅剩下的那一具同样用二元（像素0或255表达）来陈列显示出的极细线形残骸走向骨架图模板。
    """
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 3)
    dist_transform = cv2.normalize(
        dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX
    )
    _, skeleton = cv2.threshold(
        dist_transform, 0.2 * np.max(dist_transform), 1.0, cv2.THRESH_BINARY
    )
    skeleton = (skeleton * 255).astype(np.uint8)
    skeleton = cv2.ximgproc.thinning(skeleton)
    return skeleton


def extract_centerlines_for_labels(labels: np.ndarray) -> np.ndarray:
    """给那些全都在这前排里登场挂着号的全部实打实组成有效成分标签均提取制备出它们独家的主干中心骨架线路。

    Args:
        labels: 标好字号有区分色彩的一副满是分块标号码的整图大阵图 (一般用到了像 [5, 5] 这里所在的码作为整个基调大背景区）。

    Returns:
        一样拥有跟进原来源板等大同宽的长宽阵型，但是其现在只保留极少数真正属于中心线上的那一细排被印上了专属标号的像素点，别的一概通统刷去变成0。
    """
    unique_labels = np.unique(labels)
    centerlines = np.zeros_like(labels, dtype=np.uint8)
    background_label = labels[5, 5]

    for label in unique_labels:
        if label == background_label:
            continue
        mask = (labels == label).astype(np.uint8) * 255
        centerline = extract_centerline(mask)
        centerlines[centerline > 0] = label

    return centerlines


def get_endpoints(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """定位找出这一幅残图里其所有形似一字长蛇阵断裂最外侧那头仅仅孤悬在外只接触一个同胞伴随相连的极度尽头之处。

    Args:
        skeleton: 一整张干干净净只剩极瘦条线穿行在其中的骨架构像图册。

    Returns:
        包裹着一份含有多个类似于如 (纵轴位置y, 横轴所在x) 那般定位精准无误标识终极断裂口断头座标端点的成串清单集合体。
    """
    endpoints = []
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x] > 0:
                neighborhood = skeleton[y - 1 : y + 2, x - 1 : x + 2]
                if np.sum(neighborhood) == 2:  # self + 1 neighbour 只包含自己与外侧孤立存在的一个陪伴点
                    endpoints.append((y, x))
    return endpoints


def dilate_line(
    image: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], width: int = 3
) -> np.ndarray:
    """在一幅全凭真假（True 或 False）布尔型作判定画质的模板里挥洒连接起两个目标座标并且将这一相通连接的线段予以粗壮膨大化扩宽操作。

    Args:
        image: 用于作为接受这般写画挥扫改动而承载容纳变更之后墨迹影响的布尔形式大原版地图册。
        pt1: 起笔初始一头的明确位标座次 (y 轴竖列, x 轴横排)。
        pt2: 停笔收尾降落定点目标的终局精确经纬 (y 列向, x 行向)。
        width: 具体执行增幅把轨迹墨水线变多大变粗实的实际膨胀展宽规模值。

    Returns:
        最终接纳并且完好拓印好那粗实过相接线后经过更新了新局面的图像结果结构体。
    """
    rr, cc = skimage.draw.line(pt1[0], pt1[1], pt2[0], pt2[1])
    for dr in range(-width // 2, width // 2 + 1):
        for dc in range(-width // 2, width // 2 + 1):
            r, c = rr + dr, cc + dc
            valid = (r >= 0) & (r < image.shape[0]) & (c >= 0) & (c < image.shape[1])
            image[r[valid], c[valid]] = True
    return image


def map_labels_to_consecutive_integers(
    crossings: list[list[int]],
) -> tuple[list[list[int]], dict[int, int]]:
    """将原本混杂不齐的各路重合叠压涉及相撞对象编码，洗牌均统一由最低点即 0 为起点向后依次重排对齐赋予新的一套完备连贯连续整数代号号源。

    Args:
        crossings: 原初包含像类似于这样装束了各种带有 [上方线标兵1号, 遮蔽藏在底下暗线2号, 断线同袍暗线3号] 形制情况交代的三连结构名录组。

    Returns:
        捆绑输送出一款含有形似如下 (整编过后重新下发身份编号的新式全体名单阵列, 一份足以从旧时编号对照翻译查询当今新任派发身份密码对应法则密码本字典薄) 形式的双元组合物。
    """
    unique_labels = set()
    for crossing in crossings:
        unique_labels.update(crossing)

    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    mapped = [
        [label_mapping[l1], label_mapping[l2], label_mapping[l3]]
        for l1, l2, l3 in crossings
    ]

    return mapped, label_mapping


def examine_crossings(
    centerlines: np.ndarray, is_visualize: bool = False
) -> list[list[int]]:
    """凭据分析端点处那些截断面所在情况来实施逆推倒算揪寻找一切关于藏在底下交叉处的真相大白工作核心机构。

    面对每一双有着极大关联可能却被割断隔开而具有相异归属称号标签头部的终端极点而言采取以下探测法：
    1. 在二者间隔空气真空地带隔空连拉飞出构建虚拟并刻意粗壮加粗试探搭通一条连带实线段络联两者。
    2. 核验搜查有还存在着到底被哪个另外不相干的第三方越线干扰阻拦半途截断了试探搭建出这段跨线天桥地界。
    3. 要是这桥身下面不偏不歪确确实实刚好只是压上和骑着只存在单独一个唯一一个此外其余第三方它人标记路段时，此情况即确认为符合铁证无疑地发现在此处发生了掩盖跨线越线节点一桩事实。

    Args:
        centerlines: 仅残余留下瘦削中干标号那些作为线骨组成身段构件所在的像素点的全体二维大合集。
        is_visualize: 用控制扳机开启决断这一巡狩勘算途次之中途时候产生一系列中途演算画面需不需要显化放出给别人观察参谋。

    Returns:
        带回来属于各方搜查斩获大胜包含形如形同如这类装填有 [架空乘高过之上界压制牌号, 埋于其底遭受割断底层半段残肢甲, 同是埋头底圈底层半段残肢乙] 交代陈列方式交接事实集合的清单总报。
    """
    endpoints = {}
    unique_labels = np.unique(centerlines)
    background_label = centerlines[5, 5]

    for label in unique_labels:
        if label == background_label:
            continue
        mask = centerlines == label
        endpoints[label] = get_endpoints(mask)

    all_endpoints = [
        (y, x, label)
        for label, points in endpoints.items()
        for y, x in points
    ]

    if not all_endpoints:
        return []

    # Visualization setup
    if is_visualize:
        import matplotlib.pyplot as plt
        vis_img = centerlines.copy()
        _, vis_img = cv2.threshold(vis_img, 0, 255, cv2.THRESH_BINARY)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        for y, x, label in all_endpoints:
            cv2.circle(vis_img, (x, y), 2, (0, 0, 255), -1)
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_img)
        plt.axis("off")
        plt.title("Endpoints Visualization")
        plt.show()

    # Build KD-tree for nearest-neighbour queries
    tree = KDTree([(y, x) for y, x, _ in all_endpoints])
    crossings = []
    matched_endpoints = set()

    for y, x, label in all_endpoints:
        distances, indices = tree.query((y, x), k=6)
        for idx in indices[1:]:  # Skip self
            ny, nx, nlabel = all_endpoints[idx]
            if nlabel == label:
                continue

            if (y, x) in matched_endpoints or (ny, nx) in matched_endpoints:
                continue

            # Draw test line between the two endpoints
            test_image = np.zeros_like(centerlines, dtype=bool)
            test_image = dilate_line(test_image, (y, x), (ny, nx), width=3)

            # Check which labels the line crosses
            intersecting_labels = set(centerlines[test_image])
            intersecting_labels.discard(background_label)
            intersecting_labels.discard(label)
            intersecting_labels.discard(nlabel)

            # Valid crossing: exactly one other label between endpoints
            if len(intersecting_labels) == 1:
                crossing = [list(intersecting_labels)[0], label, nlabel]
                crossing[1], crossing[2] = sorted([label, nlabel])

                if crossing not in crossings:
                    crossings.append(crossing)
                    matched_endpoints.update([(y, x), (ny, nx)])

                    if is_visualize:
                        cv2.circle(vis_img, (x, y), 2, (0, 255, 0), -1)
                        cv2.circle(vis_img, (nx, ny), 2, (0, 255, 0), -1)
                        cv2.line(vis_img, (x, y), (nx, ny), (0, 255, 0), 1)

    if is_visualize:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_img)
        plt.axis("off")
        plt.title("Crossings Visualization")
        plt.show()

    return crossings
