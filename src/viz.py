"""包含一些给带标签划分标记有颜有色图文提供展现显现使用到的关于视觉展示呈现用途小工具组件。

囊括了可执行包括展示涂抹具备随意五花八门不同随机打乱缤纷多彩绚烂色泽、印上表明各个所属派系身份记号小字、外加以备可以添加在原处叠压蒙板上供框出描出关于相交重点发生交警区域圈画线方框等的诸多各类辅助图形显示附加用途显示方法操作函数库集合功能模块。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random


def visualize(img: np.ndarray, title: str, colorful: bool = False) -> None:
    """去依靠唤起挂靠 matplotlib 这个老牌大户的深厚实力手段给直观直截痛爽干净利落地陈展放出图像到画面之表处并做示人展出。

    Args:
        img: 那张目前处于等待期盼欲将期展示流放让用户过目视探这影像本尊阵列正主儿源头版资料身。
        title: 给马上破空腾挪要弹射张开弹出新建窗口图画配设所命写而起之的门庭牌匾字标题明号抬头。
        colorful: 即这充作一枚决绝之开关闸门参数用来定夺：假如真命拨开此闸赋以设“确是它（True）”参数之姿其必教展图换发满盘全版之五彩缤纷大千世界大放它满眼真如炫色之景; 不然那相反若是按它关门定它非确设 false 值那么其定教此图归向一意枯坐冷调清冷之灰色基调素雅极简风格以灰面展示于众了。
    """
    plt.figure()
    if colorful:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.show()


def show_each_components(separated_layers: np.ndarray) -> None:
    """去逐层剥离一层再层拨弄开依次递进分发并且展示令看这每一单单由其自身标定认领的一重带单独一个标识印号所属单独图纸涂层板以它那种独有孤芳呈现灰冷阴灰色面貌分别呈现展览展出之景象。

    Args:
        separated_layers: 由以前分割出来那一捆内所包覆堆叠有起这含有 (总体各异种类数目量总标记层片之多, 这阵画面之本身之高 H 尺码长, 这阵画面宽度横栏它所有长 W 尺码长) 那般全满大立体多维厚度量阵巨制厚型叠册之版簿格式全样底板子集其一立体图层源块原本库基材原本。
    """
    num_labels, height, width = separated_layers.shape
    for i in range(num_labels):
        layer = separated_layers[i]
        plt.imshow(layer, cmap="gray")
        plt.title(f"Component {i}")
        plt.show()


def _is_surrounded_by_same_label(
    labels: np.ndarray, x: int, y: int, label: int, radius: int = 6
) -> bool:
    """悄然无声去四处打探核实验证查看于指定坐标定位落点处于落位点那处于被安处所在点 (x 列座向排位处, y 即所在高度行数位列点处) 向周围放眼望是不是能够完全彻彻底绝皆已被同样一种具有无异共源属性一样之身分配色标识完全给层层全周遍一切统共包裹紧紧包围住这等景置和这般四面皆同样这现象其果真存在不。

    这番四望查探之大工程之所以需开展缘由盖起乃是这便被使被派遣来要其在此图图茫茫人海像素点子之中寻访找到这么一块可以放宽一百个心绝对堪称妥帖平平且为至关安全保险不可之宝地位置好用此用来放落安置书写要其书在图版面上用之注解图样印章留名标注的短促标识铭牌标签字数，以用来做去使得将要去安放在这儿上面所留下铭文印记名条能够大保其安全使得绝不会使其因此跨越过从而同其他周遭不同族党别之异色标记划定边界之标界线相产生混成一团交错互相冒犯覆盖乱压成不当纠纷交叠其灾难乱景发生而已特留查察安全间距保底缓冲间而设专用专岗。

    Args:
        labels: 二维度的主观区域画面全阵二维带标数字基板。
        x: 横向那道行着在坐标格的落子定位坐标值。
        y: 纵向那道横着在这坐标方格的立脚之坐标所到处高度值。
        label: 所内心巴望着满时期盼预设之属于当前需考察的这主家其标签代码指号码牌值。
        radius: 作为在以所给原核立脚之标的立脚点向四周向外边扩展做这视线探查勘扫雷达一扫覆盖包罗范围时采用画圈向外画这周巡圈大其之多寡测算值参数设置大小。

    Returns:
        只要能实打实全盘地彻彻底底绝无留任何一点疏漏查遍这所被要求所圈定规定探测范围半径幅面这雷达扫查大圆覆盖全场之中均确悉属于并且所被认属此唯一同一个名唤为要核查同这一模一样标签身份归属存在现象方能给打一个大圆满确认且这事才得以能确为无差方放传过个这是“确已如其实（True）”结果，如但逢有一个哪怕个别例外其也将立马作停而打回并作定那断定其并非这等“大不为假现象”。
    """
    height, width = labels.shape
    for dx in range(-radius, radius):
        for dy in range(-radius, radius):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if labels[ny, nx] != label:
                    return False
    return True


def visualize_labels_with_random_color_and_text(labels: np.ndarray) -> None:
    """全面展示并高亮打点输出显示这在版面上这带被分别做定贴发配标属有一重重标签分区版图底片并对其每一单部分随兴涂画随意抛泼去完全不随既定去随其偶然这般施染以不同之诸类毫无特定安排由机器抽奖式随拿抽获取来彩色涂染漆并在此外还会再给那每一色堆正上方再印戳加给贴覆悬带带数字这这字型牌坊这标记名讳注解版这番大动作展示出来呈现以视觉之上以给众旁所观察目览大宴这图展示盛况。

    由于每一独一份独领风尚不重号独特专属独门归落之这一派独归门头名牌号此后定会被机器在背后从那千千色缸其任意抛点乱丢不随机乱蒙抽取到一种不预知色样而作它的代系归身专属这专属代表这一色旗专属领色归它这一派一脉单穿传这代表色泽作袍披在己身。这此一后伴之随即伴有的一串字面文字标注字符其它将被选择落脚安置给安置于一片这在其图里版这处周边全方位皆被自己自家人全都是包在同类此同种标记一伙阵之中重重其里这团环抱保护极度安全内部心腹地区方进行放手给在其身之上面印打出这一方这标注名称铭牌字形标识去定安好在那。

    Args:
        labels: 标好记号有定分划定这拥有此二向二维平摊规格带编码数位二维组列印的此阵面画板这二维的数字底标号印版其图列基础资料原本库阵。
    """
    unique_labels = np.unique(labels)
    color_dict = {label: np.random.rand(3) for label in unique_labels}

    height, width = labels.shape
    colored_image = np.ones((height, width, 3))

    for label in unique_labels:
        mask = labels == label
        colored_image[mask] = color_dict[label]

    plt.figure(figsize=(10, 10))
    plt.imshow(colored_image)

    for label in unique_labels:
        mask = labels == label
        if np.any(mask):
            yx_pairs = np.argwhere(mask)
            for y, x in yx_pairs:
                if _is_surrounded_by_same_label(labels, x, y, label):
                    plt.text(
                        x, y, str(label),
                        color="white", fontsize=12,
                        ha="center", va="center",
                    )
                    break

    patches = [
        mpatches.Patch(color=color_dict[label], label=f"Label {label}")
        for label in unique_labels
    ]
    plt.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    plt.axis("off")
    plt.show()


def visualize_labels_with_crossings(
    labels: np.ndarray,
    crossings: list,
    window_size: int = 5,
) -> None:
    """去高挂悬挂显示并播放铺平这张全由区分标记分别画号分定带牌带界区分各版这有区别标注数字地盘板块同时兼顾去给上边给铺上扣上套在这表叠给加上这外挂其展示出包含有查交交锋路段路口结出交结点位这一探测所用来侦察打这套方型这一观察游标其大框子在画的这一套外覆图层此交合混合出展呈现全展示动作景象表现显出这来大相映成之情之境。

    Args:
        labels: 具有长和这高同其具有这一套完全能以表现面此具有带分布着具有二维特征其排布着写出全副标记了号二维度阵基底这原标代码表二向二维图印阵式阵源版列组基原底层大料单原图结构大册列样。
        crossings: 收罗并装进打包好关于(核心坐标纵向定位 x 点所在位, 核心横跨它定位所处在 y, 作为表相盖过去这一道顶部一线, 被其身跨坐于其后这被埋底之这一在下一线)如此这类有关在碰上这一打交错其交叉这地口点等这全系其大打包交叉这些大信息这类包含有这组包并将其罗织做总名册列打包交收上来那组带有相交各方其内幕信息这一包元组包裹总数全打包其信息那集合大全集那列包这份单子目录集合薄。
        window_size: 其这大小就是被采用做用来在去开展找寻这交结点位在作为交叉这一碰面这探测扫雷作为这一其进行找点扫探测打框寻找这小操作观察游荡打观察那小方寸大小游标探窗这一窗口框尺寸该尺寸这长宽高之尺寸幅度号尺量宽限尺寸比例这设定其用上所占到用到这具体数值框窗那个框定此方形这扫窗口宽号框其尺寸设定尺宽幅框号大小尺寸之多之少之边距那数字那个限定限度量数了。
    """
    unique_labels = np.unique(labels)
    height, width = labels.shape
    color_map = {}

    for label in unique_labels:
        if label == unique_labels[-1]:
            color_map[label] = (0, 0, 0)  # 底下最下边打大背底衬那板这大其背景当背景全铺就给抹一色那就是黑
        else:
            color_map[label] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    for label in unique_labels:
        color_image[labels == label] = color_map[label]

    # 添加文本标注等文本加印这文本附记等这字迹上去
    for label in unique_labels:
        positions = np.argwhere(labels == label)
        if len(positions) > 0:
            y, x = positions[random.randint(0, len(positions) - 1)]
            if np.all(
                labels[
                    max(y - 1, 0) : min(y + 2, height),
                    max(x - 1, 0) : min(x + 2, width),
                ]
                == label
            ):
                cv2.putText(
                    color_image, str(label), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1,
                )

    # 描绘画那画出那被用来套定标用作那描画去套用于圈定那个去找寻探测碰出那交叉结点发生地的四四框出这套出那正方四方观察方这用框框用这交叉路去这用来框圈套这它交叉那探这交察扫框窗
    half_window = window_size // 2
    for crossing in crossings:
        x, y, _, _ = crossing
        top_left = (x - half_window, y - half_window)
        bottom_right = (x + half_window, y + half_window)
        cv2.rectangle(color_image, top_left, bottom_right, (0, 255, 0), 2)

    plt.imshow(color_image)
    plt.title("Crossings with Sliding Window Visualization")
    plt.show()
