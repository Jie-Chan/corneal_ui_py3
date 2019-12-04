import os
import imageio
import numpy as np

from operator import itemgetter  # operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为一些序号
from itertools import groupby  # groupby(iterable[, keyfunc])返回:按照keyfunc函数对序列每个元素执行后的结果分组(每个分组是一个迭代器), 返回这些分组的迭代器
from collections import Counter
# from scipy.misc import imsave
from scipy.signal import find_peaks_cwt
from scipy.cluster import vq
from scipy.optimize import fsolve

from configuration import conf, air_puff_force

root = os.path.dirname(os.path.realpath(__file__))
# imageio.plugins.ffmpeg.download()


def convert_to_frames(video_name):
    # print("进入分真")
    """Convert video to frames."""
    video_name = video_name.rstrip('.avi')  # 删除 string 字符串末尾的指定字符
    video_path = root + '/repo/{}.avi'.format(video_name)  # 格式转换
    print(video_path)
    base_name = video_path.split('/')[-1][:-4]  # 指定分隔符对字符串进行切片str.split(str="", num=string.count(str)).
    # print(base_name)
    frame_dir = root + '/static/cache/frame/{}'.format(base_name)
    # print(frame_dir)
    if os.path.exists(frame_dir):
        return sorted([os.path.join(frame_dir, p) for p in os.listdir(frame_dir)])
    else:
        os.makedirs(frame_dir)
        video_reader = imageio.get_reader(video_path, 'ffmpeg')
        frame_names = []
        for i, im in enumerate(video_reader, start=1):
            frame_name = '{}/{}_{:03d}.jpg'.format(frame_dir, base_name, i)
            imageio.imsave(frame_name, im, format='jpeg')
            frame_names.append(frame_name)
        return frame_names


# 判断视频已经过处理
def is_already_executed():
    # print("执行")
    """If all videos have been already executed, the inferred directory should be exist."""
    status = [os.path.exists(root + '/static/cache/infer/{}'.format(video_name[:-4])) for video_name in os.listdir(root + '/repo')]

    return all(status)


def isolate_bound(im):
    # print("进入isolate_bound")
    """Separate the upper and lower bound in a curve mask.
    Argument:
        im: (array) black-and-white image
    Returns:
        xs: (list) the sequential column index of input image
        y_up: (list) the row index which is relatively smaller
        y_lw: (list) the row index which is relatively larger
    """
    coords = sorted(list(zip(*np.where(im > 0)[::-1])))
    # print("coords",coords)
    d = {}
    for key, value in coords:
        d[key] = d.get(key, []) + [value]

    if len(coords) & 1:
        # something wrong with the mask, where y is not a unique mapping of x
        # say the two pixels in x-axis is identical
        bad = {key: d[key] for key in d if len(d[key]) > 2}
        for k, v in bad.items():
            d[k] = vq.kmeans(np.array(v, dtype=float), 2)[0].astype(int).tolist()

    xs, ys = [*d.keys()], [v for dv in d.values() for v in dv]
    # print("xsiso",xs,'\n',"ysiso",ys)
    y_up, y_lw = [ys[i::2] for i in [0, 1]]
    # print("isolatey_up",y_up)
    # print("isolatey_lw",y_lw)
    return xs, y_up, y_lw


def compute_curvature(primary_dicts, roi_deviation):
    print("即进入compute_curvature")
    """ Compute the curvature according to the specific area
    Argument:
        roi_deviation: (int) deviation from the middle of image that truly counted
    """
    def frame_curvature(primary_item, deviation):
        # print("进入每张图片画曲线")
        """
        Argument:
            primary_item: (dict) E.g., {'index': ..., gdssfhbs'thickness': ..., 'xs': ..., 'y_up': ..., 'y_lw': ...}
            deviation: (int) deviation from the middle of image that truly counted
        """
        # for i in primary_item:

        middle = conf.frame_width // 2
        x, y_up, y_lw = [primary_item.get(key) for key in ['xs', 'y_up', 'y_lw']]
        bulk = [z for z in zip(*[x, y_up, y_lw]) if middle - deviation <= z[0] <= middle + deviation]
        # print("y_up", y_up)
        x, y_up, _ = list(zip(*bulk))
        f_up = np.poly1d(np.polyfit(x, y_up, 2))
        # Derivative and compute the curvature in interested range, only care about the front surface
        return abs(np.polyval(np.polyder(f_up), x)).mean()

    return [round(frame_curvature(pd, roi_deviation), 3) for pd in primary_dicts]


def generate_scatter3d_curve(primary_dicts, curve_type='y_up'):
    print("产生3d曲线")
    """Show the curve in 3d scatter"""
    # print(primary_dicts)
    scatter3d_curve = []
    for i, pd in enumerate(primary_dicts):
        for j, pd_x in enumerate(pd['xs']):
            # print(i)
            # print(pd)
            # print(pd_x)
            scatter3d_curve.append([i, pd_x, -pd[curve_type][j]])
    # print("3d3d",scatter3d_curve)
    return scatter3d_curve

def get_applanation_time(curvatures, value=0):
    print("获得亚平时间")
    """Given the curvatures line is mirror symmetry in horizontal."""

    ar = np.array(curvatures)
    mid = len(ar) // 2
    lhs_index = (np.abs(ar[:mid] - value)).argmin()
    rhs_index = (np.abs(ar[mid:] - value)).argmin() + mid
    max_index = ar[lhs_index: rhs_index].argmax() + lhs_index
    # print("ls_index",lhs_index)
    # print("rhs_index", rhs_index)
    # print("max_index", max_index)
    return lhs_index, rhs_index, max_index


def get_2peak(primary_dicts):
    print("get2peak")
    peak_lhs0_vertex = []
    peak_rhs0_vertex = []
    for i in range(len(primary_dicts)):
        x0, y0 = [np.array(primary_dicts[i][key]) for key in ['xs', 'y_up']]
        # print(x0)
        peak_index_0 = find_peaks_cwt(y0[100:286], np.arange(1, 20))
        peak_pair_0 = sorted(list(zip(y0[peak_index_0], peak_index_0 + 100 - 1)))
        # print("ccc", peak_pair_0)
        # (y, x)默认升序
           # print('peak_pair_0',peak_pair_0)
        # Middle lowest vertex with max y-index
        peak_y0_1, peak_x0_1 = peak_pair_0[0]
        peak_lhs0_vertex.append(peak_pair_0[0])
        # Compute the peak distance
        peak_y0_2, peak_x0_2 = 0, 0
        x0distance = []
        for p in peak_pair_0[1:]:

            peak_y0_2, peak_x0_2 = p
            # print(peak_x0_2)
            x0_distance = abs(peak_x0_2 - peak_x0_1)
            y0_distance = abs(peak_y0_2 - peak_y0_1)
            # print("x_distance",y0_distance)
            if x0_distance > 350 and y0_distance > 30:
                break
            x0distance.append(x0_distance)
        # x0_distance=max(x0distance)
        # print("x0distance",x0_distance)
        peak_rhs00_vertex = peak_y0_2, peak_x0_2
        # print(peak_rhs00_vertex)
        peak_rhs000_vertex = peak_rhs00_vertex
        peak_rhs0_vertex.append(peak_rhs000_vertex)
    return peak_lhs0_vertex, peak_rhs0_vertex


def get_applanation_length(primary_item):
    print("亚平长度")
    x, y = [np.array(primary_item[key]) for key in ['xs', 'y_up']]
    # print(y)# upper
    y_most = list(map(itemgetter(0), Counter(y).most_common(2)))
    # print("y_most",y_most)
    y_flat = np.mean(y_most)
    # print("y_flat",y_flat)
    x_most = x[np.where((y >= min(y_most)) & (y <= max(y_most)))]
    # print("x_most1", x_most)
    # print("x_most1",sorted(x_most)[:5])
    x_start = np.mean(sorted(x_most)[:5])
    # print("x_start2",x_start)
    x_end = np.mean(sorted(x_most)[-5:])

    return abs(x_end - x_start), y_flat


def get_peak_vertex(primary_dicts, peak_index):
    print("峰值")
    x, y = [np.array(primary_dicts[peak_index][key]) for key in ['xs', 'y_up']]
    peak_index = find_peaks_cwt(y[100:470], np.arange(1, 14))
    peak_pair = sorted(list(zip(y[peak_index], peak_index + 100 - 1)))  # (y, x)

    # Middle lowest vertex with max y-index
    peak_y_1, peak_x_1 = peak_pair[0]
    # print('peak_x_1',peak_x_1)

    peak_y_m, peak_x_m = peak_pair[-1]
    # print('peak_x_m',peak_x_m)

    # Compute the peak distance
    peak_y_2, peak_x_2 = 0, 0
    for p in peak_pair[1:]:
        peak_y_2, peak_x_2 = p
        x_distance = abs(peak_x_2 - peak_x_1)
        y_distance = abs(peak_y_2 - peak_y_1)
        # print("x_distance",x_distance)
        if x_distance > 200 and y_distance < 5:
            break
    # Make sure the left-hand-side is the one with smaller x-index
    peak_mid_vertex = peak_x_m, peak_y_m
    peak_lhs_vertex, peak_rhs_vertex = sorted([[peak_x_1, peak_y_1], [peak_x_2, peak_y_2]])
    # print('peak_lhs_vertex', peak_lhs_vertex)
    # print('peak_rhs_vertex', peak_rhs_vertex)
    # print('peak_mid_vertex', peak_mid_vertex)
    return peak_lhs_vertex, peak_mid_vertex, peak_rhs_vertex


# 顶点的位移
def get_peak_deviation(primary_dicts, peak_x, roi_width=2):
    print("峰值dev")
    """roi_width: interested width around the peak is 2"""
    peak_ys = np.array([np.mean(primary_dicts[i]['y_up'][(peak_x - roi_width): (peak_x + roi_width)])
                        for i in range(len(primary_dicts))])
    # 水平方向0.0156mm/像素点，竖直方向0.0165/像素点

    # corvis弧长左右3.5mm，3.5/0.0156=
    # print("len dicts", len(primary_dicts))
    # print("dicts_yuplen", type(primary_dicts[0]['y_up']))
    # # print("dicts_yup",primary_dicts[0]['y_up'][512-3:512+3])
    # # print("y_up",primary_dicts['y_up'])
    # print("lendicts", len(primary_dicts))
    #
    # print("peak_x", peak_x)
    # # print("roi_width",roi_width)
    # print("peak_ys", peak_ys)
    # deviation
    return peak_ys - peak_ys[0]


class BioParams:
    """Compute the common biological parameters about corneal limbus"""

    def __init__(self, checked_video_name):
        # print("视频从名称")
        npy_path = root + '/static/cache/infer/primary_results_{}.npy'.format(checked_video_name)
        # self.primary_dicts = np.load('./static/cache/infer/primary_results_{}.npy'.format(checked_video_name))
        self.primary_dicts = np.load(npy_path, allow_pickle=True)
        self.video_length = len(self.primary_dicts)
        self.curvatures = compute_curvature(self.primary_dicts, roi_deviation=100)
        self.flat_index_1, self.flat_index_2, self.peak_index = get_applanation_time(self.curvatures)
        # print("flat1",self.flat_index_1,"\n","flat2",self.flat_index_2)
        self.flat_length_1, self.flat_y_1 = get_applanation_length(self.primary_dicts[self.flat_index_1])
        # print("flat1length",self.flat_length_1,"\n",self.flat_y_1)
        self.flat_length_2, self.flat_y_2 = get_applanation_length(self.primary_dicts[self.flat_index_2])
        # print("flat2length", self.flat_length_2, "\n", self.flat_y_2)
        self.peak_lhs_vertex, self.peak_mid_vertex, self.peak_rhs_vertex = get_peak_vertex(self.primary_dicts,
                                                                                           self.peak_index)
        self.peak_lhs0_vertex, self.peak_rhs0_vertex = get_2peak(self.primary_dicts)
        self.peak_deviation = get_peak_deviation(self.primary_dicts, self.peak_mid_vertex[0])
        # print("peak_deviation", self.peak_deviation)

    # add by tree
    def arc_length(self):
        # print("arcarc")
        arc_x0 = np.array(self.peak_lhs0_vertex)[:, 1]
        # print("arc_x0", arc_x0)
        arc_x1 = np.array(self.peak_rhs0_vertex)[:, 1]

        # arc_x1 = peak_x
        # print("arc_x1", arc_x1)
        # print("arc_x1",xx)
        d_arc = abs(arc_x0 - 288 + 1) *2* 0.0156
        d_arc = d_arc - d_arc[0]
        # for i in range(len(d_arc)):
        #     if d_arc[i] < d_arc[0]:
        #         d_arc[i] = d_arc[0]
        # print(d_arc)
        return d_arc

    def get_apex(self):
        # print("get_apex")
        peak_ys = self.peak_deviation

        # print("peak_ys", peak_ys)
        apex = peak_ys - peak_ys[0]
        return apex

    #

    def get_applanation_velocity(self, interval_frames=10):
        # print("亚平速度")
        """Compute the velocity from initial peak to flat status by the interval
        Argument:
            interval_frames: the frame interval to get consumed time, timed by frame-time-ratio (0.231)
        """
        first = self.flat_index_1 - interval_frames
        second = self.flat_index_2 + interval_frames
        first_y_peak = np.sort(self.primary_dicts[first].get('y_up'))[:5].mean()
        second_y_peak = np.sort(self.primary_dicts[second].get('y_up'))[:5].mean()

        sec = interval_frames * conf.frame_time_ratio
        v_in = abs(self.flat_y_1 - first_y_peak) * conf.y_scale / sec
        v_out = abs(self.flat_y_2 - second_y_peak) * conf.y_scale / sec
        # print("vin=",v_in,"vout=",v_out)
        return v_in, v_out

    def get_deformation_amplitude(self):
        y_init = np.sort(self.primary_dicts[0].get('y_up'))[:5].mean()
        y_concave = np.sort(self.primary_dicts[self.peak_index].get('y_up'))[-5:].mean()
        # print("abs(y_init - y_concave)",abs(y_init - y_concave))
        return abs(y_init - y_concave)

    def get_curvature_radius(self):
        """Compute the highest concavity radius"""

        def circle_equation(c, points):
            D, E, F = c[0], c[1], c[2]
            return [x ** 2 + y ** 2 + D * x + E * y + F for x, y in points]

        radius = 0
        if any(self.peak_rhs_vertex):  # success to find another peak
            vertexs = np.array([self.peak_lhs_vertex, self.peak_rhs_vertex, self.peak_mid_vertex]).reshape([3, 2])
            points = vertexs * [conf.x_scale, conf.y_scale]
            # points = [[peak_x_1 * conf.x_scale, peak_y_1 * conf.y_scale],
            #           [peak_x_2 * conf.x_scale, peak_y_2 * conf.y_scale],
            #           [peak_x_m * conf.x_scale, peak_y_m * conf.y_scale]]
            D, E, F = fsolve(circle_equation, [0, 0, 0], args=(points))
            radius = np.sqrt(D ** 2 + E ** 2 - 4 * F) / 2
            # centroid = [-D / 2, -E / 2]
            # print("radius=", radius)
        return radius

    def get_inward_velocity(self):
        """Return the inward velocity between the 33th and the 42th frame."""
        peak_y_33 = self.peak_deviation[33 - 1]
        peak_y_42 = self.peak_deviation[42 - 1]
        # print("py33",peak_y_33)
        # print("py42",peak_y_42)
        v_inward = abs(peak_y_42 - peak_y_33) * conf.y_scale / ((42 - 33) * 0.231)  # frame rate

        # print( "vinward_inward",v_inward)
        return v_inward

    def get_corneal_creep_rate(self):
        print("here")
        trend = [b - a for a, b in zip(self.peak_deviation[::1], self.peak_deviation[1::1])]
        # print("trend=",trend)
        index = 1
        for neg_k, tr in groupby(trend, lambda k: k < - 0.0001):
            # print("很好")
            tr_list = list(tr)
            # print(tr_list)
            index += len(tr_list)
            if neg_k and np.mean(tr_list) < - 1:
                break

        xs = range(index, self.video_length)
        # print("xs=",xs)
        # print(xs)
        ys = self.peak_deviation[xs]
        # print("ys=",ys)
        linear_func = np.polyfit(xs, ys, deg=1)
        return linear_func[0]

    def get_corneal_contour_deformation(self):
        y_init = np.array(self.primary_dicts[0]['y_up'])[:20].mean()
        y_peak = np.array(self.primary_dicts[self.peak_index]['y_up'])[:20].mean()
        return abs(y_init - y_peak)

    def get_max_deformation_area(self):
        x_peak, y_peak = [np.array(self.primary_dicts[self.peak_index][key]) for key in ['xs', 'y_up']]
        y_init = np.array(self.primary_dicts[0]['y_up'])
        index = np.where((x_peak > self.peak_lhs_vertex[0]) & (x_peak < self.peak_rhs_vertex[0]))
        sum_area = abs(y_init[index] - y_peak[index]).sum()
        return sum_area

    # add by tree
    # def get_arc_length(self):

    def get_max_deformation_time(self):
        trend = [b - a for a, b in zip(self.curvatures[::1], self.curvatures[1::1])]
        index = 0
        for neg_k, tr in groupby(trend, lambda k: k < -0.0001):
            tr_list = list(tr)
            index += len(tr_list)
            if neg_k and np.mean(tr_list) < -0.003:
                break
        return self.peak_index - index

    def get_energy_absorbed_area_and_k(self):
        """Relationship: air_puff_force <--> peak_displacement
        Compute k by displacement between [0.2, 0.5]
        """
        epsilon = 0.05

        rel = np.array(list((zip(self.peak_deviation * conf.y_scale, air_puff_force))))
        # print("rel", rel[:, 0])
        #
        # lim = 0.3
        #
        # test_epsilon = epsilon
        test_rel = abs(rel[:, 0])
        #
        test_rel_max=max(test_rel)
        test_rel_lw=round(0.1*test_rel_max,2)
        test_rel_up=round(0.85*test_rel_max,2)
        # print("relmax",test_rel)
        # print("rellw", test_rel_lw)
        # print("relup", test_rel_up)
        if test_rel_max <0.5:
            start_i, end_i = [np.where(abs(rel[:, 0] - lim) < epsilon)[0][0] for lim in [test_rel_lw, test_rel_up]]
        # # aaa = np.where(abs(rel[:, 0] - lim) < epsilon)
        # # print("11111111111111111111111111111111", test_epsilon, '\n', test_rel, '\n', "aaa", aaa)
        else:
            start_i, end_i = [np.where(abs(rel[:, 0] - lim) < epsilon)[0][0] for lim in [0.2, 0.5]]  # 没有0.5
        # else:
        # start_i, end_i = [np.where(abs(rel[:, 0] - lim) < epsilon)[0][0] for lim in [test_rel_lw, test_rel_up]]  # 没有0.5
        # print("start_i", start_i, "end_i", end_i)

        x_displacement, y_puff = rel[end_i] - rel[start_i]

        k = y_puff / x_displacement

        max_displacement_index = rel.argmax(axis=0)[0]
        onload_area = rel[:max_displacement_index, 1].sum()
        unload_area = rel[max_displacement_index:, 1].sum()
        energy_absorbed_area = onload_area - unload_area
        return energy_absorbed_area, k
