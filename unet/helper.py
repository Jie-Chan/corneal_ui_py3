import numpy as np
import skimage
# from mlab.releases import R2014a
from skimage import color, filters
from scipy.signal import savgol_filter
from skimage.feature import corner_harris, corner_peaks
from skimage.filters import threshold_isodata
from skimage.morphology import remove_small_objects, convex_hull_image, \
    binary_closing, binary_opening, disk, rectangle
from skimage.morphology import label as morphology_abel
from skimage.segmentation import find_boundaries
from skimage import measure, color
# from scipy.misc import imread, imsave, imresize
import cv2 as cv

from utils import isolate_bound


def blend(image, label, coords=[None] * 3, alpha=0.4):
    print("进入blend")
    """"Simulate colormap `jet`."""
    image, label = [item.astype(float) for item in [image, label]]
    # print("image",image)
    r = label * alpha + 20
    b = (image + image.mean()) * (1 + alpha)
    g = np.minimum(r, b)
    rgb = np.dstack([r, g, b] + image * 0.2)#r*0.299+g*0.587+b*0.114
    if coords[0] is not None:
        # curve_mask = curve_mask[..., None]  # for broadcast
        # rgb += curve_mask * 0.5
        xs, y_up_fit, y_lw_fit = coords
        xs=xs[1:576]
        y_up_fit=y_up_fit[1:200]
        y_lw_fit=y_lw_fit[1:200]
        # print("yuop",y_up_fit)

        if len(xs) == len(y_up_fit) == len(y_lw_fit):
            curve_mask = np.zeros_like(label)
            # print("curve_mask", curve_mask)
            # print("y_lw_fit", len(y_lw_fit))
            curve_mask[y_up_fit.astype(int), xs] = 255
            curve_mask[y_lw_fit.astype(int), xs] = 255
            rgb[..., 1] += curve_mask * 0.5  # add to blue channel
    # vis.image(rgb.transpose(2, 0, 1))
    return rgb.astype(np.uint8)


def remove_watermark(frame, template_bw, surround_bw):
    # print("进入移除水印")
    """Remove watermark by fill surrounding intensity."""
    # Gray value around template
    surround_intensity = frame[surround_bw].mean()
    # Subtract
    demark = np.select([~template_bw], [frame], default=surround_intensity)
    # print("demark", demark)
    return demark


def traditional_seg(im):
    print("传统分割标记")
    tsh = threshold_isodata(im, return_all=False)
    bw = im > tsh
    bw = binary_closing(bw, selem=disk(5))
    bw = binary_opening(bw, selem=disk(5))
    bw = remove_small_objects(bw, min_size=1024)
    # print("bw",bw)
    return bw.astype(int)


def post_process(mask):
    """Mainly remove the convex areas"""

    bw = morphology_abel(mask == 1)
    # thresh = filters.threshold_li(bw)
    # bw = (bw >= 0.5) * 1
    # 0) fill gap
    bw = binary_closing(bw, disk(2))
    # 1) detach mislabeled pixels
    bw = binary_opening(bw, rectangle(2, 20))
    # 2) remove small objects
    con_labels = measure.label(bw, connectivity=2)  # 8联通区域标记
    # dst = color.label2rgb(con_labels)
    # imsave("static/{}.jpg".format({}), con_labels)
    cv.imwrite("static/{}.jpg".format({}), con_labels)

    num = con_labels.max() + 1
    # print("con_labels_num", num)
    # bb = measure.regionprops(con_labels)

    region_area = [region.area for region in measure.regionprops(con_labels)]
    print("con_labels_area", len(region_area))
    print("region_area", region_area)
    region_area_sort = np.sort(region_area)
    if len(region_area) == 1:
        # 2) remove small objects
        bw = remove_small_objects(con_labels, min_size=5120, connectivity=2)

    # elif region_area[-1] > 12000:
    #
    #     # region_area_max = region_area_sort[-1]
    #     # region_area_sum = sum(region_area)
    #     region_area_min = region_area_sort[-3]
    #     # 2) remove small objects
    #     bw = remove_small_objects(con_labels, min_size=region_area_min , connectivity=2)
    else:
    # region_area_max = region_area_sort[-1]
    # region_area_sum = sum(region_area)
        region_area_min = region_area_sort[-2]
        # 2) remove small objects
        bw = remove_small_objects(con_labels, min_size=region_area_min + 10, connectivity=2)

    # else:
    #     # region_area_max = region_area_sort[-1]
    #     # region_area_sum = sum(region_area)
    #     region_area_min = region_area_sort[-2]
    #     # 2) remove small objects
    #     bw = remove_small_objects(con_labels, min_size=region_area_min + 10, connectivity=2)
    # 3) solve the defeat, typically the convex outline
    coords = corner_peaks(corner_harris(bw, k=0.2), min_distance=5)
    # print("coords", coords)
    valid = [c for c in coords if 100 < c[1] < 476]  # only cares about this valid range
    # print("valid", valid)
    if valid:
        y, x = zip(*valid)
        # corners appear in pair
        if len(y) % 2 == 0:
            # select the lowest pair
            left_x, right_x = [func(x[0], x[1]) for func in (min, max)]
            sep_x = np.arange(left_x, right_x + 1).astype(int)
            sep_y = np.floor(np.linspace(y[0], y[1] + 1, len(sep_x))).astype(int)
            # make the gap manually
            bw[sep_y, sep_x] = 0
            bw = binary_opening(bw, disk(6))
        else:
            mask = np.zeros_like(bw)
            mask[y, x] = 1
            chull = convex_hull_image(mask)
            bw = np.logical_xor(chull, bw)
            bw = binary_opening(bw, disk(6))
    return bw


#
# # import matplotlib.pyplot as plt
# def post_process(mask):
#     print("进入post_progress")
#     """Mainly remove the convex areas"""
#     bw = morphology_abel(mask == 1) * 1
#
#     # print("bw_postprogress", np.sum(bw))
#     # 0) fill gap
#     bw = binary_closing(bw, disk(3))
#     # print("bw_bw_1", np.sum(bw))
#     # 1) detach mislabeled pixels
#     bw = binary_opening(bw, rectangle(2, 20))
#     # print("bw_bw_2", np.shape(bw))
#     con_labels = measure.label(bw, connectivity=2)  # 8联通区域标记
#     # dst = color.label2rgb(con_labels)
#     imsave("static/{}.jpg".format({}), con_labels)
#     num=con_labels.max() + 1
#     print("con_labels_num", num )
#     # bb = measure.regionprops(con_labels)
#
#     region_area = [region.area for region in measure.regionprops(con_labels)]
#     print("con_labels_area", len(region_area))
#     print("region_area", region_area)
#     region_area_sort = np.sort(region_area)
#     if len(region_area) == 1:
#     # 2) remove small objects
#         bw = remove_small_objects(con_labels, min_size=5120, connectivity=2)
#     else:
#         region_area_max = region_area_sort[-1]
#         region_area_sum = sum(region_area)
#         region_area_min = region_area_sort[-2]
#
#         # 2) remove small objects
#         bw = remove_small_objects(con_labels, min_size=region_area_min + 10, connectivity=2)
#     chull = convex_hull_image(mask)
#     bw = np.logical_xor(chull, bw)
#     bw = binary_opening(bw, disk(6))
#     return bw
#
# # print('region_area',region_area_sort)
# region_area_max = region_area_sort[-1]
# region_area_sum = sum(region_area)
# region_area_min = region_area_sort[-2]
# # print("sum",region_area_sum)
# # print("region_area_min", region_area_min)
# if region_area_min > 4096:
#     bw = remove_small_objects(bw, min_size=region_area_min + 10, connectivity=2)
# else:
#     bw = remove_small_objects(bw, min_size=4096, connectivity=2)


# if num == 1 or len(region_area) == 1:
# coords = corner_peaks(corner_harris(bw, k=0.2), min_distance=5)
# valid = [c for c in coords if 100 < c[1] < 476]  # only cares about this valid range
# if valid:
#     y, x = zip(*valid)
#     mask = np.zeros_like(bw)
#     mask[y, x] = 1
#     chull = convex_hull_image(mask)
#     bw = np.logical_xor(chull, bw)
#     bw = binary_opening(bw, disk(6))

# # 3) solve the defeat, typically the convex outline
# coords = corner_peaks(corner_harris(bw, k=0.2), min_distance=5)
# valid = [c for c in coords if 100 < c[1] < 476]  # only cares about this valid range
# if valid:
#     y, x = zip(*valid)
#     # corners appear in pair
#     if len(y) % 2 == 0:
#         # select the lowest pair
#         left_x, right_x = [func(x[0], x[1]) for func in (min, max)]
#         sep_x = np.arange(left_x, right_x + 1).astype(int)
#         sep_y = np.floor(np.linspace(y[0], y[1] + 1, len(sep_x))).astype(int)
#         # make the gap manually
#         bw[sep_y, sep_x] = 0
#         bw = binary_opening(bw, disk(6))
#         np.save("post_progress_bw.npy", bw)
#     else:
#         mask = np.zeros_like(bw)
#         mask[y, x] = 1
#         chull = convex_hull_image(mask)
#         bw = np.logical_xor(chull, bw)
#         bw = binary_opening(bw, disk(6))
#         np.save("post_progress_bw.npy", bw)
#     return bw
# else:
#     region_area_sort = np.sort(region_area)
#     # print('region_area',region_area_sort)
#     region_area_max = region_area_sort[-1]
#     region_area_sum=sum(region_area)
#     region_area_min = region_area_sort[-2]
#     # print("sum",region_area_sum)
#     # if region_area_max >= 8500 and region_area_max <= 13000:
#     if abs(region_area_min-region_area_max) < 300:
#         bw = np.load("post_progress_bw.npy")
#         return bw
#     else:
#         if region_area_max >= 8500:
#         # 2) remove small objects
#
#             # print("region_area_min", region_area_min)
#             if region_area_min > 4096:
#                 bw = remove_small_objects(bw, min_size=region_area_min + 10, connectivity=2)
#             else:
#                 bw = remove_small_objects(bw, min_size=4096, connectivity=2)
#
#             # print(np.sum(bw2))
#             # 3) solve the defeat, typically the convex outline
#             coords = corner_peaks(corner_harris(bw, k=0.2), min_distance=1)  # k越小检测越锋利的点
#             # print("coords", coords[0])
#             valid = [c for c in coords if 1 < c[0] < 576]  # only cares about this valid range
#             # valid = [c for c in coords if 1 < c[1] < 576]  # only cares about this valid range
#             if valid:
#                 # print("vavaa")
#                 y, x = zip(*valid)
#                 # corners appear in pair
#                 if len(y) % 2 == 0:
#                     # select the lowest pair
#                     left_x, right_x = [func(x[0], x[1]) for func in (min, max)]
#                     sep_x = np.arange(left_x, right_x + 1).astype(int)
#                     sep_y = np.floor(np.linspace(y[0], y[1] + 1, len(sep_x))).astype(int)
#                     # make the gap manually
#                     bw[sep_y, sep_x] = 0
#                     bw = binary_opening(bw, disk(6))
#                     np.save("post_progress_bw.npy", bw)
#                     return bw
#                 else:
#                     mask = np.zeros_like(bw)
#                     mask[y, x] = 1
#                     chull = convex_hull_image(mask)  # 凸包是指一个凸多边形，这个凸多边形将图片中所有的白色像素点都包含在内。
#                     bw = np.logical_xor(chull, bw)
#                     bw = binary_opening(bw, disk(6))
#                     # print("lastbw",last_bw)
#                     np.save("post_progress_bw.npy", bw)
#                     return bw
#         else:
#             bw = np.load("post_progress_bw.npy")
#             return bw


def fitting_curve(mask, margin=(60, 60)):
    print("进入fitting_curve")
    """Compute thickness by fitting the curve
    Argument:
        margin: indicate valid mask region in case overfit
    Return:
        thickness: between upper and lower limbus
        (xs, y_up_fit, y_lw_fit): coordinates of corresponding results
    """
    # 1. Find boundary
    bound = find_boundaries(mask > 127, mode='outer')
    # np.set_printoptions(threshold=1e7)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(200,576))
    # plt.imshow(bound)
    # plt.show()
    # 2. Crop marginal parts (may be noise)
    # print('mask',mask)
    lhs, rhs = margin
    bound[:, :lhs] = 0  # left hand side
    bound[:, -rhs:] = 0  # right hand side
    # print('boundval',np.sum(bound=='false'))
    # pdb.set_trace()

    # 3. Process upper and lower boundary respectively
    xs, y_up, y_lw = isolate_bound(bound)
    # 1) fit poly
    f_up, f_lw = [np.poly1d(np.polyfit(xs, ys, 6)) for ys in [y_up, y_lw]]
    # 2) interpolation
    rw, width = 30, mask.shape[1]  # roi width
    y_up_fit, y_lw_fit = [f(xs) for f in [f_up, f_lw]]
    thickness = (y_up_fit - y_lw_fit)[width // 2 - rw: width // 2 + rw]

    return abs(thickness.mean()), (xs, y_up_fit, y_lw_fit)


def interp1d_curve(mask, margin=(60, 60)):
    print("进入interpld_curve")
    """Compute thickness by interpolation
    Argument:
        margin: indicate valid mask region in case overfit

    Return:
        thickness: between upper and lower limbus
        (xs, y_up_interp, y_lw_interp): coordinates of corresponding results
    """
    # print("mask111", max(list(mask)))

    # 1. Find boundary
    # import cv2
    bound = find_boundaries(mask > 126, mode='outer')
    # print("mask222", mask)
    # np.set_printoptions(threshold='nan')
    bound1 = 1 * bound
    # print('bound1', np.sum(bound1))
    # print("boundshape", bound.shape)
    # print("boundval", bound)

    # 2. Crop marginal parts (may be noise)
    lhs, rhs = margin
    bound[:, :lhs] = 0  # left hand side
    bound[:, -rhs:] = 0  # right hand side
    # print("bound2", bound)
    # 3. Process upper and lower boundary respectively
    # xs=456
    xs, y_up, y_lw = isolate_bound(bound)
    # print("boundvvv", xs)
    # print("yyyup:", np.shape(y_up))
    # print("yyylw:", np.shape(y_lw))
    len_yup = len(y_up)
    len_ylw = len(y_lw)
    len1 = min(len_ylw, len_yup)
    xs = xs[:len1]
    y_up = y_up[:len1]
    y_lw = y_lw[:len1]


    # pdb.set_trace()#要打印的地方（打印前）输入代码,pdb.set_trace（）就是把程序暂停在那里，手动断点只有在Debug时才有用，run时是没用的
    # try:
    # 1) interp1d
    y_up_interp, y_lw_interp = [savgol_filter(y, 7, 2) for y in [y_up, y_lw]]
    y_up_interp[y_up_interp > 199] = 199
    y_up_interp[y_up_interp < 0] = 0
    y_lw_interp[y_lw_interp > 199] = 199
    y_lw_interp[y_lw_interp < 0] = 0
    # print(y_up_interp)
    # print(y_lw_interp)
    # global y_up_interp, y_lw_interp
    # for y in [y_up, y_lw]:
    #     if y_up is None:
    #        # y_up_interp, y_lw_interp = savgol_filter(y[i - 2], 7, 2)
    #         pass
    #     else:
    #         y_up_interp, y_lw_interp = savgol_filter(y, 7, 2)
    # print("interp_upyyy",y_up_interp,'\n',"interp_lwyyy",y_lw_interp)
    # except :
    #     pass
    # else:
    # 2) get thickness
    rw, width = 30, mask.shape[1]  # roi width
    thickness = (y_up_interp - y_lw_interp)[width // 2 - rw: width // 2 + rw]
    return abs(thickness.mean()), (xs, y_up_interp, y_lw_interp)
