import os
from os import path
import sys
import torch
import imageio
import warnings
import numpy as np

from glob import glob
from tqdm import tqdm
from torch.autograd import Variable
# from scipy.misc import imread, imsave, imresize
import cv2 as cv
from flask import Flask, flash, redirect, render_template, request


from utils import convert_to_frames
from unet.model import UNetVanilla
from unet.helper import remove_watermark, post_process, traditional_seg, blend, interp1d_curve, fitting_curve
import io
import sys

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

# assert torch.cuda.is_available(), 'Error: CUDA not found!'

# Init
# ===========================================================================
root = os.path.dirname(os.path.realpath(__file__))
model = UNetVanilla()
weights = torch.load(root + '/unet/weights/UNetVanilla_best.pth')
model.load_state_dict(weights)
model.eval().cuda()
# template: binary image of watermark
# surround: dilated template for later computation of surrounding intensity
template_bw, surround_bw = [np.load(root + '/unet/weights/{}.npy'.format(f))
                            for f in ['template_bw', 'surround_bw']]
# ===========================================================================

def do_deploy():
    # flash("hello world")
    thickness_store = []
    para_store = []
    for video_name in os.listdir(root + '/repo'):
        video_name = video_name[:-4]
        # print('===> Processing video 【{}.avi】...'.format(video_name))
        flash('===> Processing video 【{}.avi】...'.format(video_name))
        frame_paths = sorted(glob(root + '/static/cache/frame/{}/*.jpg'.format(video_name)))
        primary_results = []
        for i, p in tqdm(enumerate(frame_paths, start=1), file=sys.stdout,
                         total=len(frame_paths), unit=' frames', dynamic_ncols=True):
            # im = remove_watermark(cv.imread(p, mode='L'), template_bw, surround_bw)
            im = remove_watermark(cv.imread(p,flags=cv.IMREAD_GRAYSCALE), template_bw, surround_bw)
            # x = Variable(torch.from_numpy(im[None, None, ...]), volatile=True).float().cuda()
            with torch.no_grad():
                x = Variable(torch.from_numpy(im[None, None, ...])).float().cuda()
            # if image is the first frame of a video
            # then get the segmentation by traditional threshold method
            if i == 1:
                bw = traditional_seg(im)
                # y_prev = Variable(torch.from_numpy(bw[None, None, ...]), volatile=True).float().cuda()
                with torch.no_grad():
                    y_prev = Variable(torch.from_numpy(bw[None, None, ...])).float().cuda()
            output = model(x, y_prev)
            # y_prev_prev = y_prev
            y = output.data.max(1)[1]
            # y_prev = Variable(y[None, ...], volatile=True).float()  # as next frame's input
            with torch.no_grad():
                y_prev = Variable(y[None, ...]).float()
            y = y[0].cpu().numpy()
            # imsave("static/ori/{}.jpg".format(i),y)
            cv.imwrite(root + "/static/ori/{}.jpg".format(i),y)
            # post-process
            with warnings.catch_warnings():
                # ignore the warning about remove_small_objects
                warnings.simplefilter("ignore")
                # if y is not None:
                display = post_process(y) * 255
            try:
                # thickness, (xs, y_up, y_lw) = fitting_curve(display)
                thickness, (xs, y_up, y_lw) = interp1d_curve(display)
                if y_up is None:
                    thickness, (xs, y_up, y_lw) = thickness_store[i - 2], para_store[i - 2]

            except (IndexError, TypeError):
                thickness, (xs, y_up, y_lw) = 0, [None] * 3
                tqdm.write('Oops, fail to detect {}th frame...'.format(i))

            thickness_store.append(thickness)
            # 1
            para_store.append((xs, y_up, y_lw))
            # Append and save results
            primary_results.append(
                {'index': i - 1, 'thick': thickness, 'xs': xs, 'y_up': y_up, 'y_lw': y_lw})
            # Save and Display
            deploy_dir = [root + '/static/cache/infer/{}/{}'.format(video_name, subdir) for subdir in ['bw', 'blend']]
            [os.makedirs(dd) for dd in deploy_dir if not os.path.exists(dd)]

            # imsave(deploy_dir[0] + '/{:03d}.jpg'.format(i), display)
            # imsave(deploy_dir[1] + '/{:03d}.jpg'.format(i), blend(im, display, [xs, y_up, y_lw]))
            cv.imwrite(deploy_dir[0] + '/{:03d}.jpg'.format(i), display)
            cv.imwrite(deploy_dir[1] + '/{:03d}.jpg'.format(i), blend(im, display, [xs, y_up, y_lw]))
        np.save(root + '/static/cache/infer/primary_results_{}.npy'.format(video_name), primary_results)


def generate_video():
    print("进入产生视频")
    for video_name in os.listdir(root + '/repo'):
        video_name = video_name[:-4]
        # print('===> Generating inferred video 【{}.mp4】'.format(video_name))
        # Generate inferred video and make original frames into video in mp4 format
        with imageio.get_writer(root + '/static/cache/infer/blend_{}.mp4'.format(video_name), mode='I') as writer:
            for im_path in sorted(glob(root + '/static/cache/infer/{}/blend/*.jpg'.format(video_name))):
                # image = imresize(cv.imread(im_path), (208, 576))  # resize for video compatibility
                image = cv.resize(cv.imread(im_path), (576, 208))
                writer.append_data(image)
        with imageio.get_writer(root + '/static/cache/infer/original_{}.mp4'.format(video_name), mode='I') as writer:
            for im_path in sorted(glob(root + '/static/cache/frame/{}/*.jpg'.format(video_name))):
                # image = imresize(cv.imread(im_path), (208, 576))  # resize for video compatibility
                image = cv.resize(cv.imread(im_path), (576, 208))
                writer.append_data(image)


if __name__ == '__main__':
    torch.cuda.set_device(0)
    # 1. Generate all frames of the video in the repository
    [convert_to_frames(video_name) for video_name in os.listdir(root + 'repo')]
    # print("world")
    # 2. Do the deploy
    do_deploy()
    # 3. Output the video for visualization
    generate_video()
