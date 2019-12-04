import os
import glob
import pdb
from scipy.misc import imread, imsave
os.chdir('/media/szu/HELLOTREE/U12_data_analysis/20180321_VIDEO_noremark/Vedio/')
Vedio_name_list=[]
for vedio_name in glob.glob("*.avi"):
    Vedio_name_list.append(vedio_name[:-4])

os.chdir('/media/szu/HELLOTREE/U12_data_analysis/20180321_VIDEO_noremark/output/output_Vedio_Frame1/')
frame_name_list=[]
for frame_name in glob.glob("*.png"):
    frame_name_list.append(frame_name[:-4])

for i in Vedio_name_list:
    for j in frame_name_list:
        framename=j[:-6]
        if str(i) == str(framename):
            os.chdir('/media/szu/HELLOTREE/U12_data_analysis/20180321_VIDEO_noremark/output/output_Vedio_Frame1/')
            img=imread(j)
            output_dir = ['/media/szu/HELLOTREE/U12_data_analysis/20180321_VIDEO_noremark/output/out_frame/{}'.format(i)]
            [os.makedirs(dd) for dd in Vedio_name_list if not os.path.exists(dd)]
            imsave('/media/szu/HELLOTREE/U12_data_analysis/20180321_VIDEO_noremark/output/out_frame/{}/{}'.format(i,j),img)
            pdb.set_trace()
        # pdb.set_trace()
# path = '/media/szu/HELLOTREE/U12_data_analysis/20180321_VIDEO_noremark/Vedio/'
# vedios = glob.glob(path + '/*')
# vedios=vedios.T
# for vedio in vedios:
#     video_name = vedios[:-4]
#     frame_paths = sorted(glob('static/cache/frame/{}/*.png'.format(video_name)))
#     na = vedio.split('/')[-1]
#     os.rename()

