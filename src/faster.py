from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import cv2
from motrackers import CentroidKF_Tracker, SORT
from motrackers.utils import draw_tracks
import os

from lib.detectors.car_pose import CarPoseDetector
from lib.opts import opts
import shutil
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['net', 'dec']

video_file = os.path.join(os.getcwd(),'./data/video/video.mp4')

tracker = 'SORT' # Input tracker: CentroidKF_Tracker or SORT (as string type)

if tracker == 'CentroidKF_Tracker':
    tracker = CentroidKF_Tracker(max_lost=0, tracker_output_format='mot_challenge')
elif tracker == 'SORT':
    tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
else:
    print ('TRACKER NOT IMPLEMENTED!')

def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    opt.heads = {'hm': opt.num_classes, 'hps': 18, 'rot': 8, 'dim': 3, 'prob': 1}
    opt.hm_hp=False
    opt.reg_offset=False
    opt.reg_hp_offset=False
    opt.faster=True
    Detector = CarPoseDetector
    detector = Detector(opt)
   
    time_tol=0

    vidcap = cv2.VideoCapture(video_file)
    success, image = vidcap.read()

    height, width, layers = image.shape
    size = (width, height)
    final_video = cv2.VideoWriter(os.path.join(os.getcwd(),'./data/final_video/final_video.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, size)

    num = 0

    while success:
        ret = detector.run(image,counter=num)

        tracks = tracker.update(ret['results_track'][0], ret['results_track'][1], ret['results_track'][2])
      
        final_frame = draw_tracks(ret['results_track'][3], tracks)

        zero_prefix = '0' * (6 - len(str(num)))
        cv2.imwrite(os.path.join(os.getcwd(), './data/final_frames/' + zero_prefix + '%d.png' % num),final_frame)

        final_video.write(final_frame)

        num+=1
        success, image = vidcap.read()

        time_str = ''
        for stat in time_stats:
            time_tol=time_tol+ret[stat]
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        time_str=time_str+'{} {:.3f}s |'.format('tol', time_tol/num)
        print(time_str)

    final_video.release()

if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
