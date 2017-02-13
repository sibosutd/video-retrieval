"""
Sampe script of feature extraction for single video. Adapted from eval_net.py.
"""

import argparse
import os
import sys
import math
import cv2
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix

sys.path.append('.')

parser = argparse.ArgumentParser()
parser.add_argument('modality', type=str, choices=['rgb', 'flow'])
parser.add_argument('frame_path', type=str, help="root directory holding the frames")
parser.add_argument('net_proto', type=str)
parser.add_argument('net_weights', type=str)
parser.add_argument('--rgb_prefix', type=str, help="prefix of RGB frames", default='img_')
parser.add_argument('--flow_x_prefix', type=str, help="prefix of x direction flow images", default='flow_x_')
parser.add_argument('--flow_y_prefix', type=str, help="prefix of y direction flow images", default='flow_y_')
parser.add_argument('--save_scores', type=str, default=None, help='the filename to save the scores in')
parser.add_argument('--num_worker', type=int, default=1)
parser.add_argument("--caffe_path", type=str, default='./lib/caffe-action/', help='path to the caffe toolbox')
parser.add_argument("--gpus", type=int, nargs='+', default=None, help='specify list of gpu to use')

args = parser.parse_args()
print args

sys.path.append(os.path.join(args.caffe_path, 'python'))
from pyActionRecog.action_caffe import CaffeNet

gpu_list = args.gpus

# specify the feature name here, refer to deploy.prototxt file
score_name = 'fc-action'


def build_net():
    global net
    my_id = multiprocessing.current_process()._identity[0] \
        if args.num_worker > 1 else 1
    if gpu_list is None:
        net = CaffeNet(args.net_proto, args.net_weights, my_id-1)
    else:
        net = CaffeNet(args.net_proto, args.net_weights, gpu_list[my_id - 1])


def eval_video():
    global net

    video_frame_path = args.frame_path
    print 'frame path:', video_frame_path

    # simple way to calculate the frame count
    frame_cnt = len([name for name in os.listdir(video_frame_path)]) / 3
    print 'frame count:', frame_cnt 

    if args.modality == 'rgb':
        step = 1
        stack_depth = 1
    elif args.modality == 'flow':
        step = 3
        stack_depth = 5
    
    frame_ticks = range(1, frame_cnt+1-stack_depth+1, step)

    frame_scores = []
    for tick in frame_ticks:
        if args.modality == 'rgb':
            name = '{}{:05d}.jpg'.format(args.rgb_prefix, tick)
            frame = cv2.imread(os.path.join(video_frame_path, name), cv2.IMREAD_COLOR)
            scores = net.predict_single_frame([frame,], score_name, frame_size=(340, 256))
            frame_scores.append(scores)
        if args.modality == 'flow':
            frame_idx = [min(frame_cnt, tick+offset) for offset in xrange(stack_depth)]
            flow_stack = []
            for idx in frame_idx:
                x_name = '{}{:05d}.jpg'.format(args.flow_x_prefix, idx)
                y_name = '{}{:05d}.jpg'.format(args.flow_y_prefix, idx)
                flow_stack.append(cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_GRAYSCALE))
                flow_stack.append(cv2.imread(os.path.join(video_frame_path, y_name), cv2.IMREAD_GRAYSCALE))
            scores = net.predict_single_flow_stack(flow_stack, score_name, frame_size=(340, 256))
            frame_scores.append(scores)

    print 'feature extraction done'
    sys.stdin.flush()
    return np.array(frame_scores)


build_net()
video_features = eval_video()

# save features
if args.save_scores is not None:
    np.save(args.save_scores, video_features)
