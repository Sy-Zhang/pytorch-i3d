import argparse
import h5py
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-npy_dir', default='/hdfs/resrchvc/v-yale/v-sozhan/Data/HACS-SF-RGB-Features/', type=str)
parser.add_argument('-save_dir', default='/home/v-yale', type=str)
parser.add_argument('-save_name', default='slowfast_features.h5', type=str)
parser.add_argument('-video_list_file', default='videos.txt', type=str)
args = parser.parse_args()

if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)
f = h5py.File(os.path.join(args.save_dir, args.save_name), 'w')

video_list = open(args.video_list_file).readlines()
video_list = [item.strip() for item in video_list]

for video_name in video_list:
    video_id = video_name.split('.')[0]
    features = np.load(os.path.join(args.npy_dir,video_id+'.npy'))
    f.create_dataset(video_name, data=features)