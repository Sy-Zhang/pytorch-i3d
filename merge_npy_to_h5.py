import argparse
import h5py
import os
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-npy_dir', default='/hdfs/resrchvc/v-yale/v-sozhan/Data/HACS-SF-RGB-Features/', type=str)
parser.add_argument('-save_dir', default='/home/v-yale', type=str)
parser.add_argument('-save_name', default='slowfast_features.h5', type=str)
parser.add_argument('-video_list_file', default='../videos.txt', type=str)
args = parser.parse_args()

if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)
f = h5py.File(os.path.join(args.save_dir, args.save_name), 'w')

video_list = open(args.video_list_file).readlines()
video_list = [item.strip() for item in video_list]

pbar = tqdm(total=len(video_list))
for video_name in video_list[1000:2000]:
    video_id = video_name.split('.')[0]
    try:
        #print(os.path.join(args.npy_dir,video_id+'.npy'))
        features = np.load(os.path.join(args.npy_dir,video_id+'.npy'))
    except:
        print('load error')
        continue
    f.create_dataset(video_name, data=features)
    pbar.update(1)
pbar.close()
