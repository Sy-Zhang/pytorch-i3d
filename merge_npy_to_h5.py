import argparse
import h5py
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-root', default='/localdisk/szhang83/Developer/', type=str)
parser.add_argument('-npy_dir', default='/localdisk/szhang83/Developer/LocalizingMoments/i3d_features/i3d_features_25fps', type=str)
parser.add_argument('-save_dir', default='/localdisk/szhang83/Developer/LocalizingMoments/data', type=str)
parser.add_argument('-save_name', default='i3d_features.h5', type=str)
parser.add_argument('-video_list_file', default='/localdisk/szhang83/Developer/LocalizingMoments/data/video_list.txt', type=str)
args = parser.parse_args()

if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)
f = h5py.File(os.path.join(args.save_dir, args.save_name), 'w')

video_list = open(args.video_list_file).readlines()
video_list = [item.strip() for item in video_list]

for video_name in video_list:
    video_id = video_name.split('.')[0]
    features = np.load(os.path.join(args.npy_dir,video_id+'.npy'))
    mean_features = np.zeros((6,1024))
    for i, feat in enumerate(np.split(features,[5,10,15,20,25])):
        if feat.shape[0] != 0:
            mean_features[i] = np.mean(feat,axis=0)
    f.create_dataset(video_name, data=features)