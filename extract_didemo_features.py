import os
import argparse
from subprocess import DEVNULL, STDOUT, check_call

import h5py
import numpy as np
import torch
import torchvision
import transforms
import torch.nn.functional as F
from pytorch_i3d import InceptionI3d
from PIL import Image
import random
random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('-mode', default='rgb', type=str, help='rgb or flow')
parser.add_argument('-load_model', default='models/rgb_imagenet.pt', type=str)
parser.add_argument('-video_list_file', default='/localdisk/szhang83/Developer/LocalizingMoments/data/video_list.txt', type=str)
parser.add_argument('-root', default='/localdisk/szhang83/Developer/LocalizingMoments/download/videos', type=str)
parser.add_argument('-gpu', default='1', type=str)
parser.add_argument('-save_dir', default='/localdisk/szhang83/Developer/LocalizingMoments/i3d_data', type=str)
parser.add_argument('-save_name', default='i3d_features.h5', type=str)
parser.add_argument('-batch_size', default=12, type=str)
parser.add_argument('-fps', default=10, type=int)
parser.add_argument('-start', default=0, type=int)
parser.add_argument('-end', default=10642, type=int)
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

def run(mode, root, load_model, save_dir, save_name, video_list_file, batch_size, fps):
    trans = torchvision.transforms.Compose([
        transforms.GroupResize(256, interpolation=Image.BILINEAR),
        transforms.GroupCenterCrop(224),
        transforms.Stack4d(roll=False),
        transforms.ToTorchFormatTensor4d(div=True),
        transforms.GroupNormalize(mean=[.5, .5, .5],std=[.5, .5, .5])
    ])
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load(load_model),strict=False)
    i3d.cuda()
    i3d.eval()

    # read video list from the txt list
    video_names = {vname.split('.')[0]: vname for vname in os.listdir(root)}
    video_list = open(video_list_file).readlines()
    video_list = [item.strip() for item in video_list]
    video_list = video_list[args.start:args.end]
    temp_path = os.path.join(os.getcwd(), 'temp')
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    error_fid = open('error.txt', 'a')

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    f = h5py.File(os.path.join(save_dir, save_name+'-{}~{}'.format(args.start,args.end)), 'w')

    max_time = 30

    for video_name in video_list:
        video_id = video_name.split('.')[0]
        if video_names[video_id] in f:
            continue
        video_path = os.path.join(root, video_names[video_id])
        print('video_path', video_path)
        frame_path = os.path.join(temp_path, video_id)
        if not os.path.exists(frame_path):
            os.mkdir(frame_path)

        print('Extracting video frames ...')
        # using ffmpeg to extract video frames into a temporary folder
        # example: ffmpeg -i video_validation_0000051.mp4 -q:v 2 -f image2 output/image%5d.jpg
        os.system('/localdisk/szhang83/.linuxbrew/bin/ffmpeg -i ' + video_path + ' -q:v 2 -f image2 -vf fps={} '.format(fps) + frame_path + '/image_%6d.jpg')

        image_list = sorted(os.listdir(frame_path))[:-1]
        total_frames = min(len(image_list), fps*max_time)
        if total_frames == 0:
            error_fid.write(video_name + '\n')
            print('Fail to extract frames for video: %s' % video_name)
            continue
        n_feat = total_frames // fps
        valid_frames = n_feat*fps
        image_list = image_list[:valid_frames]

        n_batch = n_feat // batch_size
        if n_feat - n_batch * batch_size > 0:
            n_batch = n_batch + 1
        print('n_frames: %d; n_feat: %d; n_batch: %d' % (total_frames, n_feat, n_batch))

        features = []
        for i in range(n_batch):
            input_blobs = []
            num_sample = batch_size if i < n_batch-1 else n_feat-(n_batch-1) * batch_size
            for j in range(num_sample):
                imgs = [Image.open(os.path.join(frame_path, image_list[k])) for k in
                        range((i * batch_size + j)*fps, fps*(i * batch_size + j+1))]
                imgs = trans(imgs)
                input_blobs.append(imgs)
            input_blobs = torch.stack(input_blobs).permute(0, 4, 1, 2, 3).cuda()
            batch_output = i3d.extract_features(input_blobs).view(num_sample,-1)
            features.append(batch_output.cpu().detach())

        features = torch.cat(features, 0)
        features = features.detach().numpy()
        np.save(os.path.join(save_dir,video_id),features)
        # f.create_dataset(video_name, data=features)
        print('%s has been processed...' % video_names[video_id])


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode,
        root=args.root,
        load_model=args.load_model,
        save_dir=args.save_dir,
        save_name=args.save_name,
        video_list_file=args.video_list_file,
        batch_size=args.batch_size,
        fps=args.fps)
