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
import glob
import random
random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('-mode', default='rgb', type=str, help='rgb or flow')
parser.add_argument('-load_model', default='models/rgb_imagenet.pt', type=str)
parser.add_argument('-video_list_file', default='videos.txt', type=str)
parser.add_argument('-root', default='/data/home2/hacker01/MSM/Data/HACS-Segments', type=str)
parser.add_argument('-save_dir', default='/data/home2/hacker01/MSM/Data/HACS-Features', type=str)
parser.add_argument('-batch_size', default=12, type=int)
parser.add_argument('-start', default=0, type=int)
parser.add_argument('-end', default=6672, type=int)
args = parser.parse_args()

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def run(mode, root, load_model, save_dir, video_list_file, batch_size, fps = 24):
    if mode == 'rgb':
        trans = torchvision.transforms.Compose([
            transforms.GroupResize(256, interpolation=Image.BILINEAR),
            transforms.GroupCenterCrop(224),
            transforms.Stack4d(roll=False),
            transforms.ToTorchFormatTensor4d(div=True),
            transforms.GroupNormalize(mean=[.5, .5, .5],std=[.5, .5, .5])
        ])
    else:
        trans = torchvision.transforms.Compose([transforms.CenterCrop(224)])

    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.eval()
    i3d.load_state_dict(torch.load(load_model),strict=False)
    i3d.cuda()


    # read video list from the txt list
    video_list = open(video_list_file).readlines()
    video_list = [item.strip() for item in video_list]
    video_list = video_list
    temp_path = os.path.join(os.getcwd(), 'temp')
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    error_fid = open('error.txt', 'a')

    stride = 8

    for video_path in video_list[10000:30000]:
        video_id = video_path.split('/')[-1].split('.')[0]
        save_path = os.path.join(save_dir,video_id+'.npy')
        if os.path.exists(save_path):
            continue
        print('video_path', video_path)
        frame_path = os.path.join(temp_path, video_id)
        if not os.path.exists(frame_path):
            os.mkdir(frame_path)

        print('Extracting video frames ...')
        # using ffmpeg to extract video frames into a temporary folder
        # example: ffmpeg -i video_validation_0000051.mp4 -q:v 2 -f image2 output/image%5d.jpg

        # for philly
        os.system('cp '+video_path+' '+frame_path)
        duplicate_video_path = os.path.join(frame_path, video_path.split('/')[-1])
        os.system('/home/v-yale/ffmpeg-4.2.1-amd64-static/ffmpeg -i ' + duplicate_video_path + ' -q:v 2 -f image2 -vf fps={} '.format(fps) + frame_path + '/image_%6d.jpg')

        # for gpu07
        # os.system('ffmpeg -i ' + video_path + ' -q:v 2 -f image2 -vf fps={} '.format(fps) + frame_path + '/image_%6d.jpg')

        if mode == 'flow':
            os.system(
                "python /home/v-yale/flownet2-pytorch/main.py "
                "--inference --model FlowNet2 --save_flow --inference_dataset ImagesFromFolder "
                "--inference_dataset_root {} --inference_dataset_iext jpg "
                "--resume /home/v-yale/flownet2-pytorch/checkpoints/FlowNet2_checkpoint.pth.tar "
                "--inference_batch_size 64".format(frame_path)
            )
        if mode == 'rgb':
            image_list = sorted(glob.glob(frame_path+'/*.jpg'))[:-1]
        else:
            image_list = sorted(glob.glob(frame_path+'/*.flo'))[:-1]

        total_frames = len(image_list)
        if total_frames == 0:
            error_fid.write(video_path + '\n')
            print('Fail to extract frames for video: %s' % video_id)
            continue
        nb_segments = round(total_frames / fps)
        valid_frames = min(nb_segments*fps+stride, total_frames)
        image_list = image_list[:valid_frames]

        sample_list = list(range(stride,valid_frames-stride,stride))
        n_feat = len(sample_list)
        n_batch = n_feat // batch_size
        if n_feat - n_batch * batch_size > 0:
            n_batch = n_batch + 1
        print('n_frames: %d; n_feat: %d; n_batch: %d' % (total_frames, n_feat, n_batch))

        features = []
        for i in range(n_batch):
            input_blobs = []
            num_sample = batch_size if i < n_batch-1 else n_feat-(n_batch-1) * batch_size
            for j in range(num_sample):
                if mode == 'rgb':

                    imgs = [Image.open(os.path.join(frame_path, image_list[k])) for k in
                            range(sample_list[i * batch_size + j]-stride, sample_list[i * batch_size + j]+stride)]
                else:
                    imgs = [readFlow(os.path) for k in
                            range(sample_list[i * batch_size + j]-stride, sample_list[i * batch_size + j]+stride)]
                imgs = trans(imgs)
                input_blobs.append(imgs)
            input_blobs = torch.stack(input_blobs).permute(0, 4, 1, 2, 3).cuda()
            batch_output = i3d.extract_features(input_blobs).view(num_sample,-1)
            features.append(batch_output.cpu().detach())

        features = torch.cat(features, 0)
        features = features.numpy()
        np.save(save_path, features)
        print('%s has been processed...' % video_id)

        # clear temp frame folders
        os.system('rm -rf ' + frame_path)

if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode,
        root=args.root,
        load_model=args.load_model,
        save_dir=args.save_dir,
        video_list_file=args.video_list_file,
        batch_size=args.batch_size)
