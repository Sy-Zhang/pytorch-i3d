import argparse
import h5py
import os

parser = argparse.ArgumentParser()
parser.add_argument('-mode', default='rgb', type=str, help='rgb or flow')
parser.add_argument('-load_model', default='models/rgb_imagenet.pt', type=str)
parser.add_argument('-root', default='/localdisk/szhang83/Developer/TALL/exp_data/Charades_STA/Charades_v1_480', type=str)
parser.add_argument('-video_list_file', default='/localdisk/szhang83/Developer/TALL/exp_data/Charades_STA/video_list.txt', type=str)
parser.add_argument('-save_dir', default='/localdisk/szhang83/Developer/TALL/exp_data/Charades_STA/i3d', type=str)
parser.add_argument('-save_name', default='i3d_features.h5', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    f = h5py.File(os.path.join(args.save_dir, args.save_name), 'w')

    video_list = open(args.video_list_file).readlines()
    video_list = [item.strip() for item in video_list]
    video_names = {vname.split('.')[0]: vname for vname in os.listdir(args.root)}

    f2 = h5py.File(os.path.join(args.save_dir, args.save_name + "-0~2000"), 'r')
    for video_name in video_list[0:2000]:
        video_id = video_name.split('.')[0]
        f.copy(f2[video_name],video_name)
    f2.close()

    f2 = h5py.File(os.path.join(args.save_dir, args.save_name + "-2000~4000"), 'r')
    for video_name in video_list[2000:4000]:
        video_id = video_name.split('.')[0]
        f.copy(f2[video_name],video_name)
    f2.close()

    f2 = h5py.File(os.path.join(args.save_dir, args.save_name + "-4000~6000"), 'r')
    for video_name in video_list[4000:6000]:
        video_id = video_name.split('.')[0]
        f.copy(f2[video_name],video_name)
    f2.close()

    f2 = h5py.File(os.path.join(args.save_dir, args.save_name + "-6000~6672"), 'r')
    for video_name in video_list[6000:]:
        video_id = video_name.split('.')[0]
        f.copy(f2[video_name],video_name)
    f2.close()


    # f2 = h5py.File(os.path.join(args.save_dir, args.save_name + "-0~1"), 'r')
    # for video_name in ['12090392@N02_13482799053_87ef417396.mov']:
    #     video_id = video_name.split('.')[0]
    #     try:
    #         f.copy(f2[video_names[video_id]],video_name)
    #     except:
    #         continue
    # f2.close()
    f.close()