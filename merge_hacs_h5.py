import argparse
import h5py
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-save_dir', default='/data/home2/hacker01/Data/', type=str)
parser.add_argument('-load_dir', default='/data/home2/hacker01/MSM/Data/HACS-SF-RGB-Features-H5', type=str)
parser.add_argument('-save_name', default='slowfast_features.h5', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    # f = h5py.File(os.path.join(args.save_dir, args.save_name), 'w')
    # for i in range(61):
    #     print(args.save_name + ".part{:02d}".format(i))
    #     try:
    #         f_i = h5py.File(os.path.join(args.load_dir, args.save_name + ".part{:02d}".format(i)), 'r')
    #     except:
    #         print("part{:02d} error".format(i))
    #         continue
    #     pbar = tqdm(total=len(list(f_i.keys())))
    #     for vid in list(f_i.keys()):
    #         # print(vid.split('.')[0])
    #         if vid in f:
    #             continue
    #         try:
    #             f.copy(f_i[vid], vid.split('.')[0])
    #         except:
    #             print("{} error".format(vid))
    #         pbar.update(1)
    #     pbar.close()
    #     f_i.close()
    # f.close()
    print(os.path.join(args.save_dir, args.save_name))
    f = h5py.File(os.path.join(args.save_dir, args.save_name), 'r')
    for vid in list(f.keys()):
        pass
        # print(vid)