import os
import torchvision
import numpy as np
from PIL import Image
import cv2
import torch
import videotransforms
gt_rgb = np.load("data/v_CricketShot_g04_c01_rgb.npy")
print(gt_rgb.shape)
print(gt_rgb.max())
print(gt_rgb.min())

trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256, interpolation=Image.BILINEAR),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    # torchvision.transforms.Normalize(mean=[.434, .405, .378], std=[.152, .149, .157])
])
test_transforms = torchvision.transforms.Compose([videotransforms.CenterCrop(224)])

image_dir = "/localdisk/szhang83/Developer/LocalizingMoments/download/temp_videos/images/"
image_list = sorted(os.listdir(image_dir))

images = []
for i in image_list:
    # img = cv2.imread(os.path.join(image_dir, i))[:, :, [2, 1, 0]]
    # w,h,c = img.shape
    # if w < 226 or h < 226:
    #     d = 226.-min(w,h)
    #     sc = 1+d/min(w,h)
    #     img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    # img = (img/255.)*2 - 1
    # images.append(img)

    img = Image.open(os.path.join(image_dir, i))
    img = trans(img).permute(1, 2, 0)
    images.append(img.numpy())
# images = test_transforms(np.array(images))
images = np.array(images)[np.newaxis, :][:, 0:-1, :, :, :]
np.save('data/my_v_CricketShot_g04_c01_rgb.npy', images)
print ('images.shape: ',images.shape)
diff = np.sum(np.abs(images - gt_rgb))
print ('diff: ',diff)
total = np.sum(np.abs(gt_rgb))
print (total)
print (diff / float(total) * 100, "%")
