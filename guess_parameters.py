import os
import torchvision
import numpy as np
from PIL import Image
import torch

gt_rgb = np.load("data/v_CricketShot_g04_c01_rgb.npy")
print(gt_rgb.shape)
print(gt_rgb.max())
print(gt_rgb.min())

trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256, interpolation=Image.BILINEAR),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])
image_dir = "/localdisk/szhang83/Developer/LocalizingMoments/download/temp_videos/images/"
image_list = sorted(os.listdir(image_dir))

images = []
for i in image_list:
    img = Image.open(os.path.join(image_dir, i))
    img = trans(img).permute(1, 2, 0)
    images.append(img.numpy())
images = np.array(images)[np.newaxis, :][:, 0:-1, :, :, :]
np.save('data/my_v_CricketShot_g04_c01_rgb.npy', images)
print ('images.shape: ',images.shape)
diff = np.sum(np.abs(images - gt_rgb))
print ('diff: ',diff)
total = np.sum(np.abs(gt_rgb))
print (total)
print (diff / float(total) * 100, "%")
