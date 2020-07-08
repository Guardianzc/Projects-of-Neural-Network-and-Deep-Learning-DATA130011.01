import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
#from torchvision.transforms import functional as F
from torch.nn import functional as F
from PIL import Image
import argparse
from torch.utils.data import sampler
import pickle as pkl
import cv2
def resize(image, size):
    channel, height, width = image.shape
    new_width = int(width * size / height)
    image = F.interpolate(image.unsqueeze(0), size=(size, new_width), mode="nearest").squeeze(0)
    return image
def get_imagepath(root):
    image_paths = list(map(lambda x: os.path.join(root, x), [t for t in os.listdir(root) if (('jpg' in t) and ('nii' not in t))]))
    return image_paths

def get_landmarkpath(root):
    landmark_paths = list(map(lambda x: os.path.join(root, x), [t for t in os.listdir(root) if (('txt' in t) and ('nii' not in t))]))
    return landmark_paths

def get_landmarkpath(root):
    landmark_paths = list(map(lambda x: os.path.join(root, x), [t for t in os.listdir(root) if (('txt' in t) and ('nii' not in t))]))
    return landmark_paths
def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad
class ImageFolder(data.Dataset):
    def __init__(self, config):
        self.image_paths = get_imagepath(config.image_root)
        self.image_root = config.image_root
        self.image_paths = pkl.load(open('./task2_crnn/filter_img.pkl','rb'))
        self.landmark_paths = config.landmark_root
        self.landmark_paths = pkl.load(open('./task2_crnn/filter_label.pkl','rb'))
        self.landmark_root = config.landmark_root
        self.img_H = config.img_H
        self.file_length = len(self.image_paths)
        print("image count :{}".format(self.file_length))
    
    def __getitem__(self, index):
        transform1 = T.Compose([
                     T.ToTensor(),
                     ]
                     )
        normalization = T.Compose([
            T.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
        ])
        image_path = self.image_root+self.image_paths[index]
        landmark_path = self.landmark_root+self.landmark_paths[index]
        img = Image.open(image_path).convert('RGB')
        ###########################################
        # resize here
        img = np.array(Image.open(image_path))
        img = cv2.resize(img, (int(img.shape[1] * (32 / img.shape[0])), 32))
        # print('Before resize', img.shape)
        img = transform1(img)
        # print('After resize', img.shape)
        ###########################################
        
        # Handle images with less than three channels
        c, h, w = img.shape
        if c!= 3:
            n = 3-c
            img_new = img
            for i in range(n):
                img_new = torch.cat((img_new,img),0)
            img = img_new
        _, h, w = img.shape
        if _!=3:
            print(img.shape)
        img = normalization(img)
        ############################################
        # padding here
        padding_length = 256 - w
        pad_dims = (0, padding_length,
                    0, 0,
                    0, 0)
        matrix = F.pad(img, pad_dims, "constant", value=0)
        # print('After padding', matrix.shape)

        ############################################ 
        with open(landmark_path) as f:
            landmarks = f.readlines()
            if landmarks == []:
                GT = '###'
            else:
                GT = landmarks[0]
        return matrix, GT
    
    def __len__(self):
        return self.file_length

def get_loader(config, image_path, crop_size, batch_size,sampler, num_workers=2, mode='train', augmentation_prob=0.4):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(config)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
 
    return data_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_H', type=int, default=32)
    parser.add_argument('--image_root', type=str, default='/root/final/data/result/image/')
    parser.add_argument('--landmark_root', type=str, default='/root/final/data/result/gt/')
    config = parser.parse_args()
    dataset = ImageFolder(config)
    print(dataset[0])
