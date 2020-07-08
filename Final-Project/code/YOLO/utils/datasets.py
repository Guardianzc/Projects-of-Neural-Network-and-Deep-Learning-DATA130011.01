import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, path, img_size=416, augment=True, multiscale=True, normalized_labels=False):
        # with open(list_path, "r") as file:
        #     self.img_files = file.readlines()
        self.label_list = {'Latin':0,'Arabic':1,'Chinese':2,'Japanese':3,'Korean':4,'Bangla':5, 'Hindi':6,'Symbols':7,'None':8,'Mixed':9}

        self.path = path
        self.img_files = list(sorted(os.listdir(os.path.join(path, "img"))))
        # self.imgs = ['tr_img_07634.jpg']
        self.gt = list(sorted(os.listdir(os.path.join(path, "gt"))))
        # self.gt = ['tr_img_07634.txt']
        # self.label_files = [
        #     path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
        #     for path in self.img_files
        # ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = os.path.join(self.path, "img", self.img_files[index])

        # Extract image as PyTorch tensor
          
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        
        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        # label_path = self.label_files[index % len(self.img_files)].rstrip()
        gt_path = os.path.join(self.path, "gt", self.gt[index])
        gtf  = open(gt_path,'r',encoding='utf8')
        gt = gtf.readlines()
        gtf.close()
        gt = list(map(lambda x :x.strip().split(','),gt))

        N = len(gt)#
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        classes = []
        for i in range(N):
            point1 = list(map(lambda x:int(x),gt[i][0:2]))
            point2 = list(map(lambda x:int(x),gt[i][2:4]))
            point3 = list(map(lambda x:int(x),gt[i][4:6]))
            point4 = list(map(lambda x:int(x),gt[i][6:8]))
            xmin = min(point1[0],point2[0],point3[0],point4[0])
            xmax = max(point1[0],point2[0],point3[0],point4[0])
            ymin = min(point1[1],point2[1],point3[1],point4[1])
            ymax = max(point1[1],point2[1],point3[1],point4[1])
            x1.append(xmin)
            y1.append(ymin)
            x2.append(xmax)
            y2.append(ymax)
            # 加label
            # classes.append(self.label_list[gt[i][8]])
            # 就一类
            classes.append(0)
        targets = None
        x1,y1,x2,y2,classes = np.array(x1),np.array(y1),np.array(x2),np.array(y2),np.array(classes)
        boxes = np.zeros([N,5])
    # if os.path.exists(label_path):
        # boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
        # Extract coordinates for unpadded + unscaled image
        # x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        # y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        # x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        # y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]
        
        # Returns (x, y, w, h)
        boxes[:, 1] = ((x1+x2)/2)/ padded_w
        boxes[:, 2] = ((y1+y2)/2) / padded_h
        boxes[:, 3] = (x2-x1)*w_factor / padded_w
        boxes[:, 4] = (y2-y1)*h_factor / padded_h
        boxes[:, 0] = classes
        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = torch.tensor(boxes)

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)

if __name__ == "__main__":
    dataset = ListDataset('data/train', augment=False, multiscale=True)
    print(dataset[0])
