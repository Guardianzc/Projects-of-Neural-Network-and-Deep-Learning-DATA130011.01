'''
@Author: your name
@Date: 2020-06-18 11:37:09
@LastEditTime: 2020-06-18 11:38:28
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /final/YOLO/filter_img.py
'''
import cv2
from PIL import Image
import os
import numpy as np
import itertools
import pickle as pkl
from tqdm import tqdm
ImgRootPath = os.path.join('/root/dl_nn/final/data/result/image')
labelPath = os.path.join('/root/dl_nn/final/data/result/gt')
img_files = list(sorted(os.listdir(ImgRootPath)))
filter_img = []
filter_label = []
for imgName in tqdm(img_files):
    try:
        labelname = imgName[:-4]+'.txt'
        Img = np.array(Image.open(ImgRootPath + '/' + imgName).convert('L'))
        Label = open(labelPath+ '/' +labelname).readline()
        ResizedImg = cv2.resize(Img, (int(Img.shape[1] * (32 / Img.shape[0])), 32))
        l = [len(list(g)) for k, g in itertools.groupby(Label)]
        repeat_number = 0
        for n in l:
            if n > 1:
                repeat_number += (n - 1)
        input_length = 256 // 4 
        if len(Label)+repeat_number+2 <= input_length:
            filter_img.append(imgName)
            filter_label.append(labelname)
    except:
        print(imgName)
print(len(filter_img))
pkl.dump(filter_img,open('filter_img.pkl','wb'))
pkl.dump(filter_label,open('filter_label.pkl','wb'))


