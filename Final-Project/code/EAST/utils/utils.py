# -*- encoding: utf-8 -*-
'''
@文件    :utils.py
@说明    :
@时间    :2020/06/11 21:09:43
@作者    :罗瑞璞
@版本    :1.0
'''
import os
import sys
sys.path.append('/root/dl_nn/final/task1')
from dataset import custom_dataset
from torch.utils import data
from models.model import EAST
import torch
class evaluater(object):
    ''' Do evaluation in training process

        Args: 
            model: the model you need to evaluate on 
            val_img_path: the val img path
            val_gt_path: the val ground truth path
            val_num: the number of data you want to evaluate


    '''
    def __init__(self,val_img_path,val_gt_path,val_num):
        super(evaluater).__init__()
        self.val_img_list = [os.path.join(val_img_path, img_file) for img_file in sorted(os.listdir(val_img_path))][:val_num]
        self.val_gt_list = [os.path.join(val_gt_path, gt_file) for gt_file in sorted(os.listdir(val_gt_path))][:val_num]
    def evaluate(self,model):
        for idx in range(len(self.val_img_list)):
                pass
    

if __name__ == "__main__":
    trainset = custom_dataset('data/val/img','data/val/gt')
    train_loader = data.DataLoader(trainset, batch_size=4, num_workers=8,drop_last=True)
    img, gt_score, gt_geo, ignored_map,_ = next(iter(train_loader))
    model_path  = 'task1/pths/model_epoch_100.pth'
    model = EAST(pretrained = False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    output = model(img)
    print(1)