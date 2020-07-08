import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import sys
import csv
from Model import CRNN as crnn
from Alphabet import Alphabet
import dataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from random import shuffle
import numpy as np
from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import argparse
from torch.utils.data import sampler
import pickle as pkl
import cv2
import numpy as np 
import shapely
from shapely.geometry import Polygon,MultiPoint  #多边形
from torch.optim.lr_scheduler import StepLR
def progress_bar(bar_len,loss,val_f1,test_f1,lr,currentNumber, wholeNumber):
    """
    bar_len 进度条长度
    currentNumber 当前迭代数
    wholeNumber 总迭代数
    """
    filled_len = int(round(bar_len * currentNumber / float(wholeNumber)))
    percents = round(100.0 * currentNumber / float(wholeNumber), 1)
    bar = '\033[32;1m%s\033[0m' % '>' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write(\
        '[%d/%d][%s] %s%s \033[31;1mloss\033[0m = %4f \033[36;1macc\033[0m= %4f \033[36;1mbestAcc\033[0m= %4f \033[33;1mlr\033[0m= %4f  \r' %\
         (int(currentNumber),int(wholeNumber),bar, '\033[32;1m%s\033[0m' % percents, '%',loss,val_f1,test_f1,lr))
    sys.stdout.flush()
def get_accuracy(text):
    match = 0
    for pair in text:
        if pair[0]==pair[1]:
            match+=1
    return match

def get_testaccuracy(label,gt):
    match = 0
    for l in label:
        if l in gt:
            match += 1
            gt.remove(l)

    return match
def transformer(img):
    transform1 = T.Compose([
                T.ToTensor(),
                    ]
                    )
    normalization = T.Compose([
        T.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
    ])

    # resize here
    img = np.array(img)
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
    return matrix
    ############################################ 
class Solver(object):

    def __init__(self, config, train_loader, valid_loader):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        # Models
        self.net = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.batch_size = config.batch_size
          # 
        # todo: find out how to use self-written cost function
        
        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.lstm_hidden = config.lstm_hidden
        self.converter = Alphabet(dataLoader.get_landmarkpath(config.landmark_root), store=False, load= True)
        self.blank = self.converter.blank()
        self.criterion = nn.CTCLoss(blank=self.blank, reduction='mean')

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        # 所有文本的路径
        self.text_path = config.text_path
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.nclass = self.converter.length()
        self.img_H = config.img_H
        self.build_model()
    

    def build_model(self):
        """Build generator and discriminator."""
        # self.imgH: High of image
        self.net = crnn(self.img_H, self.img_ch, self.nclass, self.lstm_hidden)
        self.optimizer = optim.Adam(list(self.net.parameters()), self.lr, (self.beta1, self.beta2))
        self.scheduler = StepLR(self.optimizer, step_size = self.num_epochs_decay, gamma=0.5)
        self.data_parallel = False
        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
            # model = DataParallelModel(model)
            self.data_parallel = True

        self.net.to(self.device)
        classname = self.net.__class__.__name__
        if classname.find('Conv') != -1:
            self.net.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            self.net.weight.data.normal_(1.0, 0.02)
            self.net.bias.data.fill_(0)
        # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()  # return the number of usage of this parameter
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = g_lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.net.zero_grad()

    def compute_accuracy(self, preds, GT):
        preds_size = Variable(torch.IntTensor([preds.size(0)] * self.batch_size))
        _, preds = preds.max(2)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, GT):
            if pred == target:
                n_correct += 1
        return n_correct
        
    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training =========================================== #
        # =========================================================================================== #
        net_path = os.path.join(self.model_path, '%s-%d-%.4f-%d_val_best.pkl' % (self.model_type,self.num_epochs,self.lr,self.num_epochs_decay))
        # Net Train
        # Train for Encoder
        lr = self.lr
        meanloss = 0
        acc = 0.	# Accuracy
        length = 0
        best_acc = 0
        for epoch in range(self.num_epochs):

            self.net.train(True)
            epoch_loss = 0
            length = 0
            num_train = len(self.train_loader)
            data_length = 0
            for i, (images, GT) in enumerate(self.train_loader):
                # GT : Ground Truth
                images = images.to(self.device)

                # preds: Result
                t, l = self.converter.encode(GT)
                t = t.to(self.device)
                # ? 我还没理解CTCLoss的输入格式
                l = l.to(self.device)
                preds = self.net(images)  # [length, batch_size, total_length]
                preds_size = Variable(torch.IntTensor([preds.size(0)] * self.batch_size))
                if len(l) != self.batch_size:
                    continue
                loss = self.criterion(preds, t, preds_size, l)
                # Backprop + optimize
                self.reset_grad()  # zero the gradient buffers, not zero the parameters
                loss.backward()
                self.optimizer.step()
                # Print the log info
                progress_bar(50,loss.item(),acc,best_acc,self.lr,i+epoch*len(self.train_loader),self.num_epochs*len(self.train_loader))

            self.scheduler.step()
                
            

            if (epoch+1)%20==1:
                # ===================================== Validation ====================================#
                self.net.train(False)  # control the train_phase from the model
                self.net.eval()  # the model will automatically fix BN and Dropout
                acc = 0.
                for i, (images, GT) in enumerate(self.valid_loader):

                    images = images.to(self.device)
                    # GT = GT.to(self.device)

                    SR = self.net(images)
                    SR = SR.permute(1,0,2).max(2)[1]
                    text = []
                    for i in range(SR.shape[0]):
                        for j in range(len(SR[i,:])):
                            if SR[i, j] != self.blank:
                                SR[i, j] -= 1
                        decode_text = [list(map(lambda x:self.converter.index[x],SR[i,:].detach().cpu().numpy().tolist())),GT[i]]
                        decode_text[0] =''.join([ x if x !='<BLANK>' else '' for x in decode_text[0]])
                        if decode_text[0]:
                            new_text = decode_text[0][0]
                            for char_index in range(1, len(decode_text[0])):
                                if decode_text[0][char_index] != decode_text[0][char_index - 1]:
                                    new_text += decode_text[0][char_index]
                            if new_text  == '#':
                                new_text = '###'
                            decode_text[0] = new_text                        
                        text.append(decode_text)
                    acc += get_accuracy(text)
                acc = acc/len(self.valid_loader)
                # Save Best Net model
                if acc > best_acc: 
                    best_acc = acc
                    net_path = os.path.join(self.model_path, '%s-%d-%.4f-%d_best_val.pkl' % (self.model_type,self.num_epochs,self.lr,self.num_epochs_decay))
                    print('Best %s model score : %.4f' % (self.model_type, best_acc))
                    state_dict = self.net.module.state_dict() if self.data_parallel else self.net.state_dict()
                    torch.save(state_dict, net_path)



    def line_slope(self,x1,y1,x2,y2,x3,y3,x4,y4):
                k1=(y2-y1)/(x2-x1)
                k2=(y3-y2)/(x3-x2)
                k3=(y4-y3)/(x4-x3)
                k4=(y1-y4)/(x1-x4)
                return k1,k2,k3,k4

    def get_IOU(self,coord1,coord2):
        line1=coord1   #四边形四个点坐标的一维数组表示，[x,y,x,y....]
        a=np.array(line1).reshape(4, 2)   #四边形二维坐标表示
        poly1 = Polygon(a).convex_hull  #python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上

        line2=coord2
        b=np.array(line2).reshape(4, 2)
        poly2 = Polygon(b).convex_hull
        
        union_poly = np.concatenate((a,b))   #合并两个box坐标，变为8*2
        #print(union_poly)

        if not poly1.intersects(poly2): #如果两四边形不相交
            iou = 0
        else:
            try:
                inter_area = poly1.intersection(poly2).area   #相交面积
                union_area = MultiPoint(union_poly).convex_hull.area
                if union_area == 0:
                    iou= 0
                iou=float(inter_area) / union_area
                # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积  
                # 第二种： 交集 / 并集（常见矩形框IOU计算方式） 
            except shapely.geos.TopologicalError:
                print('shapely.geos.TopologicalError occured, iou set to 0')
                iou = 0
        return iou
    def get_max_IOU_box(self,coord,gt_coord):
        '''
        input: 
            coord: detect的box
            gt_coord: 该张图片里所有的gt box
        return:
            max_IOU: input box 和 gt 里的box 最大匹配到的iou是多少
            idx：该最大匹配的gt box 是第几个
        '''
        max_iou = 0
        idx = 0
        for i,gt_box in enumerate(gt_coord):
            iou = self.get_IOU(coord,gt_box)
            if iou>max_iou:
                max_iou = iou
                idx = i
        return idx,max_iou
    def val(self):

        landmark_path = './data/submit'
        image_path = './data/val/img/'
        note_path = './data/val/gt/'
        image_files = list(sorted(os.listdir(image_path)))
        label_files = list(sorted(os.listdir(landmark_path)))
        note_files = list(sorted(os.listdir(note_path)))
        count = 0
        count_files = 0
        length = len(image_files)
        """val encoder, generator and discriminator."""
        # ====================================== Training =========================================== #
        # =========================================================================================== #
        net_path = os.path.join(self.model_path, '%s-%d-%.4f-%d_best_val.pkl' 
        % (self.model_type,self.num_epochs,self.lr,self.num_epochs_decay))
        # Net Train
        # if False:
        if os.path.isfile(net_path):
            # Load the pretrained Encoder
            self.net.load_state_dict(torch.load(net_path))
            print('%s is Successfully Loaded from %s'%(self.model_type, net_path))
            # ===================================== Validation ====================================#
            self.net.train(False)  # control the train_phase from the model
            self.net.eval()  # the model will automatically fix BN and Dropout
            acc = 0.
            best_acc = 0
            txt_length = 0

        ############################### 开始测试 ##################################
        precision = []
        recall = []

        for ii in tqdm(range(length)): # ii 为单张图片的 index
            img_path = os.path.join(image_path, image_files[ii])
            label_path = os.path.join(landmark_path, label_files[ii])
            note_paths = os.path.join(note_path, note_files[ii])
            image = Image.open(img_path)
            landmarks = open(label_path,'r',encoding='utf8').readlines()
            notes = open(note_paths,'r',encoding='utf8').readlines()
            len_notes = len(notes)
            landmark_length = len(landmarks)

            gt = []# 用来存放 正确的文本
            text = []#用来存放错误的文本
            gt_coord = []# 用来存放正确的框

            # notes中为 单张图片所有的gt
            for k in notes:
                gt.append(k.rstrip('\n').split(',')[-1])
                gt_coord.append(list(map(lambda x:int(x),k.rstrip('\n').split(',')[:8])))
            # landmarks 中是detect出来的框

            match=0# match 用来计算每张图片的 precision 和 recall 并计算每张图片的 f1

            for k in range(landmark_length):
                landmark = landmarks[k]
                coord = landmark.split(',')[:8:]
                for j in range(8):
                    coord[j] = int(coord[j])
                
                # 得到最大IOU的那一个gt框的index
                try:
                    idx,max_IOU = self.get_max_IOU_box(coord,gt_coord)
                except:
                    max_IOU = 0


                # 切图片
                if max_IOU>=0.5:
                    x1 = min(coord[0], coord[2], coord[4], coord[6])
                    x2 = max(coord[0], coord[2], coord[4], coord[6])
                    y1 = min(coord[1], coord[3], coord[5], coord[7])
                    y2 = max(coord[1], coord[3], coord[5], coord[7])
                    images = image.crop((x1,y1,x2,y2))
                    try:
                        images = transformer(images)
                    except:
                        continue
                    images = images.unsqueeze(0)
                    images = images.to(self.device)
                    SR = self.net(images)
                    SR = SR.permute(1,0,2).max(2)[1]
                    
                    # 对所有batch进行计算 实际只有1
                    for i in range(SR.shape[0]):
                        # 将SR与字典对齐
                        for j in range(len(SR[i,:])):
                            if SR[i, j] != self.blank:
                                SR[i, j] -= 1
                        decode_text = [list(map(lambda x:self.converter.index[x],SR[i,:].detach().cpu().numpy().tolist()))]
                        decode_text[0] =''.join([ x if x !='<BLANK>' else '' for x in decode_text[0]])
                        # 简单粗暴的删去重复的字符 
                        if decode_text[0]:
                            new_text = decode_text[0][0]
                            for char_index in range(1, len(decode_text[0])):
                                if decode_text[0][char_index] != decode_text[0][char_index - 1]:
                                    new_text += decode_text[0][char_index]
                            if new_text  == '#':
                                new_text = '###'
                        
                    decode_text[0] = new_text
                    if gt[idx] == decode_text[0]:
                        match +=1
                else:
                    pass
            if landmark_length!=0:
                precision.append(match/landmark_length)
            else:
                precision.append(0)
            recall.append(match/len(gt))
        mean_precision = sum(precision)/len(precision)
        mean_recall = sum(recall)/len(recall)
        mean_f1 = (2*mean_precision*mean_recall)/(mean_precision+mean_recall)
        print('mean_precision:%f'%(mean_precision))
        print('mean_recall:%f'%(mean_recall))
        print('mean_f1:%f'%(mean_f1))

    def detect(self):
        landmark_path = './data/submit'
        image_path = './data/val/img/'
        note_path = './data/val/gt/'
        image_files = list(sorted(os.listdir(image_path)))
        label_files = list(sorted(os.listdir(landmark_path)))
        note_files = list(sorted(os.listdir(note_path)))
        count = 0
        count_files = 0
        length = len(image_files)
    
        net_path = os.path.join(self.model_path, '%s-%d-%.4f-%d_best_val.pkl' 
        % (self.model_type,self.num_epochs,self.lr,self.num_epochs_decay))
        # Net Train
        # if False:
        if os.path.isfile(net_path):
            # Load the pretrained Encoder
            self.net.load_state_dict(torch.load(net_path))
            print('%s is Successfully Loaded from %s'%(self.model_type, net_path))
            # ===================================== Validation ====================================#
            self.net.train(False)  # control the train_phase from the model
            self.net.eval()  # the model will automatically fix BN and Dropout


        for ii in tqdm(range(length)): # ii 为单张图片的 index
            img_path = os.path.join(image_path, image_files[ii])
            label_path = os.path.join(landmark_path, label_files[ii])
            note_paths = os.path.join(note_path, note_files[ii])
            image = Image.open(img_path)
            landmarks = open(label_path,'r',encoding='utf8').readlines()
            notes = open(note_paths,'r',encoding='utf8').readlines()
            len_notes = len(notes)
            landmark_length = len(landmarks)

            result_file = os.path.join('./data/final_submit', note_files[ii])
            rf = open(result_file,'w',encoding = 'utf8')
            for k in range(landmark_length):
                landmark = landmarks[k]
                coord = landmark.split(',')[:8]
                for j in range(8):
                    coord[j] = int(coord[j])
                
                # 切图片
                x1 = min(coord[0], coord[2], coord[4], coord[6])
                x2 = max(coord[0], coord[2], coord[4], coord[6])
                y1 = min(coord[1], coord[3], coord[5], coord[7])
                y2 = max(coord[1], coord[3], coord[5], coord[7])
                images = image.crop((x1,y1,x2,y2))
                try:
                    images = transformer(images)
                except:
                    continue
                images = images.unsqueeze(0)
                images = images.to(self.device)
                SR = self.net(images)
                SR = SR.permute(1,0,2).max(2)[1]
                
                # 对所有batch进行计算 实际只有1
                for i in range(SR.shape[0]):
                    # 将SR与字典对齐
                    for j in range(len(SR[i,:])):
                        if SR[i, j] != self.blank:
                            SR[i, j] -= 1
                    decode_text = [list(map(lambda x:self.converter.index[x],SR[i,:].detach().cpu().numpy().tolist()))]
                    decode_text[0] =''.join([ x if x !='<BLANK>' else '' for x in decode_text[0]])
                    # 简单粗暴的删去重复的字符 
                    if decode_text[0]:
                        new_text = decode_text[0][0]
                        for char_index in range(1, len(decode_text[0])):
                            if decode_text[0][char_index] != decode_text[0][char_index - 1]:
                                new_text += decode_text[0][char_index]
                        if new_text  == '#':
                            new_text = '###'
                    
                decode_text[0] = new_text
                for j in range(8):
                    coord[j] = str(coord[j])
                coord.append(decode_text[0])
                rf.write(','.join(coord)+'\n')
            rf.close()
