'''
@Author: your name
@Date: 2020-06-11 11:52:57
@LastEditTime: 2020-06-17 04:04:38
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /final/train.py
'''
import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from utils.dataset import custom_dataset
from models.model import EAST
from utils.loss import Loss
import pickle as pkl
import os
import subprocess
import time
import numpy as np

import sys
import matplotlib
matplotlib.use('AGG')#或者PDF, SVG或PS
import matplotlib.pyplot as plt
from detect import detect_dataset

from torch.utils.data import sampler
def progress_bar(bar_len,loss,currentNumber, wholeNumber,lr):
    """
    bar_len 进度条长度
    currentNumber 当前迭代数
    wholeNumber 总迭代数
    """
    filled_len = int(round(bar_len * currentNumber / float(wholeNumber)))
    percents = round(100.0 * currentNumber / float(wholeNumber), 1)
    bar = '\033[32;1m%s\033[0m' % '>' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write(\
        '[%d/%d][%s] %s%s \033[31;1mloss\033[0m = %4f \033[31;1mlr\033[0m = %4f \r' %(int(currentNumber),int(wholeNumber),bar, '\033[32;1m%s\033[0m' % percents, '%',loss,lr))
    sys.stdout.flush()

def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval,checkpoint,eval_interval,test_img_path,submit_path):
	file_num = len(os.listdir(train_img_path))
	trainset = custom_dataset(train_img_path, train_gt_path)
	train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle = True, num_workers=num_workers, drop_last=True)
	
	criterion = Loss()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(pretrained = False)
	if checkpoint:
		model.load_state_dict(torch.load(checkpoint))
	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		# model = DataParallelModel(model)
		data_parallel = True
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=0)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1)
	whole_number = epoch_iter*(len(trainset)/batch_size)
	print("epoch size:%d"%(epoch_iter))
	print("batch size:%d"%(batch_size))
	print("data number:%d"%(len(trainset)))
	all_loss = []
	current_i = 0
	for epoch in range(epoch_iter):	
		
		model.train()
		
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_score, gt_geo, ignored_map,_) in enumerate(train_loader):
			current_i +=1
			start_time = time.time()
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			pred_score, pred_geo = model(img)
			loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
			
			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			lr_now = scheduler.get_last_lr()
			progress_bar(40,loss.item(),current_i,whole_number,lr_now[0])
		scheduler.step()
		print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
		all_loss.append(epoch_loss/int(file_num/batch_size))
		print(time.asctime(time.localtime(time.time())))
		plt.plot(all_loss)
		plt.savefig('loss_landscape.png')
		plt.close()
		print('='*50)
		if (epoch + 1) % interval == 0:
			state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1)))
			output = open(os.path.join(pths_path, 'loss.pkl'), 'wb')
			pkl.dump(all_loss, output)
		


if __name__ == '__main__':
	train_img_path = os.path.abspath('../data/train/img')
	train_gt_path  = os.path.abspath('../data/train/gt')
	test_img_path = os.path.abspath('../data/val/img')
	submit_path = './submit'
	pths_path      = './pths'
	checkpoint = './pths/model_epoch_300.pth'
	batch_size     = 32
	lr             = 1e-3
	num_workers    = 32
	epoch_iter     = 400
	save_interval  = 5
	eval_interval  = 5
	train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval,checkpoint,eval_interval,test_img_path,submit_path)	
	
