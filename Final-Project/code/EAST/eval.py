'''
@Author: your name
@Date: 2020-06-11 00:53:33
@LastEditTime: 2020-06-17 16:19:42
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /final/EAST/eval.py
'''
import time
import torch
import subprocess
import os
from models.model import EAST
from detect import detect_dataset
import numpy as np
import shutil


def eval_model(model_name, test_img_path, submit_path, save_flag=True):
	if os.path.exists(submit_path):
		shutil.rmtree(submit_path) 
	os.mkdir(submit_path)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(pretrained = False).to(device)
	model.load_state_dict(torch.load(model_name))
	model.eval()
	
	start_time = time.time()
	detect_dataset(model, device, test_img_path, submit_path)
	os.chdir(submit_path)
	res = subprocess.getoutput('zip -q submit.zip *.txt')
	res = subprocess.getoutput('mv submit.zip ../')
	os.chdir('../')
	res = subprocess.getoutput('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')
	print(res)
	os.remove('./submit.zip')
	print('eval time is {}'.format(time.time()-start_time))	

	if not save_flag:
		shutil.rmtree(submit_path)


if __name__ == '__main__': 
	model_name = './pths/max.pth'
	test_img_path = os.path.abspath('../data/val/img')
	submit_path = './submit'
	eval_model(model_name, test_img_path, submit_path)
