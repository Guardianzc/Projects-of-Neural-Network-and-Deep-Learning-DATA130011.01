from torch import nn
import numpy as np
import torch 
import torchvision 
import torchvision.transforms as transforms
import os
import random
import torch.nn.functional as F
from models.network import myNet,leNet5, dropoutMyNet,resNet,my_resNet
import torch.optim as optim
from torch.utils.data import sampler
import sys
from time import time
import pickle as pkl
import argparse
from utils.test import test
from utils.train import train
from utils.loadDataset import vison_dataset_loader,mycollate_fn
from tensorboardX import SummaryWriter
import warnings
import tqdm
warnings.filterwarnings('always')

## Constants (parameters) initialization
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("-gpu", "--GPUid", type=int,choices=[0, 1, 2,3], default=0, help="Choose which GPU")
parser.add_argument("-decay", "--weight_decay", type=float, default=0.01, help="l2 weight decay")
parser.add_argument("-w", "--number_workers", type=int, default=16, help="load data number of workers")
parser.add_argument("-bs", "--batch_size", type=int, default=4, help="batch_size")
parser.add_argument("-es", "--epoch_size", type=int, default=30, help="epoch_size")
parser.add_argument("-lr", "--lr", type=float, default=1e-7, help="learning rate")
parser.add_argument("-mom", "--momentum", type=float, default=0.9, help="learning rate")
parser.add_argument("-drop", "--dropout", type=float, default=0.5, help="dropout rate")
parser.add_argument("-s_gamma", "--scheduler_gamma", type=float, default=1, help="dropout rate")
parser.add_argument("-s_stepsize", "--scheduler_stepsize", type=float, default=25, help="dropout rate")
parser.add_argument("-net", "--network", default='resNet', choices=['myNet','leNet5','dropoutMyNet','resNet','my_resNet'], help="choose wich net")
parser.add_argument("-mul", "--mul_gpu", type=bool, default=False, help="multiple gpu")
parser.add_argument("-loss", "--loss", default='MSE', help="choose loss")
parser.add_argument("-opt", "--optimizer", default='SGD', help="choose loss")

'''
python main.py -gpu 3 -decay 0 -w 32 -bs 512 -es 70 -lr 0.01 -mom 0.9 -drop 0.5 -s_gamma 0.5 --scheduler_stepsize 15 -net my_resNet  -loss crossEntropy
python main.py -gpu 3 -decay 0 -w 32 -bs 512 -es 70 -lr 0.01 -mom 0.9 -drop 0.5 -s_gamma 0.5 --scheduler_stepsize 15 -net my_resNet -loss crossEntropy --optimizer Adam
'''
'''
python -m torch.distributed.launch --master_port 29502 main.py -decay 0 -w 32 -bs 1024 -es 70 -lr 0.01 -mom 0.9 -drop 0.5 -s_gamma 0.5 --scheduler_stepsize 15 -net resNet -mul True -loss crossEntropy --optimizer SGD


'''


args = parser.parse_args()

gpuid = args.GPUid
num_workers = args.number_workers
batch_size = args.batch_size
epoch_size = args.epoch_size
lr = args.lr
momentum = args.momentum
weight_decay = args.weight_decay
dropout = args.dropout
network = args.network
mul_gpu = args.mul_gpu
scheduler_gamma = args.scheduler_gamma
scheduler_stepsize = args.scheduler_stepsize
loss = args.loss
opt = args.optimizer
class_num = 10
NUM_TRAIN = 500
import gc
# def setup_seed(seed):
#      torch.manual_seed(seed)
#      torch.cuda.manual_seed_all(seed)
#      np.random.seed(seed)
#      random.seed(seed)
#      torch.backends.cudnn.deterministic = True
# # 设置随机数种子
# setup_seed(20)
def dataset_in():
    for i in tqdm.tqdm(range(8)):
        if i == 0:
            allData = torch.load('splitData/trainData'+str(i+1)+".pkl")
        else:
            dataset = torch.load('splitData/trainData'+str(i+1)+".pkl")
            allData.data = np.concatenate([allData.data,dataset.data])
            allData.targets = allData.targets+dataset.targets
            del dataset
            gc.collect()
    return allData
#load dataset
trans = transforms.ToTensor()
# dataset = vison_dataset_loader('data/train',0,500,transform=trans)
dataset = dataset_in()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,collate_fn=mycollate_fn)

# Make sure you are using the right device.
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


#define network
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False,num_classes=10)
if mul_gpu:
    torch.distributed.init_process_group(backend="nccl")
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model)
else:
    model = model.cuda(gpuid)
# 定义Summary_Writer
writer = SummaryWriter('/root/result')

# choose loss funxtion
if loss =='crossEntropy':
    criterion = nn.CrossEntropyLoss()
elif loss =='MSE':
    criterion = nn.MSELoss()
#choose optimizer
if opt =='SGD':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
elif opt =='Adam':
    optimizer = optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
elif opt =='Adagrad':
    optimizer = optim.Adagrad(model.parameters(),lr=lr,weight_decay=weight_decay)
elif opt =='RMSprop':
    optimizer = optim.RMSprop(model.parameters(),lr=lr,weight_decay=weight_decay,momentum=momentum)

# sheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_stepsize,gamma=scheduler_gamma)

# optimizer = optim.Adam(net.parameters(),lr = lr)

#start training

start = time()
train(epoch_size,batch_size,data_loader,optimizer,gpuid,mul_gpu,model,NUM_TRAIN)
end = time()

print('训练耗时：'+str(start-end))
