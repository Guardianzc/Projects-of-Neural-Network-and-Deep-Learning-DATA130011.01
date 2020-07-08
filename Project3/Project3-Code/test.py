from sample_codes.dataset import ModelNetDataset
import os
import sys
import argparse
import torch
import torch.nn as nn
import pickle
from torch import optim
from torch.utils import data
from sample_codes.model import cls_3d, PointNetCls, cls_3d_BN
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import sampler
def progress_bar(bar_len,loss,best_acc,test_f1,lr,currentNumber, wholeNumber):
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
         (int(currentNumber),int(wholeNumber),bar, '\033[32;1m%s\033[0m' % percents, '%',loss,best_acc,test_f1,lr))
    sys.stdout.flush()
    
def get_loader(config, batch_size, num_workers=2, mode='train'):
    """Builds and returns Dataloader."""
    dataset = ModelNetDataset(config.root, config.data_list)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)
 
    return data_loader

def get_acc(x, y):
    ''' get the acc  '''
    acc = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            acc += 1
    return acc

class Trainer(object):
    def __init__(self, config, test_loader):
        self.test_loader = test_loader
        
        self.net = None
        self.optimizer = None

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        self.criterion = nn.CrossEntropyLoss()
        # Path
        self.model_path = config.model_path

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.build_model()

    def build_model(self):
        '''load the model '''
        if self.model_type == 'cls_3d':
            self.net = cls_3d()
            self.net.load_state_dict(torch.load('./model/cls_3d-200-0.0100-50_best.pkl'))
            print('%s is Successfully Loaded '%(self.model_type))
        
        elif self.model_type == 'cls_3d_BN':
            self.net = cls_3d_BN()
            self.net.load_state_dict(torch.load('./model/cls_3d_BN-200-0.0100-50_best.pkl'))
            print('%s is Successfully Loaded '%(self.model_type))
        else:
            self.net = PointNetCls(40)
            self.net.load_state_dict(torch.load('./model/PointNetCls-200-0.0100-50_best.pkl'))
            print('%s is Successfully Loaded '%(self.model_type))

        self.net.to(self.device)
    
    def test(self):
        loss = []
        total_acc = []
        best_acc = 0
        self.net.train(False)
        self.net.eval()
        epoch_loss = 0
        length = 0
        num_train = len(self.test_loader)
        data_length = 0            
        acc = 0
        for (i, data) in enumerate(self.test_loader):
            points = data['points']
            labels = data['label'].squeeze().long()
            points = points.to(self.device)
            labels = labels.to(self.device)
            pred = self.net(points)
            pred_label = pred.max(1)[1]
            acc += get_acc(pred_label, labels) 
            data_length += points.shape[0]
            progress_bar(50,0,acc,data_length,self.lr, i*self.batch_size ,self.batch_size * len(self.test_loader))
        # return the acc   
        acc = acc/(len(self.test_loader) * self.batch_size)
        print(acc)


if __name__ == '__main__':
    os.chdir(sys.path[0])
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./')
    parser.add_argument('--data_list', type=str, default='data/modelnet40_ply_hdf5_2048/test_files.txt')
    
    parser.add_argument('--model_path', type = str, default='./model')
    parser.add_argument('--model_type', type = str, default='pointnet') # cls_3d_BN, cls_3d, pointnet
    parser.add_argument('--num_epochs', type = int, default=200)
    parser.add_argument('--lr', type = int, default= 0.01)
    parser.add_argument('--num_epochs_decay', type = int, default=50)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    config = parser.parse_args()   # return a namespace, use the parameters by config.image_size
    test_loader = get_loader(config, config.batch_size, num_workers= config.num_workers)
    trainer = Trainer(config, test_loader)
    trainer.test()