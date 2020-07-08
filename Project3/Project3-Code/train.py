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
    
def get_loader(config, batch_size, sampler, num_workers=2, mode='train'):
    """Builds and returns Dataloader."""
    dataset = ModelNetDataset(config.root, config.data_list)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
 
    return data_loader

def get_acc(x, y):
    acc = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            acc += 1
    return acc
class Trainer(object):
    def __init__(self, config, train_loader, valid_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
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

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.net.zero_grad()

    def build_model(self):
        if self.model_type == 'cls_3d':
            self.net = cls_3d()
        elif self.model_type == 'cls_3d_BN':
            self.net = cls_3d_BN()
        else:
            self.net = PointNetCls(40)
        """Build generator and discriminator."""
        self.optimizer = optim.Adam(list(self.net.parameters()), self.lr, (self.beta1, self.beta2))
        self.net.to(self.device)

        classname = self.net.__class__.__name__
        if classname.find('Conv') != -1:
            self.net.weight.data.normal_(0.0, 0.02)
        self.optimizer = optim.Adam(list(self.net.parameters()), self.lr, (self.beta1, self.beta2))
        self.scheduler = StepLR(self.optimizer, step_size = self.num_epochs_decay, gamma=0.5)
    
    def train(self):
        net_path = os.path.join(self.model_path, '%s-%d-%.4f-%d.pkl' % (self.model_type,self.num_epochs,self.lr,self.num_epochs_decay))
        total_loss = []
        total_acc = []
        best_acc = 0
        for epoch in range(self.num_epochs):
            self.net.train(True)
            epoch_loss = 0
            length = 0
            num_train = len(self.train_loader)
            data_length = 0
    
            for (i, data) in enumerate(self.train_loader):
                points = data['points']
                labels = data['label'].squeeze().long()
                points = points.to(self.device)
                labels = labels.to(self.device)

                pred = self.net(points)
                loss = self.criterion(pred, labels)
                epoch_loss += loss.item()
                self.reset_grad()
                loss.backward()
                self.optimizer.step()
                progress_bar(50,loss.item(),best_acc,best_acc,self.lr,i+epoch*len(self.train_loader),self.num_epochs*len(self.train_loader))
            total_loss.append(epoch_loss)
            self.scheduler.step()
            
            if (epoch+1)%10==1:
                # ===================================== Validation ====================================#
                self.net.train(False)  # control the train_phase from the model
                self.net.eval()  # the model will automatically fix BN and Dropout
                acc = 0.
                for (i, data) in enumerate(self.valid_loader):
                    points = data['points']
                    labels = data['label'].squeeze().long()
                    points = points.to(self.device)
                    labels = labels.to(self.device)
                    pred = self.net(points)
                    pred_label = pred.max(1)[1]
                    acc += get_acc(pred_label, labels)
                
                acc = acc/(len(self.valid_loader) * self.batch_size)
                total_acc.append(acc)
                # Save Best Net model
                if acc > best_acc: 
                    best_acc = acc
                    best_net_path = os.path.join(self.model_path, '%s-%d-%.4f-%d_best.pkl' % (self.model_type,self.num_epochs,self.lr,self.num_epochs_decay))
                    print('Best %s model score : %.4f\n' % (self.model_type, best_acc))
                    state_dict = self.net.state_dict()
                    torch.save(state_dict, best_net_path)
            state_dict = self.net.state_dict()
            torch.save(state_dict, net_path)

        loss_file=open( net_path.rstrip('.pkl') + '_loss.pickle','wb')
        pickle.dump(total_loss, loss_file)
        loss_file.close()
        acc_file=open( net_path.rstrip('.pkl') + '_acc.pickle','wb')
        pickle.dump(total_acc, acc_file)
        acc_file.close()

if __name__ == '__main__':
    os.chdir(sys.path[0])
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./')
    parser.add_argument('--data_list', type=str, default='data/modelnet40_ply_hdf5_2048/train_files.txt')
    
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
    train_loader = get_loader(config, config.batch_size, sampler = sampler.SubsetRandomSampler(range(0,2200)), num_workers= config.num_workers)
    valid_loader = get_loader(config, config.batch_size, sampler = sampler.SubsetRandomSampler(range(2200,2460)), num_workers= config.num_workers)
    trainer = Trainer(config, train_loader, valid_loader)
    trainer.train()