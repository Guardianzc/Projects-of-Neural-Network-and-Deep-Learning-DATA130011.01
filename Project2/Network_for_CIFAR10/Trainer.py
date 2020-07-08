import os
import numpy as np
import time
import datetime
import torch
import torchvision
import torch.nn as nn
import csv
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from model import ResNet_Transform, ResNet_pooling, VGG_A, VGG_A_BN

class Solver(object):
    def __init__(self, config, train_loader, test_loader):
        # Data Loader
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # Models
        self.net = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = nn.MSELoss()

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == "Res_18":
            self.net = ResNet_Transform(False)
        elif self.model_type == "Res_18_pooling":
            self.net = ResNet_pooling(True)
        self.optimizer = optim.Adam(list(self.net.parameters()), self.lr, (self.beta1, self.beta2), weight_decay = 1e-5)
        #self.optimizer = optim.SGD(list(self.net.parameters()) , lr = self.lr, weight_decay = 1e-5)
        #self.optimizer = optim.RMSprop(list(self.net.parameters()), lr=self.lr, alpha=0.9, weight_decay = 1e-5)
        self.net.to(self.device)

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

    def compute_accuracy(self, SR, GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu() == SR_flat.data.cpu()

    def train(self):
        """Train encoder, generator and discriminator."""

    # ====================================== Training =========================================== #
    # =========================================================================================== #

        net_path = os.path.join(self.model_path, '%s-%d-%.4f-%d.pkl' % (self.model_type,
                                                                                self.num_epochs,
                                                                                self.lr,
                                                                                self.num_epochs_decay,
                                                                                ))
        # when you save the model, also save the hyper parameters
        print(net_path)

        # Net Train
        if os.path.isfile(net_path):
            # Load the pretrained Encoder
            self.net.load_state_dict(torch.load(net_path))
            print('%s is Successfully Loaded from %s'%(self.model_type, net_path))
            torch.save(self.net.state_dict(), 'model.pkl')
        else:
            # Train for Encoder
            lr = self.lr
            best_acc = 0
            for epoch in range(self.num_epochs):

                self.net.train(True)
                epoch_loss = 0
                
                acc = 0.	# Accuracy
                length = 0
                num_train = len(self.train_loader) * self.batch_size
                for i, data in enumerate(self.train_loader):
                    images, labels = data
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    output = self.net(images)
                    label = torch.zeros(output.size()).to(self.device)
                    for j, index in enumerate(labels):
                        label[j][index] = 1
                    print(output.shape)
                    print(labels.shape)
                    loss = self.criterion(output, label)
                    
                    epoch_loss += loss.item()

                    # Backprop + optimize
                    self.reset_grad()  # zero the gradient buffers, not zero the parameters
                    loss.backward()
                    self.optimizer.step()  

                    _, predicted = torch.max(output.data, 1)

                    length += images.size(0)
                    acc += predicted.eq(labels.data).cpu().sum()   
                    if length % 100 == 0:
                        print('Epoch [%d/%d], Iteration [%d/%d], Loss: %.4f, \n[Training] ACC: %.4f' % (epoch+1, self.num_epochs, length, num_train, epoch_loss/(i+1), int(acc)/length))
                print(acc, length)
                acc = int(acc) / length
                print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f' % (epoch+1, self.num_epochs, epoch_loss/(i+1), acc))                   
                with open('log.txt','a+') as f:
                    line = 'Epoch [' + str(epoch+1) + '/' + str(self.num_epochs) + '], Loss: ' + str(epoch_loss/(i+1)) + ', \n[Training] Acc: ' + str(acc) + '\n'
                    f.write(line)
                    f.flush()

                # Decay learning rate
                if (epoch+1) > (self.num_epochs - self.num_epochs_decay):  # only decay from the epoch: num_epochs_decay
                    lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print('Decay learning rate to lr: {}.'.format(lr))

                self.net.train(False)
                self.net.eval()
                acc = 0.	# Accuracy
                correct = 0
                length = 0
                total = 0
                for i, data in enumerate(self.test_loader):
                    images, labels = data
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    output = self.net(images)
                            
                    length += images.size(0)
                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                print(total)
                print('Acc = %.4f' % (int(correct) / total))
                acc = int(correct)/length

                f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
                wr = csv.writer(f)
                wr.writerow([self.model_type, acc, epoch, self.num_epochs_decay,
                                ])
                f.close()

                model_path = 'model.pkl' 
                if acc > best_acc:
                    torch.save(self.net.state_dict(), model_path)
                    best_acc = acc





    def test(self):
        # ===================================== Test ==================================== #
        del self.net
        net_path = './models/model.pkl'
        self.build_model()
        self.net.load_state_dict(torch.load(net_path))
        
        self.net.train(False)
        self.net.eval()

        acc = 0.	# Accuracy
        correct = 0
        length = 0
        total = 0
        for i, data in enumerate(self.test_loader):
            images, labels = data
            images = images.to(self.device)
            labels = labels.to(self.device)
            output = self.net(images)
                    
            length += images.size(0)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
        print(total)
        print('Acc = %.4f' % (int(correct) / total))
        acc = int(acc)/length

        f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow([self.model_type, acc, self.num_epochs, self.num_epochs_decay,
                        ])
        f.close()