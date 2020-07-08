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
    
        # seq2seq
        self.encoder = config.encoder
        self.decoder = config.decoder
    
    def weights_init(self, model):
    # Official init from torch repo.
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def build_model(self):
        """Build generator and discriminator."""
        # self.imgH: High of image
        '''
        self.net = CRNN(self.img_H, self.img_ch, self.nclass, self.lstm_hidden)
        self.optimizer = optim.Adam(list(self.net.parameters()), self.lr, (self.beta1, self.beta2))
        self.net.to(self.device)
        classname = self.net.__class__.__name__
        if classname.find('Conv') != -1:
            self.net.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            self.net.weight.data.normal_(1.0, 0.02)
            self.net.bias.data.fill_(0)
        # self.print_network(self.unet, self.model_type)
        '''
        # seq2seq
         # create crnn/seq2seq/attention network
        encoder = crnn.Encoder(channel_size=3, hidden_size=256)
        # for prediction of an indefinite long sequence
        decoder = crnn.Decoder(hidden_size= 256, output_size=self.nclass, dropout_p=0.1, max_lrngth=16)
        print(encoder)
        print(decoder)
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=self.lr, betas=(0.5, 0.999))
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=self.lr, betas=(0.5, 0.999))
        encoder.apply(utils.weights_init)
        decoder.apply(utils.weights_init)
        if self.encoder:
            print('loading pretrained encoder model from %s' % self.encoder)
            encoder.load_state_dict(torch.load(self.encoder))
        if self.decoder:
            print('loading pretrained encoder model from %s' % self.decoder)
            decoder.load_state_dict(torch.load(self.decoder))
        self.encoder.to(self.device)
        self.decoder.to(self.device)

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
        net_path = os.path.join(self.model_path, '%s-%d-%.4f-%d_final.pkl' % (self.model_type,self.num_epochs,self.lr,self.num_epochs_decay))
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
            length = 0
            for i, (images, GT) in enumerate(self.train_loader):

                images = images.to(self.device)
                # GT = GT.to(self.device)

                SR = self.net(images)
                SR = SR.permute(1,0,2).max(2)[1]
                t, l = self.converter.encode(GT)
                text = []
                for i in range(SR.shape[0]):
                    for j in range(len(SR[i,:])):
                        if SR[i, j] != self.blank:
                            SR[i, j] -= 1
                    decode_text = [list(map(lambda x:self.converter.index[x],SR[i,:].detach().cpu().numpy().tolist())),GT[i]]
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
                    text.append(decode_text)
                acc += get_accuracy(text)
                length += images.shape[0]
                progress_bar(50, 0.0, acc, length,self.lr,i * self.batch_size ,len(self.train_loader) * self.batch_size)
            acc = acc/ (len(self.train_loader) * self.batch_size)
            print(acc)
        else:
            # Train for Encoder
            lr = self.lr
            meanloss = 0
            acc = 0.	# Accuracy
            length = 0
            best_acc = 0
            for epoch in range(self.num_epochs):

                self.net.train(True)
                epoch_loss = 0
                #acc = 0.	# Accuracy
                length = 0
                num_train = len(self.train_loader)
                data_length = 0
                for i, (images, GT) in enumerate(self.train_loader):
                    # GT : Ground Truth
                    images = images.to(self.device)

                    # preds: Result
                    t, l = self.converter.encode(GT)
                    t = t.to(self.device)
                    l = l.to(self.device)


                    '''
                    # CRNN
                    preds = self.net(images)  # [length, batch_size, total_length]
                    '''
                    # s2s                    
                    for encoder_param, decoder_param in zip(encoder.parameters(), decoder.parameters()):
                        encoder_param.requires_grad = True
                        decoder_param.requires_grad = True
                    encoder.train()
                    decoder.train()
                    


                    preds_size = Variable(torch.IntTensor([preds.size(0)] * self.batch_size))
                    if len(l) != self.batch_size:
                        continue
                    loss = self.criterion(preds, t, preds_size, l)
                    # Backprop + optimize
                    self.reset_grad()  # zero the gradient buffers, not zero the parameters
                    loss.backward()
                    self.optimizer.step()
                    
                    # acc += self.compute_accuracy(preds, GT)
                    
                    # compute the acc
                    SR = preds.permute(1,0,2).max(2)[1]
                    text = []
                    for i in range(SR.shape[0]):
                        decode_text = [list(map(lambda x:self.converter.index[x],SR[i,:].detach().cpu().numpy().tolist())),GT[i]]
                        decode_text[0] =''.join([ x if x !='<BLANK>' else '' for x in decode_text[0]])
                        text.append(decode_text)
                    acc += get_accuracy(text)
                    data_length += images.size(0)
                    
                    # Print the log info
                    progress_bar(50,loss.item(),acc,best_acc,self.lr,i+epoch*len(self.train_loader),self.num_epochs*len(self.train_loader))
                final_net = self.net.state_dict()
                net_path = os.path.join(self.model_path, '%s-%d-%.4f-%d_final.pkl' % (self.model_type,self.num_epochs,self.lr,self.num_epochs_decay))
                torch.save(final_net, net_path)

                if (epoch+1)%20==1:
                    # ===================================== Validation ====================================#
                    self.net.train(False)  # control the train_phase from the model
                    self.net.eval()  # the model will automatically fix BN and Dropout
                    acc = 0.
                    for i, (images, GT) in enumerate(self.train_loader):

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
                            text.append(decode_text)
                        acc += get_accuracy(text)
                    acc = acc/len(self.train_loader)
                    # Save Best Net model
                    if acc > best_acc: 
                        best_acc = acc
                        best_epoch = epoch
                        best_net = self.net.state_dict()
                        print('Best %s model score : %.4f' % (self.model_type, best_net_score))
                        torch.save(best_net, net_path)


