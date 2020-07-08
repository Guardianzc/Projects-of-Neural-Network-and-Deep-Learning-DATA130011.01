import torch
import torchvision 
import torchvision.transforms as transforms
import argparse
import os
import sys
from torch.backends import cudnn
from ResNet18 import ResNet_Transform
from Trainer import Solver


def main(config):
    cudnn.benchmark = True
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)

    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    transform_train = transforms.Compose( [transforms.RandomCrop(32, padding=4),  
                                     transforms.RandomHorizontalFlip(), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2) 
    
    transform_test = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2)
    
    solver = Solver(config, trainloader, testloader)

    if config.mode == 'train':
        solver.train()
    else:
        solver.test()

if __name__ == "__main__":
    os.chdir(sys.path[0])
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default= 300)
    parser.add_argument('--num_epochs_decay', default= 200)
    parser.add_argument('--batch_size', default= 16)
    parser.add_argument('--lr', default= 0.001)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--model_type', type=str, default='Res_18')
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--result_path', default='./result')
    parser.add_argument('--model_path', type=str, default='../models')
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=10)    
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.9)      # momentum2 in Adam    
    config = parser.parse_args()

    main(config)