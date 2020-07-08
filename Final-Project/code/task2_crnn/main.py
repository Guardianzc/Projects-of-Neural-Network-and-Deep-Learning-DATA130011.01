'''
@Author: your name
@Date: 2020-06-17 13:06:18
@LastEditTime: 2020-06-19 15:13:45
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /final/task2_crnn/main.py
'''
import argparse
import os
from Trainer import Solver
from dataLoader import get_loader
from torch.backends import cudnn
import random
from torch.utils.data import sampler


def main(config):

    cudnn.benchmark = True  # to improve the efficiency

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    # lr = random.random()*0.0005 + 0.0000005
    # augmentation_prob = random.random()*0.7
    # epoch = random.choice([100, 150, 200, 250])
    # decay_ratio = random.random()*0.8
    # decay_epoch = int(epoch*decay_ratio)
    #
    # config.augmentation_prob = augmentation_prob
    # config.num_epochs = epoch
    # config.lr = lr
    # config.num_epochs_decay = decay_epoch

    print(config)

    # Notice the difference between these loaders
    train_loader = get_loader(config = config,
                              image_path=config.train_path,
                              crop_size=config.crop_size,
                              batch_size=config.batch_size,
                              sampler = sampler.SubsetRandomSampler(range(0,100000)),
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(config = config,
                              image_path=config.valid_path,
                              crop_size=config.crop_size,
                              batch_size=config.batch_size,
                              sampler = sampler.SubsetRandomSampler(range(100000,103943)),
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0.)

    solver = Solver(config, train_loader, valid_loader)
    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'val':
        solver.val()
    else:
        solver.detect()
    # todo: change the test method and write the save prediction function


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--crop_size', type=int, default=120)    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--img_H', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_epochs_decay', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2*128)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='val')
    parser.add_argument('--model_type', type=str, default='CRNN')
    parser.add_argument('--model_path', type=str, default='./task2_crnn/models')
    parser.add_argument('--train_path', type=str, default='../data/train/')
    parser.add_argument('--valid_path', type=str, default='../data/test/')
    parser.add_argument('--result_path', type=str, default='../result/')
    parser.add_argument('--text_path', type=str, default='../text/')
    parser.add_argument('--image_root', type=str, default='./data/result/image/')
    parser.add_argument('--landmark_root', type=str, default='./data/result/gt/')
    parser.add_argument('--lstm_hidden', type=int, default=256)
    # todo: validate image num in these folders

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()   # return a namespace, use the parameters by config.image_size
    main(config)
