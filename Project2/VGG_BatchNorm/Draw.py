import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch
import pickle as pkl

def tensor2numpy(t):
    l = []
    for ten in  t:
        l.append(ten)
    return l

def loss_landscape():
    names = ['./NO_BN/loss0.0001.npy', './NO_BN/loss0.0005.npy','./NO_BN/loss0.001.npy', './NO_BN/loss0.002.npy']
    losses = []
    for i in range(len(names)):
        loss = np.load(names[i], allow_pickle=True) 
        losses.append(tensor2numpy(loss))
    losses = np.array(losses)
    max_loss = np.max(losses, axis = 0)
    min_loss = np.min(losses, axis = 0)
    np.save('max.npy', max_loss)
    np.save('min.npy', min_loss)
    NO_BN_max = np.load('./NO_BN/max.npy', allow_pickle=True)
    NO_BN_min = np.load('./NO_BN/min.npy', allow_pickle=True)

    interval = 25
    x = range(1,len(NO_BN_max), interval)
    BN_max = np.load('./BN/max.npy', allow_pickle=True)
    BN_min = np.load('./BN/min.npy', allow_pickle=True)
    NO_BN_max = NO_BN_max[::interval]
    NO_BN_min = NO_BN_min[::interval]
    BN_max = tensor2numpy(BN_max)[::interval]
    BN_min = tensor2numpy(BN_min)[::interval]
    plt.fill_between(x, NO_BN_max, NO_BN_min, facecolor="royalblue", label = 'Standard VGG')
    plt.title('Loss Lanscape') 
    plt.xlabel('Steps')
    plt.ylabel('Loss Lanscape') 
    plt.legend()
    plt.fill_between(x, BN_max, BN_min, facecolor="lightseagreen",  label = 'Standard VGG + BatchNorm')
    plt.legend()
    plt.show()


def VGG_Grad_Pred():
    interval = 40

    names =  ['./BN/grad_BN0.pkl', './BN/grad_BN1.pkl','./BN/grad_BN2.pkl']
    names =  ['./BN/gradBNZC0.001.pkl', './BN/gradBNZC0.002.pkl','./BN/gradBNZC0.0001.pkl','./BN/gradBNZC0.0005.pkl']
    all_BN_first = []

    for i in range(len(names)):
        BN_first = []
        grad_BN = pkl.load(open(names[i],'rb'))
        #grad_BN = np.load(names[i], allow_pickle=True) 
        l = len(grad_BN)
        for i in range(l-1):
            BN_first.append(torch.norm(grad_BN[i] - grad_BN[i+1]))
        all_BN_first.append(BN_first)
    all_BN_first = np.array(all_BN_first)
    max_BN = np.max(all_BN_first, axis = 0)[::interval]
    min_BN = np.min(all_BN_first, axis = 0)[::interval]
    #print('max_train = ', max(max_train))
    x = range(1,len(BN_first)+1, interval)

    names =  ['./BN/grad_0.pkl', './BN/grad_1.pkl','./BN/grad_2.pkl','./BN/gradZC0.001.pkl']
    #names =  ['./BN/gradZC0.001.pkl','./BN/gradZC0.002.pkl','./BN/gradZC0.0001.pkl','./BN/gradZC0.0005.pkl']

    all_NO_BN_first = []
    for i in range(len(names)):
        NO_BN_first = []
        grad_NO_BN_first = pkl.load(open(names[i],'rb')) 
        l = len(grad_NO_BN_first)
        for i in range(l-1):
            NO_BN_first.append(torch.norm(grad_NO_BN_first[i] - grad_NO_BN_first[i+1]))
        all_NO_BN_first.append(NO_BN_first)

    all_NO_BN_first = np.array(all_NO_BN_first)
    max_NO_BN = np.max(all_NO_BN_first, axis = 0)[::interval]
    min_NO_BN = np.min(all_NO_BN_first, axis = 0)[::interval]
    #print('max_eval = ', max(max_eval))
    plt.fill_between(x, max_NO_BN, min_NO_BN, facecolor="lightseagreen",  label = 'Standard VGG')

    plt.title('Gradient Predictiveness') 
    plt.xlabel('Steps')
    plt.ylabel('l2-distance') 
    plt.ylim(0,4)
    plt.legend()
    plt.fill_between(x, max_BN, min_BN, facecolor="royalblue", label = 'Standard VGG + BatchNorm')
    plt.legend()
    plt.show()


def VGG_Beta_Smooth():
    interval = 40

    names =  ['./BN/grad_BN0.pkl', './BN/grad_BN1.pkl','./BN/grad_BN2.pkl']
    names =  ['./BN/gradBNZC0.001.pkl', './BN/gradBNZC0.002.pkl','./BN/gradBNZC0.0001.pkl','./BN/gradBNZC0.0005.pkl']
    all_BN_first = []

    for i in range(len(names)):
        BN_first = []
        grad_BN = pkl.load(open(names[i],'rb'))
        #grad_BN = np.load(names[i], allow_pickle=True) 
        l = len(grad_BN)
        for i in range(1,l-1):
            BN_first.append(torch.norm(2 * grad_BN[i] - grad_BN[i+1] - - grad_BN[i-1]))
        all_BN_first.append(BN_first)
    all_BN_first = np.array(all_BN_first)
    max_BN = np.max(all_BN_first, axis = 0)[::interval]
    min_BN = np.min(all_BN_first, axis = 0)[::interval]
    #print('max_train = ', max(max_train))
    x = range(1,len(BN_first)+1, interval)

    names =  ['./BN/grad_0.pkl', './BN/grad_1.pkl','./BN/grad_2.pkl','./BN/gradZC0.001.pkl']
    #names =  ['./BN/gradZC0.001.pkl','./BN/gradZC0.002.pkl','./BN/gradZC0.0001.pkl','./BN/gradZC0.0005.pkl']
    all_NO_BN_first = []
    for i in range(len(names)):
        NO_BN_first = []
        grad_NO_BN_first = pkl.load(open(names[i],'rb'))
        l = len(grad_NO_BN_first)
        for i in range(1, l-1):
            NO_BN_first.append(torch.norm(2*grad_NO_BN_first[i] - grad_NO_BN_first[i+1] - grad_NO_BN_first[i-1]))
        all_NO_BN_first.append(NO_BN_first)

    all_NO_BN_first = np.array(all_NO_BN_first)
    max_NO_BN = np.max(all_NO_BN_first, axis = 0)[::interval]
    min_NO_BN = np.min(all_NO_BN_first, axis = 0)[::interval]
    #print('max_eval = ', max(max_eval))
    plt.fill_between(x, max_NO_BN, min_NO_BN, facecolor="lightseagreen",  label = 'Standard VGG')

    plt.title('Beta-smoothness') 
    plt.xlabel('Steps')
    plt.ylabel('Beta-smoothness') 
    plt.ylim(0,10)
    plt.legend()
    plt.fill_between(x, max_BN, min_BN, facecolor="royalblue", label = 'Standard VGG + BatchNorm')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    os.chdir(sys.path[0])
    loss_landscape()
    VGG_Grad_Pred()
    VGG_Beta_Smooth()
