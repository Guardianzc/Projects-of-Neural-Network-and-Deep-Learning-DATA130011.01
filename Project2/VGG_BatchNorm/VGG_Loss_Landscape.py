import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display
import pickle as pkl

from models.vgg import VGG_A, VGG_A_BatchNorm

from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 1
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.

device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:{}".format(2) if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))


# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
'''
for X,y in train_loader:
    ## --------------------
    # Add code as needed
    #
    #
    #
    #
    ## --------------------
    break
'''


# This function is used to calculate the accuracy of model classification
def get_accuracy(output, labels):
    ## --------------------
    # Add code as needed
    acc = 0
    _, predicted = torch.max(output.data, 1)
    acc += predicted.eq(labels.data).cpu().sum()  
    return acc
    #
    #
    ## --------------------
    pass

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train(True)

        loss_list = []  # use this to record the loss value of each step
        grad_list = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve
        train_accuracy_curve[epoch] = 0
        val_accuracy_curve[epoch] = 0
        length = 0
        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            ## --------------------
            # Add your code
            loss.backward()
            grad = model.classifier[-1].weight.grad.clone().cpu()
            loss_list.append(loss)
            grads.append(grad)
            optimizer.step()
            train_accuracy_curve[epoch] += get_accuracy(prediction, y)
            length += x.size(0)
            ## --------------------

        losses_list += loss_list
        

        train_accuracy_curve[epoch] = int(train_accuracy_curve[epoch]) / length

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        ## --------------------
        # Add code as needed
        model.train(False)
        model.eval()
        length = 0
        for data in val_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            length += x.size(0)
            val_accuracy_curve[epoch] += get_accuracy(prediction, y)
        val_accuracy_curve[epoch] = int(val_accuracy_curve[epoch]) / length
        #
        #
        #
        ## --------------------

    return losses_list, grads, train_accuracy_curve, val_accuracy_curve


# Train your model
# feel free to modify
epo = 20
loss_save_path = ''
grad_save_path = ''

set_random_seeds(seed_value=2020, device=device)
model = VGG_A()
lrs = [1e-3, 2e-3, 1e-4, 5e-4]
loss_list = []
train_accuracy_curve_list = []
val_accuracy_curve_list = []

for lr in lrs:
    set_random_seeds(seed_value=2020, device=device)
    model = VGG_A_BatchNorm()
    print('At learning rate', str(lr))
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    loss, grads, train_accuracy_curve, val_accuracy_curve = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    loss_name = 'loss' + str(lr) + '.npy'
    grad_name = 'gradBNZC' + str(lr) + '.pkl'
    train_name = 'train' + str(lr) + '.npy'
    val_name = 'val' + str(lr) + '.npy'
    #np.save(os.path.join(loss_save_path, loss_name), loss)
    with open(grad_name,'wb') as f:
        pkl.dump(grads,f)
    np.save(os.path.join(loss_save_path, train_name), train_accuracy_curve)
    np.save(os.path.join(grad_save_path, val_name), val_accuracy_curve)
    loss_list.append(loss)
    train_accuracy_curve_list.append(train_accuracy_curve)
    val_accuracy_curve_list.append(val_accuracy_curve)

for lr in lrs:
    set_random_seeds(seed_value=2020, device=device)
    model = VGG_A()
    print('At learning rate', str(lr))
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    loss, grads, train_accuracy_curve, val_accuracy_curve = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    loss_name = 'loss' + str(lr) + '.npy'
    grad_name = 'gradZC' + str(lr) + '.pkl'
    train_name = 'train' + str(lr) + '.npy'
    val_name = 'val' + str(lr) + '.npy'
    np.save(os.path.join(loss_save_path, loss_name), loss)
    with open(grad_name,'wb') as f:
        pkl.dump(grads,f)
    np.save(os.path.join(loss_save_path, train_name), train_accuracy_curve)
    np.save(os.path.join(grad_save_path, val_name), val_accuracy_curve)
    loss_list.append(loss)
    train_accuracy_curve_list.append(train_accuracy_curve)
    val_accuracy_curve_list.append(val_accuracy_curve)
# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
min_curve = []
max_curve = []
## --------------------
# Add your code
# max_curve = np.max(loss_list,axis = 0)
# min_curve = np.min(loss_list,axis = 0)

# np.save('max.npy', max_curve)
# np.save('min.npy', min_curve)
#
#
#
## --------------------

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape():
    ## --------------------
    # Add your code
    #
    #
    #
    #
    ## --------------------
    pass