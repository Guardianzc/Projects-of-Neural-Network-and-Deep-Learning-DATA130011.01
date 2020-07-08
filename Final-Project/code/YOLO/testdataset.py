from utils.datasets import *
dataset = ListDataset('data_all/train', augment=False, multiscale=True)
print(dataset[0])
print(1)