import os
import h5py
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class ModelNetDataset(Dataset):

    def __init__(self,
                 root,
                 data_list,):
        super().__init__()

        self.root = root
        self.data_list = data_list
        self.cat = {}
        self.pts = []
        self.labels = []

        # We have 5 files for training and 2 files for testing
        with open(os.path.join(self.root, self.data_list)) as f:
            for file_name in f:
                file_name = file_name.strip()
                data = h5py.File(os.path.join(self.root, file_name), 'r')
                self.pts.append(data['data'])
                self.labels.append(data['label'])
        # Combine model data from all files
        self.pts = np.vstack(self.pts)
        self.labels = np.vstack(self.labels)

    def __getitem__(self, index):

        pts = self.pts[index]
        label = self.labels[index]

        # Put the channel dimension in front for feeding into the network
        pts = pts.transpose(1,0)

        return {
            "points": pts,
            "label": label
        }

    def __len__(self):
        return self.pts.shape[0]