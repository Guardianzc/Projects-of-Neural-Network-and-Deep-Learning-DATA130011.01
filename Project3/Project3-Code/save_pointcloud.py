from sample_codes.dataset import ModelNetDataset
import os
import sys
import argparse
from torch.utils import data
def get_loader(config, batch_size, num_workers=2, mode='train'):
    """Builds and returns Dataloader."""
    dataset = ModelNetDataset(config.root, config.data_list)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)
 
    return data_loader

def store(config):
    # store the pointcloud
    data_loader = get_loader(config, config.batch_size, config.num_workers)

    for (i, data) in enumerate(data_loader):
        points = data['points'][-1].numpy().T
        print(data['label'])
        with open('test.obj','w') as f:
            for i in range(points.shape[0]):
                x, y, z = points[i]
                # format of .obj files
                lines = 'v ' + str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(x/2+0.5) + ' ' + str(y/2+0.5) + ' ' + str(z/2+0.5) + '\n'
                f.write(lines)
        print(points[0])
        break

if __name__ == '__main__':
    os.chdir(sys.path[0])
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./')
    parser.add_argument('--data_list', type=str, default='data/modelnet40_ply_hdf5_2048/train_files.txt')

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)
    config = parser.parse_args()   # return a namespace, use the parameters by config.image_size
    store(config)