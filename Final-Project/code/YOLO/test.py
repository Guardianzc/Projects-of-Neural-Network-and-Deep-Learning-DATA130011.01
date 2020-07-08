from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *


import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import sampler

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle = True, num_workers=8, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    f = 0
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs1 = model(imgs)
            outputs = non_max_suppression(outputs1, conf_thres=conf_thres, nms_thres=nms_thres)

        batch_metrics = get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        sample_metrics += batch_metrics
        t,_,_ = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        P = sum(t)/targets.shape[0]
        R = sum(t)/sum([outputs1[tt].shape[0] for tt in range(1)])
        f+=(2*P*R)/(P+R)
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class,f/len(dataset)
def changeBox(box):
    x1 = box[0]-box[2]/2
    x2 = box[0]+box[2]/2
    y1 = box[1]-box[3]/2
    y2 = box[1]+box[3]/2
    return [x1,y1,x2,y2]
def evaluate_IOU(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size=1):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(7500,len(dataset))), num_workers=8, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    f = []
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs1 = model(imgs)
            outputs = non_max_suppression(outputs1, conf_thres=conf_thres, nms_thres=nms_thres)
        targets = targets[:,2:6].numpy().tolist()
        targets = [changeBox(box) for box in targets]
        if outputs[0] != None:
            outputs = outputs[0][:4].numpy().tolist()
            outputs = [changeBox(box) for box in outputs]
            T = len(outputs)
            G = len(targets)
            match = overlap(outputs,targets)
            P = match/T+0.00000001
            R = match/G+0.00000001
            f.append((2*P*R)/(P+R))
    return sum(f)/len(f)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_40.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/luo.name", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    valid_path = '../data/val'
    class_names = ['Latin','Arabic','Chinese','Japanese','Korean','Bangla', 'Hindi','Symbols','None','Mixed']

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    f = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        batch_size=opt.batch_size,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        # batch_size=1,
    )

    # print("Average Precisions:")
    # for i, c in enumerate(ap_class):
    #     print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
    print(f)
    # print(f"mAP: {AP.mean()}")
