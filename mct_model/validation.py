import os, torch
import numpy as np
import torch.nn as nn

from tqdm       import tqdm
from utils      import init_path
from utils.data import Dataset
from modeling   import build_model
from losses     import build_loss
from config     import cfg
from sklearn.metrics import average_precision_score
init_path()

DEVICE     = cfg.DEVICE.TYPE
GPU        = cfg.DEVICE.GPU
WEIGHT     = cfg.MCT.WEIGHT
VALID_PATH = cfg.PATH.VALID_PATH
RW         = cfg.MCT.RW


device = torch.device(DEVICE + ':' + str(GPU))
checkpoint = torch.load(WEIGHT, map_location=device)
model = build_model(cfg, device)
model.load_state_dict(checkpoint)
criterion = build_loss(device)
model.eval()

easy_file = "mtmc_easy_binary_multicam.txt"
hard_file = "mtmc_hard_binary_multicam.txt"
valid_tracklet_file = os.path.join(VALID_PATH, "gt_features.txt")
easy_valid_file = os.path.join(cfg.PATH.VALID_PATH, easy_file)
hard_valid_file = os.path.join(cfg.PATH.VALID_PATH, hard_file)
easy_valid_dataset = Dataset(valid_tracklet_file, easy_valid_file, hard_valid_file, "easy")
hard_valid_dataset = Dataset(valid_tracklet_file, easy_valid_file, hard_valid_file, "hard")

bce = nn.BCELoss()

def validation(model, valid_dataset):
    ap_list = list()
    loss_list = list()
    with torch.no_grad():
        for data, target, cams_target in valid_dataset.prepare_data():
            count = 0.
            data, target, cams_target = data.to(device), target.to(device), cams_target.to(device)
            A = model(data)
            
            if RW:
                preds = model.random_walk(A)
            else:
                preds = A[0][1:]
            # print (preds, target)
            # preds = (preds - preds.mean())
            # preds = preds * 100
            # preds = torch.sigmoid(preds)
            loss = bce(preds, target)
            copy_preds = preds.clone()
            copy_preds = copy_preds.cpu().numpy()
            target = target.cpu().numpy()
            ap = average_precision_score(target, copy_preds)
            ap_list.append(ap * 100.)
            loss_list.append(loss.cpu().item())

    _map = np.array(ap_list).mean()
    avg_loss = np.array(loss_list).mean()
    return _map, avg_loss

def cosine_validation(dataset):
    ap_list = list()
    loss_list = list()
    for data, target, cam_label in dataset.prepare_data():
        count = 0.
        gallery = data[1:]
        query = data[0].view(1, -1)
        cos = (query @ gallery.T) / (torch.norm(query, p=2, dim=1) * torch.norm(gallery, p=2, dim=1))
        preds = cos[0]
        preds = preds.clamp(0, 1)
        loss = bce(preds, target)
        copy_preds = preds.clone()
        copy_preds = copy_preds.cpu().numpy()
        target = target.cpu().numpy()
        ap = average_precision_score(target, preds)
        ap_list.append(ap * 100.)
        loss_list.append(loss.cpu().item())

    _map = np.array(ap_list).mean()
    avg_loss = np.array(loss_list).mean()

    return _map, avg_loss

easy_map, easy_loss = validation(model, easy_valid_dataset)
hard_map, hard_loss = validation(model, hard_valid_dataset)
# easy_map, easy_loss = cosine_validation(easy_valid_dataset)
# hard_map, hard_loss = cosine_validation(hard_valid_dataset)
print("Easy_Map={:.2f}%, Easy_Loss={:.4f}, Hard_Map={:.2f}%, Hard_Loss={:.4f}".format(easy_map, easy_loss, hard_map, hard_loss))
