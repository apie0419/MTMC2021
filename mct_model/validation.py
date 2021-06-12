import os, torch
import numpy as np
import torch.nn as nn

from tqdm       import tqdm
from utils      import init_path
from utils.data import Dataset
from modeling   import build_model
from losses     import build_loss
from config     import cfg

init_path()

DEVICE     = cfg.DEVICE.TYPE
GPU        = cfg.DEVICE.GPU
WEIGHT     = cfg.MCT.WEIGHT
VALID_PATH = cfg.PATH.VALID_PATH


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
easy_valid_dataset = Dataset(valid_tracklet_file, easy_valid_file, hard_valid_file, "easy", training=False)
hard_valid_dataset = Dataset(valid_tracklet_file, easy_valid_file, hard_valid_file, "hard", training=False)

pbar = tqdm(total=len(hard_valid_dataset))

bce = nn.BCELoss()

ap_list, loss_list = list(), list()
with torch.no_grad():
    
    for data, target, cam_label in hard_valid_dataset.prepare_data():
        count = 0.
        if data == None or target == None:
            dataset_len -= 1
            pbar.total -= 1
            pbar.refresh()
            continue
        gallery = data[1:]
        query = data[0].view(1, -1)
        
        ## Cosine
        cos = (query @ gallery.T) / (torch.norm(query, p=2, dim=1) * torch.norm(gallery, p=2, dim=1))
        preds = cos[0]
        preds = preds.to(device)
        sort_preds = torch.argsort(preds, descending=True)
        bce_target = torch.zeros(preds.size(0)).to(device)
        for i in range(target.size(0)):
            bce_target[target[i]] = 1.
        loss = bce(preds, bce_target)
        target_list = target.numpy().tolist()
        for i in range(sort_preds.size(0)):
            if len(target_list) == 0:
                break
            t = sort_preds[i]
            if t in target_list:
                target_list.remove(t)
                point = float((target.size(0) - len(target_list))) / float(i + 1)
                count += point
        ap_list.append(count / target.size(0) * 100.)
        loss_list.append(loss.cpu().item())

        # count = 0.
        # data, target, cam_label = data.to(device), target.to(device), cam_label.to(device)
        # preds = model(data)
        # bce_target = torch.zeros(preds.size(0)).to(device)
        # for i in range(target.size(0)):
        #     bce_target[target[i]] = 1.
        # loss = bce(preds, bce_target)
        # sort_preds = torch.argsort(preds, descending=True)
        # target_list = target.cpu().numpy().tolist()
        # for i in range(sort_preds.size(0)):
        #     if len(target_list) == 0:
        #         break
        #     t = sort_preds[i]
        #     if t in target_list:
        #         target_list.remove(t)
        #         point = float((target.size(0) - len(target_list))) / float(i + 1)
        #         count += point
        # ap_list.append(count / target.size(0) * 100.)
        # loss_list.append(loss.cpu().item())

pbar.close()
print ("Map:{:.2f}%, Loss:{:.4f}".format(np.array(ap_list).mean(), np.array(loss_list).mean()))