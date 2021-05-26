import os, torch
import numpy as np

from tqdm       import tqdm
from utils      import init_path
from utils.data import Dataset
from modeling   import build_model
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
model.eval()

_type = "merge"
tracklets_file = os.path.join(VALID_PATH, "gt_features.txt")
easy_file = os.path.join(VALID_PATH, "mtmc_easy_binary.txt")
hard_file = os.path.join(VALID_PATH, "mtmc_hard_binary.txt")
dataset = Dataset(tracklets_file, easy_file, hard_file, _type, training=False)

pbar = tqdm(total=len(dataset))

map_list = list()
with torch.no_grad():
    
    for data, target, cam_label in dataset.prepare_data():
        count = 0.
        if data == None or target == None:
            dataset_len -= 1
            pbar.total -= 1
            pbar.refresh()
            continue
        gallery = data[1:]
        query = data[0].view(1, -1)
        
        ## Cosine
        # cos = (query @ gallery.T) / (torch.norm(query, p=2, dim=1) * torch.norm(gallery, p=2, dim=1))
        # sort_preds = torch.argsort(cos, descending=True)[0]
        # # print (sort_preds)
        # target_list = target.numpy().tolist()
        # for i in range(sort_preds.size(0)):
        #     if len(target_list) == 0:
        #         break
        #     t = sort_preds[i]
        #     if t in target_list:
        #         target_list.remove(t)
        #         point = float((target.size(0) - len(target_list))) / float(i + 1)
        #         count += point
        # map_list.append(count / target.size(0) * 100.)
        # pbar.update()

        ## Model
        data, target = data.to(device), target
        preds = model(data)
        sort_preds = torch.argsort(preds, descending=True)
        target_list = target.numpy().tolist()
        for i in range(sort_preds.size(0)):
            if len(target_list) == 0:
                break
            t = sort_preds[i]
            if t in target_list:
                target_list.remove(t)
                point = float((target.size(0) - len(target_list))) / float(i + 1)
                count += point
        
        map_list.append(count / target.size(0) * 100.)
        pbar.update()

pbar.close()
print ("Map:{:.2f}%".format(np.array(map_list).mean()))