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

_type = "easy"
tracklets_file = os.path.join(VALID_PATH, "gt_features.txt")
easy_file = os.path.join(VALID_PATH, "mtmc_easy_binary.txt")
hard_file = os.path.join(VALID_PATH, "mtmc_hard_binary.txt")
dataset = Dataset(tracklets_file, easy_file, hard_file, _type)

pbar = tqdm(total=len(dataset))

map_list = list()
with torch.no_grad():
    
    
    for data, target in dataset.prepare_data():
        count = 0.
        if data == None or target == None:
            dataset_len -= 1
            pbar.total -= 1
            pbar.refresh()
            continue

        data, target = data.to(device), target.to(device)
        preds = model(data)
        sort_preds = torch.argsort(preds, descending=True)
        target_list = target.cpu().numpy().tolist()
        # print(preds, target)
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

