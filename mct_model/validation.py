import os, torch

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

device = torch.device(DEVICE + ':' + str(5))
tracklets_file = os.path.join(VALID_PATH, "gt_features.txt")
dataset = Dataset(tracklets_file, 20, 40)
checkpoint = torch.load(WEIGHT, map_location=device)
model = build_model(cfg, device)
model.load_state_dict(checkpoint)
model.eval()

dataset_len = len(dataset)
count = 0.

pbar = tqdm(total=dataset_len)

with torch.no_grad():
    for data, target in dataset.prepare_data():
        if data == None or target == None:
            dataset_len -= 1
            pbar.total -= 1
            pbar.refresh()
            continue

        data, target = data.to(device), target.to(device)
        preds = model(data)
        # print (preds, target[0].item())
        if preds.argmax().item() == target[0].item():
            count += 1
        
        pbar.update()

val_acc = count / dataset_len * 100.
pbar.close()
print ("Validation Accuracy:{:.2f}%".format(val_acc))