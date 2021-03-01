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
easy_file = os.path.join(VALID_PATH, "mtmc_easy.txt")
hard_file = os.path.join(VALID_PATH, "mtmc_hard.txt")
dataset = Dataset(tracklets_file, easy_file, hard_file)
checkpoint = torch.load(WEIGHT, map_location=device)
model = build_model(cfg, device)
model.load_state_dict(checkpoint)
model.eval()

dataset_len = dataset.hard_len() + dataset.easy_len()
count = 0.
easy = 0.
hard = 0.

pbar = tqdm(total=dataset_len)

with torch.no_grad():

    for _type in ["easy", "hard"]:
        for data, target in dataset.prepare_data(_type):
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
                if _type == "easy":
                    easy += 1
                else:
                    hard += 1
            pbar.update()

val_acc = count / dataset_len * 100.
hard_acc = hard / len(dataset.hard_data_list) * 100.
easy_acc = easy / len(dataset.easy_data_list) * 100.
pbar.close()
print ("Easy Accuracy:{:.2f}%, Hard Accuracy:{:.2f}%".format(easy_acc, hard_acc))
# print ("Validation Accuracy:{:.2f}%".format(val_acc))