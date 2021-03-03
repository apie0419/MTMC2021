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

device = torch.device(DEVICE + ':' + str(GPU))
tracklets_file = os.path.join(VALID_PATH, "gt_features.txt")
easy_file = os.path.join(VALID_PATH, "mtmc_easy.txt")
hard_file = os.path.join(VALID_PATH, "mtmc_hard.txt")
dataset = Dataset(tracklets_file, easy_file, hard_file)
checkpoint = torch.load(WEIGHT, map_location=device)
model = build_model(cfg, device)
model.load_state_dict(checkpoint)
model.eval()

dataset_len = dataset.hard_len() + dataset.easy_len()
easy = 0.
hard = 0.
cos_easy = 0.
cos_hard = 0.
eu_easy = 0.
eu_hard = 0.

pbar = tqdm(total=dataset_len)

with torch.no_grad():

    for _type in ["easy", "hard"]:
        for data, target in dataset.prepare_data(_type):
            if data == None or target == None:
                dataset_len -= 1
                pbar.total -= 1
                pbar.refresh()
                continue

            gallery = data[1:]
            query = data[0].view(1, -1)
            cos = (query @ gallery.T) / (torch.norm(query, p=2, dim=1) * torch.norm(gallery, p=2, dim=1))
            
            if cos.argmax().item() == target[0].item():
                if _type == "easy":
                    cos_easy += 1
                else:
                    cos_hard += 1

            eu_dist = torch.norm(torch.abs(query - gallery), p=2, dim=1)
            if eu_dist.argmin().item() == target[0].item():
                if _type == "easy":
                    eu_easy += 1
                else:
                    eu_hard += 1

            data, target = data.to(device), target.to(device)
            preds = model(data)
            # print (preds, target[0].item())
            if preds.argmax().item() == target[0].item():
                if _type == "easy":
                    easy += 1
                else:
                    hard += 1
            pbar.update()

hard_acc = hard / dataset.hard_len() * 100.
easy_acc = easy / dataset.easy_len() * 100.
cos_hard_acc = cos_hard / dataset.hard_len() * 100.
cos_easy_acc = cos_easy / dataset.easy_len() * 100.
eu_hard_acc = eu_hard / dataset.hard_len() * 100.
eu_easy_acc = eu_easy / dataset.easy_len() * 100.
pbar.close()
print ("Easy Accuracy:{:.2f}%, Hard Accuracy:{:.2f}%".format(easy_acc, hard_acc))
print ("Cos Easy Accuracy:{:.2f}%, Cos Hard Accuracy:{:.2f}%".format(cos_easy_acc, cos_hard_acc))
print ("Eu Easy Accuracy:{:.2f}%, Eu Hard Accuracy:{:.2f}%".format(eu_easy_acc, eu_hard_acc))

