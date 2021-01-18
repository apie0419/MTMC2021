import os, torch

from tqdm       import tqdm
from utils      import init_path
from utils.data import Dataset
from modeling   import build_model
from config      import cfg
from losses     import build_loss

init_path()
model = build_model(cfg)
tracklets_file = os.path.join(cfg.PATH.INPUT_PATH, "gt_features.txt")
loss = build_loss()
dataset = Dataset(tracklets_file)
for data, label in tqdm(dataset.prepare_data(), total=len(dataset), desc="Epoch"):
    pass
