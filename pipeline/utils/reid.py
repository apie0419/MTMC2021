import torch

from . import init_path

init_path()

from reid_model.modeling import make_model, build_transform

def build_model(cfg):
    WEIGHT=cfg.REID.WEIGHTS
    NUM_CLASSES=cfg.REID.NUM_CLASSES
    model = make_model(NUM_CLASSES)
    checkpoint = torch.load(WEIGHT)
    model.load_state_dict(checkpoint['state_dict'])
    return model