import torch

from . import init_path

init_path()

from mct_model.modeling.model import MCT

def build_model(cfg, device):
    dim = cfg.MCT.FEATURE_DIM * 2
    WEIGHT = cfg.MCT.WEIGHT
    model = MCT(dim, device).to(device)
    checkpoint = torch.load(WEIGHT, map_location=device)
    model.load_state_dict(checkpoint)
    return model