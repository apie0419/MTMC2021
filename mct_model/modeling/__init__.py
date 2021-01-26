from .model import MCT

def build_model(cfg, device):
    dim = cfg.MCT.FEATURE_DIM * 2
    return MCT(dim, device).to(device)
