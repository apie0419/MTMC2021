from .model import MCT

def build_model(cfg):
    num_E = cfg.MCT.E_LAYERS
    dim = cfg.MCT.FEATURE_DIM * 2
    return MCT(dim, num_E)
