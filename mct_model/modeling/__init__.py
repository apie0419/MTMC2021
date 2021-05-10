from .model import MCT

def build_model(cfg, device):
    dim = cfg.MCT.FEATURE_DIM * 2
    model = MCT(dim, device)
    model = model.to(device)
    return model
