import torch, sys

sys.path.append("..")

from sct_model.tracklets.fushion_models.tracklet_connectivity import TrackletConnectivity

def build_model(cfg):
    model = TrackletConnectivity(cfg)
    weight_path = cfg.SCT.WEIGHTS
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model