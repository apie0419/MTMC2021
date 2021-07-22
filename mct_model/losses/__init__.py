import torch.nn.functional as F
import torch.nn as nn
import torch
from .ranked_list_loss import RankedLoss

def build_loss(device):
    def loss_func(f_prime, target, P, cams, cam_target, ranked_target):
        
        rll = RankedLoss()
        bce = nn.BCELoss()
        ce = nn.CrossEntropyLoss()
        ranked_loss = rll(f_prime, ranked_target)
        cam_loss = ce(cams, cam_target)
        bce_loss = bce(P, target.float())

        return bce_loss, cam_loss, ranked_loss

    return loss_func
