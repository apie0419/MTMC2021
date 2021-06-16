import torch.nn.functional as F
import torch.nn as nn
import torch
from .triplet_loss import TripletLoss

def build_loss(device):
    def loss_func(f_prime, fij, target, P, cams, cam_target):
        # triplet = TripletLoss(device)
        bce = nn.BCELoss()
        ce = nn.CrossEntropyLoss()
        # triplet_loss = triplet(f_prime, fij, target)
        cam_loss = ce(cams, cam_target)
        bce_loss = bce(P, target.float())

        return bce_loss, cam_loss

    return loss_func
