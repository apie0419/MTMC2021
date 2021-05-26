import torch.nn.functional as F
import torch.nn as nn
import torch
from .triplet_loss import TripletLoss

def build_loss(device):
    # def loss_func(f_prime, fij, target, P):
    def loss_func(f_prime, fij, target, P, cams, cam_target):
        triplet = TripletLoss(device)
        bce = nn.BCELoss()
        ce = nn.CrossEntropyLoss()
        triplet_loss = triplet(f_prime, fij, target)
        # ce_loss = ce(P.view(-1, 1), target.long())
        bce_target = torch.zeros(P.size(0)).to(device)
        bce_target[target] = 1.
        # cam_loss = ce(cams, cam_target)

        bce_loss = bce(P, bce_target.float())

        return triplet_loss, bce_loss
        # return triplet_loss, bce_loss, cam_loss

    return loss_func
