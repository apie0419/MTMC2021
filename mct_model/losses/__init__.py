import torch.nn.functional as F

from .triplet_loss import TripletLoss

def build_loss(device):
    def loss_func(f_prime, fij, target, P):
        n = P.size(0)
        t = TripletLoss(device)
        triplet = t(f_prime, fij, target)
        cross = F.cross_entropy(P.view(1, -1), target)
        
        return triplet, cross

    return loss_func