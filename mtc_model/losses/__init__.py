import torch.nn.functional as F

from .triplet_loss import TripletLoss

def build_loss(device):
    def loss_func(f_prime, fij, target, P):
        n = P.size(0)
        triplet = TripletLoss(device)
        loss = F.cross_entropy(P.view(1, -1), target) + triplet(f_prime, fij, target)
        
        return loss

    return loss_func