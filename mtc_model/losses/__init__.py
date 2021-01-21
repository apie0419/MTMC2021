import torch.nn.functional as F

from .triplet_loss import TripletLoss

def build_loss():
    def loss_func(f_prime, fij, target, P):
        n = P.size(0)
        triplet = TripletLoss()
        return F.cross_entropy(P.view(1, -1), target) + triplet(f_prime, fij, target)[0]

    return loss_func