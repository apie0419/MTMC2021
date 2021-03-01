import torch.nn as nn
import torch
import numpy as np

class TripletLoss(object):
    
    def __init__(self, device, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.device = device
        self.triplet_margin_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def __call__(self, f_prime, fij, target):
        n, feat_size = f_prime.size()

        query_same_feats = fij[:, 0][1:] # 投影到 query 上的 feature
        query_anchor = f_prime[0].expand(n - 2, feat_size)
        query_positve = query_same_feats[target[0]].expand(n - 2, feat_size)
        query_negetive = torch.cat((query_same_feats[:target[0], :], query_same_feats[target[0] + 1:, :]))
        
        gallery_same_feats = fij[:, target[0], :].view(n, feat_size) # 投影到 target gallery 上的 feature
        gallery_same_feats = torch.cat((gallery_same_feats[:target[0] + 1], gallery_same_feats[target[0] + 2:]))
        gallery_anchor = f_prime[target[0] + 1].expand(n - 2, feat_size)
        gallery_positive = gallery_same_feats[0].expand(n - 2, feat_size)
        gallery_negetive = gallery_same_feats[1:]
        
        anchor = torch.cat((query_anchor, gallery_anchor))
        positive = torch.cat((query_positve, gallery_positive))
        negetive = torch.cat((query_negetive, gallery_negetive))

        return self.triplet_margin_loss(anchor, positive, negetive)

if __name__ == "__main__":
    import random

    device = torch.device("cuda:4")
    num_objects = 4
    feat_dim = 2048
    f_prime = torch.rand((num_objects, feat_dim)).to(device)
    fij = torch.rand((num_objects, num_objects, feat_dim)).to(device)
    targets = torch.tensor([random.randint(0, num_objects-2)]).long().to(device)
    criterion = TripletLoss(device)
    loss = criterion(f_prime, fij, targets)
    print (loss)