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
        target = target.cpu().numpy()
        anchor, positive, negetive = list(), list(), list()

        query_anchor = f_prime[0]
        query_same_feats = fij[:, 0][1:] # 投影到 query 上的 feature
        
        query_positive = list()
        query_positive.append(query_same_feats[target])

        query_negetive = [query_same_feats[:target, :]]
        query_negetive.append(query_same_feats[target+1:, :])
        query_positive = torch.cat(query_positive).view(-1, feat_size)
        query_negetive = torch.cat(query_negetive).view(-1, feat_size)
        nd = torch.norm(torch.abs(query_negetive - query_anchor), p=2, dim=1)
        pd = torch.norm(torch.abs(query_positive - query_anchor), p=2, dim=1)
        pos = query_positive[pd.argmax()]
        neg = query_negetive[nd.argmin()]
        anchor.append(query_anchor)
        positive.append(pos)
        negetive.append(neg)
        
        anchor = torch.stack(anchor)
        positive = torch.stack(positive)
        negetive = torch.stack(negetive)
        return self.triplet_margin_loss(anchor, positive, negetive)

if __name__ == "__main__":
    import random

    device = torch.device("cuda:0")
    num_objects = 6
    feat_dim = 2048
    f_prime = torch.rand((num_objects, feat_dim)).to(device)
    fij = torch.rand((num_objects, num_objects, feat_dim)).to(device)
    targets = torch.tensor([1, 3]).long().to(device)
    criterion = TripletLoss(device)
    loss = criterion(f_prime, fij, targets)
    print (loss)