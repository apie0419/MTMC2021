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
        target_list = target.cpu().numpy().tolist()
        anchor, positive, negetive = list(), list(), list()

        query_anchor = f_prime[0]
        query_same_feats = fij[:, 0][1:] # 投影到 query 上的 feature
        
        query_positive = list()
        for t in target_list:
            query_positive.append(query_same_feats[t])

        query_negetive = [query_same_feats[:target_list[0], :]]
        for i in range(1, len(target_list)):
            query_negetive.append(query_same_feats[target_list[i-1]+1:target_list[i], :])
        query_negetive.append(query_same_feats[target_list[-1]+1:, :])
        query_positive = torch.cat(query_positive).view(-1, feat_size)
        query_negetive = torch.cat(query_negetive).view(-1, feat_size)
        nd = torch.norm(torch.abs(query_negetive - query_anchor), p=2, dim=1)
        pd = torch.norm(torch.abs(query_positive - query_anchor), p=2, dim=1)
        pos = query_positive[pd.argmax()]
        neg = query_negetive[nd.argmin()]
        anchor.append(query_anchor)
        positive.append(pos)
        negetive.append(neg)

        for t in target_list:
            gallery_anchor = f_prime[t+1]
            gallery_same_feats = fij[:, t+1, :].view(n, feat_size) # 投影到 target gallery 上的 feature
            gallery_positive = [gallery_same_feats[0]]
            gallery_same_feats = gallery_same_feats[1:]
            for tt in target_list:
                if tt != t:
                    gallery_positive.append(gallery_same_feats[tt])
            
            gallery_negetive = [gallery_same_feats[:target_list[0], :]]
            for i in range(1, len(target_list)):
                gallery_negetive.append(gallery_same_feats[target_list[i-1]+1:target_list[i], :])
            gallery_negetive.append(gallery_same_feats[target_list[-1]+1:, :])
            gallery_positive = torch.cat(gallery_positive).view(-1, feat_size)
            gallery_negetive = torch.cat(gallery_negetive).view(-1, feat_size)
            nd = torch.norm(torch.abs(gallery_negetive - gallery_anchor), p=2, dim=1)
            pd = torch.norm(torch.abs(gallery_positive - gallery_anchor), p=2, dim=1)
            pos = gallery_positive[pd.argmax()]
            neg = gallery_negetive[nd.argmin()]
            anchor.append(gallery_anchor)
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