import torch.nn as nn
import torch
import numpy as np

class TripletLoss(object):
    
    def __init__(self, device, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.device = device
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.triplet_margin_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def euclidean_dist(self, x, y):
        dist = torch.norm(torch.abs(x - y), p=2, dim=1)
        return dist

    def __call__(self, f_prime, fij, target):
        n, feat_size = f_prime.size()

        query_same_feats = fij[:, 0] # 投影到 query 上的 feature
        query_f_prime = f_prime[0].expand(n - 1, feat_size)
        dist = self.euclidean_dist(query_same_feats[1:], query_f_prime)
        dist_ap, dist_an = [], []

        dist_ap.append(dist[target[0]])
        dist = torch.cat((dist[:target[0]], dist[target[0] + 1:]))
        dist_an.append(dist.min())
        dist_ap = torch.tensor(dist_ap, requires_grad=True).to(self.device)
        dist_an = torch.tensor(dist_an, requires_grad=True).to(self.device)
        y = torch.tensor([-1]).to(self.device)
        q2g_loss = self.ranking_loss(dist_ap, dist_an, y)

        gallery_same_feats = fij[:, target[0], :].view(n, feat_size) # 投影到 target gallery 上的 feature
        gallery_same_feats = torch.cat((gallery_same_feats[:target[0] + 1], gallery_same_feats[target[0] + 2:]))
        gallery_f_prime = f_prime[target[0] + 1].expand(n-1, feat_size)
        dist = self.euclidean_dist(gallery_same_feats, gallery_f_prime)
        dist_ap, dist_an = [], []

        dist_ap.append(dist[0])
        dist_an.append(dist[1:].min())

        dist_ap = torch.tensor(dist_ap, requires_grad=True).to(self.device)
        dist_an = torch.tensor(dist_an, requires_grad=True).to(self.device)
        y = torch.tensor([-1]).to(self.device)
        g2q_loss = self.ranking_loss(dist_ap, dist_an, y)
        return q2g_loss + g2q_loss

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