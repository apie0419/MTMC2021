import torch.nn as nn
import torch
import numpy as np

class TripletLoss(object):
    
    def __init__(self, device, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.device = device
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(beta=1, alpha=-2, mat1=x, mat2=y.t())
        
        dist = dist.clamp(min=1e-8).sqrt()  # for numerical stability
        return dist

    def __call__(self, f_prime, fij, target):
        n, feat_size = f_prime.size()
        query_same_feats = fij[0]
        
        dist = self.euclidean_dist(query_same_feats[1:], f_prime[1:])

        dist_ap, dist_an = [], []

        dist_ap.append(dist[target[0]].unsqueeze(0))
        dist = torch.cat((dist[:target[0]], dist[target[0] + 1:]))
        dist_an.append(dist.min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap).to(self.device)
        dist_an = torch.cat(dist_an).to(self.device)
        y = torch.ones_like(dist_an).to(self.device)
        q2g_loss = self.ranking_loss(dist_an, dist_ap, y)
        
        fij = fij[1:]
        gallery_same_feats = fij[target[0], :, :].view(n, feat_size)
        
        gallery_same_feats = torch.cat((gallery_same_feats[:target[0] + 1], gallery_same_feats[target[0] + 2:]))
        gallery_f_prime = torch.cat((f_prime[:target[0] + 1], f_prime[target[0] + 2:]))
        dist = self.euclidean_dist(gallery_same_feats, gallery_f_prime)
        dist_ap, dist_an = [], []

        dist_ap.append(dist[0].unsqueeze(0))
        dist_an.append(dist[1:].min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap).to(self.device)
        dist_an = torch.cat(dist_an).to(self.device)
        y = torch.ones_like(dist_an).to(self.device)
        g2q_loss = self.ranking_loss(dist_an, dist_ap, y) 
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