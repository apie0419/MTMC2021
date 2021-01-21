import torch.nn as nn
import torch
import numpy as np

class TripletLoss(nn.Module):
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def euclidean_dist(self, x, y):
        """
        Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
        Returns:
        dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(beta=1, alpha=-2, mat1=x, mat2=y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist


    # def forward(self, f_prime, fij, targets):
    def forward(self, f_prime, fij, target):
        n, feat_size = f_prime.size()
        # Compute pairwise distance, replace by the official when merged
        query_same_feats = fij[0]
        dist = torch.zeros(n-1)
        for i in range(1, n):
            dist[i-1] = self.euclidean_dist(query_same_feats[i].view(1, feat_size), f_prime[i].view(1, feat_size))
        
        dist_ap, dist_an = [], []

        dist_ap.append(torch.tensor(dist[target[0]]).unsqueeze(0))
        dist = torch.cat((dist[:target[0]], dist[target[0] + 1:]))
        dist_an.append(torch.tensor(dist.min()).unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        q2g_loss = self.ranking_loss(dist_an, dist_ap, y)
        
        fij = fij[1:]
        gallery_same_feats = fij[target[0], :, :].view(n, feat_size)

        ap = self.euclidean_dist(gallery_same_feats[0].view(1, feat_size), f_prime[0].view(1, feat_size))
        dist_ap, dist_an = [], []

        dist = torch.zeros(n-2)
        count = 0
        for i in range(1, n):
            if target[0] == i-1:
                dist[count] = self.euclidean_dist(gallery_same_feats[i].view(1, feat_size), f_prime[i].view(1, feat_size))
                count += 1

        dist_ap.append(ap.unsqueeze(0))
        dist_an.append(torch.tensor(dist.min()).unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
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
    criterion = TripletLoss()
    loss = criterion(f_prime, fij, targets)
    print (loss)