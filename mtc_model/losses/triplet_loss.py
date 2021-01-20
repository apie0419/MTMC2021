import torch.nn as nn
import torch
import numpy as np

class TripletLoss(nn.Module):
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(beta=1, alpha=-2, mat1=inputs, mat2=inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        dist_ap, dist_an = [], []
        anchor_dist = dist[0].numpy()

        dist_ap.append(torch.tensor(anchor_dist[targets]).unsqueeze(0))
        anchor_dist = np.delete(anchor_dist, 0)
        dist_an.append(torch.tensor(anchor_dist.min()).unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        
        return self.ranking_loss(dist_an, dist_ap, y)

if __name__ == "__main__":
    import random
    batch_size = 64
    feat_dim = 2048
    num_classes = 10
    inputs = torch.rand((batch_size, feat_dim))
    targets = torch.zeros(num_classes, dtype=torch.long)
    targets[random.randint(0, num_classes-1)] = 1
    criterion = TripletLoss()
    loss = criterion(inputs, targets)
    print (loss)