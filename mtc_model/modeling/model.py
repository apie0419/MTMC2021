import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
    nn.init.constant_(m.bias, 0.0)

class MCT(nn.Module):
    
    def __init__(self, dim, device, num_E=3):
        super(MCT, self).__init__()
        self.gkern_sig = 10
        self.random_walk_iter = 30
        self.device = device

        self.fc1 = nn.Linear(dim, dim)
        self.fc1.apply(weights_init_kaiming)
        self.E = list()
        for _ in range(num_E):
            layer = nn.Linear(dim, dim)
            layer.apply(weights_init_kaiming)
            layer = layer.to(self.device)
            self.E.append(layer)

    def attn(self, _input):
        x = self.fc1(_input)
        output = x @ x.T

        return x, output

    def projection_ratio(self, f):

        f_prime, scores = self.attn(f)
        fj_prime_mag = torch.norm(f_prime, p=2, dim=1) ** 2
        S = scores / fj_prime_mag
        S = S.view(self.num_tracklets, self.num_tracklets, 1)

        return f_prime, S

    def similarity(self, f_prime, fij):  
        assert f_prime.size() == (self.num_tracklets, self.num_tracklets, self.feature_dim)
        assert fij.size() == (self.num_tracklets, self.num_tracklets, self.feature_dim)
        A = torch.exp(-0.5 * (torch.norm(fij - f_prime, p=2, dim=2) ** 2) / (self.gkern_sig ** 2))

        return A

    def random_walk(self, A):
        ind = np.diag_indices(A.size()[0])
        A[ind[0], ind[1]] = torch.zeros(A.size()[0]).to(self.device)
        
        D = torch.diag(torch.sum(A, axis=0))
        T = torch.inverse(D) @ A
        P = T[0]

        for _ in range(self.random_walk_iter):
            P = P @ T
        
        return P

    def forward(self, f):
        """
        Return an affinity map, size(f[0], f[0])
        """
        self.num_tracklets, self.feature_dim = f.size()
        f_prime, S = self.projection_ratio(f)
        f_prime = f_prime.expand(self.num_tracklets, self.num_tracklets, self.feature_dim)
        fij = f_prime * S
        for layer in self.E:
            fij = layer(fij)
        A = self.similarity(f_prime, fij)
        A = A.to(self.device)
        P = F.softmax(self.random_walk(A)[1:], dim=0)
        if self.training:
            return P, f_prime[0], fij
        else:
            return P


if __name__ == "__main__":
    num_tracklets = 3
    feature_dim = 2048
    tracklets = list()

    for _ in range(num_tracklets):
        num_objects = random.randint(3, 10)
        tracklet = torch.rand((num_objects, feature_dim))
        mean = tracklet.mean(dim=0)
        std = tracklet.std(dim=0)
        tracklet_features = torch.cat((mean, std))
        tracklets.append(tracklet_features)
    
    tracklets = torch.stack(tracklets)
        
    model = MCT(feature_dim * 2)
    output = model(tracklets)
