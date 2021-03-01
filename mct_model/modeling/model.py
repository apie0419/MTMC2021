import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
    nn.init.constant_(m.bias, 0.0)

class MCT(nn.Module):
    
    def __init__(self, dim, device):
        super(MCT, self).__init__()
        self.gkern_sig = 15.0
        self.lamb = 0.9
        self.device = device

        self.fc1 = nn.Linear(dim, dim)
        
        ## E
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.fc4 = nn.Linear(dim, dim)

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
        
        dist = torch.norm(torch.abs(fij - f_prime), p=2, dim=2) ** 2
        ind = np.diag_indices(dist.size()[0])
        dist[ind[0], ind[1]] = torch.zeros(dist.size()[0]).to(self.device)
        A = torch.exp(-0.5 * (dist / (self.gkern_sig ** 2)))
        
        return A

    def random_walk(self, A):
        total = torch.sum(A, axis=1) - torch.diagonal(A)
        total = total.clamp(min=1e-10)
        D = torch.diag(total)
        T = torch.inverse(D) @ A
        ind = np.diag_indices(T.size()[0])
        T[ind[0], ind[1]] = torch.zeros(T.size()[0]).to(self.device)
        P0 = T[0][1:]
        T = T[1:, 1:]
        I = torch.eye(T.size()[0]).to(self.device)
        P = (1 - self.lamb) * torch.inverse(I - self.lamb * T) @ P0
        
        return P

    def forward(self, f):
        """
        Return an affinity map, size(f[0], f[0])
        """
        self.num_tracklets, self.feature_dim = f.size()
        f = self.fc2(f)
        f = self.fc3(f)
        f = self.fc4(f)
        f_prime, S = self.projection_ratio(f)
        f = f.expand(self.num_tracklets, self.num_tracklets, self.feature_dim)
        fij = f * S
        
        A = self.similarity(f, fij)
        P = self.random_walk(A)
        # 
        # print (A[0][1:])
        # P = A[0][1:]
        # P = P / P.sum()
        # print (P)
        P = (P - P.mean())
        P[P < 0] = 0
        P = P * 100
        P = F.softmax(P, dim=0)
        if self.training:
            return P, f[0], fij
        else:
            return P


if __name__ == "__main__":
    num_tracklets = 3
    feature_dim = 2048
    tracklets = list()
    device = torch.device("cuda:5")
    for _ in range(num_tracklets):
        num_objects = random.randint(3, 10)
        tracklet = torch.rand((num_objects, feature_dim))
        mean = tracklet.mean(dim=0)
        std = tracklet.std(dim=0)
        tracklet_features = torch.cat((mean, std))
        tracklets.append(tracklet_features)
    
    tracklets = torch.stack(tracklets).to(device)
        
    model = MCT(feature_dim * 2, device).to(device)
    model.eval()
    output = model(tracklets)
    print (output)