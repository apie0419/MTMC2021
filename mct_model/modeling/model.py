import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
    nn.init.constant_(m.bias, 0.0)

class MCT(nn.Module):
    
    def __init__(self, dim, device):
        super(MCT, self).__init__()
        self.gkern_sig = 20.0
        self.lamb = 0.5
        self.device = device
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(dim, dim)
        
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.fc4 = nn.Linear(dim, dim)
        self.sim_fc = nn.Linear(dim, 1)
        
        self.fc1.apply(weights_init_kaiming)
        self.fc2.apply(weights_init_kaiming) 
        self.fc3.apply(weights_init_kaiming)
        self.fc4.apply(weights_init_kaiming)
        
        
    def attn(self, _input):
        # print (_input)
        x = self.fc1(_input)
        output = x @ _input.T
        # output = _input @ _input.T
        return output

    def projection_ratio(self, f):
        scores = self.attn(f)
        ind = np.diag_indices(scores.size()[0])
        fj_prime_mag = torch.norm(f, p=2, dim=1) ** 2
        S = (scores / fj_prime_mag).T
        S = S.view(self.num_tracklets, self.num_tracklets, 1)
        scores = F.softmax(scores, dim=0)
        return scores, S

    def similarity(self, f_prime, fij):  
        assert f_prime.size() == (self.num_tracklets, self.num_tracklets, self.feature_dim)
        assert fij.size() == (self.num_tracklets, self.num_tracklets, self.feature_dim)
        
        dist = torch.norm(torch.abs(fij - f_prime), p=2, dim=2) ** 2
        ind = np.diag_indices(dist.size()[0])
        dist[ind[0], ind[1]] = torch.zeros(dist.size()[0]).to(self.device)
        A = torch.exp(-0.5 * (dist / (self.gkern_sig ** 2)))
        return A

    def similarity_model(self, f_prime, fij):
        assert f_prime.size() == (self.num_tracklets, self.num_tracklets, self.feature_dim)
        assert fij.size() == (self.num_tracklets, self.num_tracklets, self.feature_dim)
        
        dist = torch.abs(fij - f_prime) ** 2
        A = torch.sigmoid(self.sim_fc(dist))
        A = A.view(A.size(0), A.size(1))
        return A

    def random_walk(self, A):
        p2g = A[0][1:]
        g2g = Variable(A[1:, 1:].clone(), requires_grad=False)

        g2g = g2g.view(g2g.size(0), g2g.size(0), 1)
        p2g = p2g.view(1, g2g.size(0), 1)
        one_diag = Variable(torch.eye(g2g.size(0)).to(self.device), requires_grad=False)
        inf_diag = torch.diag(torch.Tensor([-float('Inf')]).expand(g2g.size(0))).to(self.device) + g2g[:, :, 0].squeeze().data
        A = F.softmax(Variable(inf_diag), dim=1)
        A = (1 - self.lamb) * torch.inverse(one_diag - self.lamb * A)
        A = A.transpose(0, 1)
        p2g = torch.matmul(p2g.permute(2, 0, 1), A).permute(1, 2, 0).contiguous()
        p2g = p2g.flatten()
        return p2g.clamp(0, 1)

    def forward(self, f):
        """
        Return an affinity map, size(f[0], f[0])
        """
        self.num_tracklets, self.feature_dim = f.size()
        
        # f = F.relu(self.fc2(f))
        # if self.training:
        #     f = self.dropout(f)
        # f = F.relu(self.fc3(f))
        # if self.training:
        #     f = self.dropout(f)
        # f = F.relu(self.fc4(f))
        # if self.training:
        #     f = self.dropout(f)
        
        scores, S = self.projection_ratio(f)
        
        f = f.expand(self.num_tracklets, self.num_tracklets, self.feature_dim).permute(1, 0, 2)
        
        fij = f * S
        # fij = f.permute(1, 0, 2) ## 不做投影
        # A = self.similarity(f, fij)
        A = self.similarity_model(f, fij)
        P = self.random_walk(A)
        # P = (P - P.mean())
        # P = P * 100
        # P = torch.sigmoid(P)
        if self.training:
            return P, f[:, 0], fij
        else:
            return P


if __name__ == "__main__":
    num_tracklets = 4
    feature_dim = 2048
    tracklets = list()
    test = torch.rand((num_tracklets,))
    device = torch.device("cuda:0")
    for _ in range(num_tracklets):
        num_objects = random.randint(3, 10)
        tracklet = torch.rand((num_objects, feature_dim))
        mean = tracklet.mean(dim=0)
        std = tracklet.std(dim=0)
        tracklet_features = torch.cat((mean, std))
        tracklets.append(tracklet_features)
    
    tracklets = torch.stack(tracklets).to(device)
    # tracklets = torch.Tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [10, 10, 10]]).float().to(device)
    # model = MCT(3, device). to(device)
    model = MCT(feature_dim * 2, device).to(device)
    model.eval()
    output = model(tracklets)
    # print (output)