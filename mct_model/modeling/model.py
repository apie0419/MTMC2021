import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0.0)

class MCT(nn.Module):
    
    def __init__(self, dim, device):
        super(MCT, self).__init__()
        self.lamb = 0.5
        self.device = device
        self.trans_fc1 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(dim, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)

        self.att_fc = nn.Linear(dim, dim)
        self.sim_fc = nn.Linear(512, 1)
        self.cam_fc1 = nn.Linear(dim, dim)
        self.cam_fc2 = nn.Linear(dim, 512)
        self.cam_fc3 = nn.Linear(512, 36)
        
        self.fc1.apply(weights_init_kaiming)
        self.fc2.apply(weights_init_kaiming)
        self.fc3.apply(weights_init_kaiming)
        self.trans_fc1.apply(weights_init_kaiming)
        self.att_fc.apply(weights_init_kaiming)
        self.sim_fc.apply(weights_init_kaiming)
        self.cam_fc1.apply(weights_init_kaiming)
        self.cam_fc2.apply(weights_init_kaiming)
        self.cam_fc3.apply(weights_init_classifier)

    def attn(self, _input):
        x = self.att_fc(_input)
        
        output = _input @ _input.T
        return output

    def projection_ratio(self, f):
        scores = self.attn(f)
        ind = np.diag_indices(scores.size()[0])
        fj_prime_mag = torch.norm(f, p=2, dim=1) ** 2
        S = (scores / fj_prime_mag).T
        S = S.view(self.num_tracklets, self.num_tracklets, 1)
        return S

    def random_walk(self, A):
        p2g = A[0][1:]
        g2g = A[1:, 1:]

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
        self.num_tracklets, _ = f.size()
        
        copy_f = Variable(f.clone(), requires_grad=True)
        cam_f = F.relu(self.cam_fc1(copy_f))
        # f -= cam_f
        cam_f = F.relu(self.cam_fc2(cam_f))
        cams = self.cam_fc3(cam_f)
        f = F.relu(self.trans_fc1(f))
        # S = self.projection_ratio(f)
        f = f.expand(self.num_tracklets, self.num_tracklets, 4096).permute(1, 0, 2)
        # fij = f * S
        fij = f.permute(1, 0, 2) ## 不做投影
        dist = abs(fij - f)
        dist = F.relu(self.fc1(dist))
        dist = F.relu(self.fc2(dist))
        dist = F.relu(self.fc3(dist))
        A = torch.sigmoid(self.sim_fc(dist))
        A = A.view(A.size(0), A.size(1))
    
        if self.training:
            return A, fij[0], fij, cams
        else:
            return A


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