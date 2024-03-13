from utils.tools import *
import torch
import logging
import kornia.augmentation as Kg
import torch.nn.functional as F

def dhd_config(config):
    config["lambda1"] = 0.0001
    config["transformation_scale"] = 0.5
    return config

class Augmentation(torch.nn.Module):
    def __init__(self, org_size, Aw=1.0):
        super(Augmentation, self).__init__()
        self.gk = int(org_size*0.1)
        if self.gk%2==0:
            self.gk += 1
        self.Aug = torch.nn.Sequential(
        Kg.RandomResizedCrop(size=(org_size, org_size), p=1.0*Aw),
        Kg.RandomHorizontalFlip(p=0.5*Aw),
        Kg.ColorJitter(brightness=0.4, contrast=0.8, saturation=0.8, hue=0.2, p=0.8*Aw),
        Kg.RandomGrayscale(p=0.2*Aw),
        Kg.RandomGaussianBlur((self.gk, self.gk), (0.1, 2.0), p=0.5*Aw))

    def forward(self, x):
        return self.Aug(x)

class DHDLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DHDLoss, self).__init__()
        self.temp = 0.2
        self.P = torch.nn.Parameter(torch.FloatTensor(config["n_class"], bit), requires_grad=True).to(config["device"])
        torch.nn.init.xavier_uniform_(self.P, gain=torch.nn.init.calculate_gain('tanh'))

    def forward(self, ST, L, ind, config):
        # (xS, xT) = ST
        xS = ST[0]
        xT = ST[1]
        xS = torch.tanh(xS)
        xT = torch.tanh(xT)
        X = F.normalize(xT, p = 2, dim = -1)
        P = F.normalize(self.P, p = 2, dim = -1)
        D = F.linear(X, P) / self.temp
        xent_loss = torch.mean(torch.sum(-L * F.log_softmax(D, -1), -1))
        # print(xS.shape, xT.shape)
        if len(xS.shape) == 1:
            xS = xS.unsqueeze(0)
            xT = xT.unsqueeze(0)
        HKDloss = (1 - F.cosine_similarity(xS, xT.detach())).mean()
        return xent_loss + 0.0001 * HKDloss