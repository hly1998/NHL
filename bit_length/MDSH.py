import torch
from utils.tools import *
import torch.nn.functional as F
import numpy as np
import torch.nn as NN
# from scipy.special import comb
def mdsh_config(config):
    config["m"] = 16
    config["alpha"] = 1
    config["beta"] = 1
    config["beta2"] = 2
    config["mome"] = 0.9
    config["epoch_change"] = 9
    config["sigma"] = 1.0
    config["gamma"] = 20.0
    config["lambda"] = 0.0001
    config["mu"] = 1
    config["nu"] = 1
    config["eta"] = 55
    config["dcc_iter"] = 10
    config["T"] = 1e-3
    return config

class MDSHLoss(NN.Module):
    def __init__(self, config, bit):
        """
        :param config: in paper, the hyper-parameter lambda is chose 0.0001
        :param bit:
        """
        device = config["device"]
        l = list(range(config['n_class']))
        super(MDSHLoss, self).__init__()
        self.config = config
        self.bit = bit
        # self.alpha_pos, self.alpha_neg, self.beta_neg, self.d_min, self.d_max = self.get_margin()
        self.hash_center = self.generate_center(bit, config['n_class'], l).to(device)
        # np.save(config['save_center'], self.hash_center.cpu().numpy())
        self.BCEloss = torch.nn.BCELoss().to(device)
        self.Y = torch.randn(self.config['num_train'], self.config['n_class']).float().to(device)
        self.U = torch.randn(config['num_train'], bit).to(device)
        self.label_center = torch.from_numpy(
            np.eye(config['n_class'], dtype=np.float32)[np.array([i for i in range(config['n_class'])])]).to(device)
        self.tanh = NN.Tanh().to(device)

    def forward(self, u1u2, y, ind, k=0):
        # (u1, u2) = u1u2
        u1 = u1u2[0]
        u2 = u1u2[1]
        k = 0
        self.U[ind, :] = u2.data
        self.Y[ind, :] = y.float()
        return self.cos_pair(u1, y, ind, k)

    
    def cos_pair(self, u, y, ind, k):
        if k < self.config['epoch_change']:
            pair_loss = 0
        else:
            last_u = self.U
            last_y = self.Y
            pair_loss = self.moco_pairloss(u, y, last_u, last_y, ind)
        cos_loss = self.cos_eps_loss(u, y, ind)
        Q_loss = (u.abs() - 1).pow(2).mean()
        
        loss = cos_loss + self.config['beta'] * pair_loss + self.config['lambda'] * Q_loss
        # return loss, cos_loss, pair_loss
        return loss

    def moco_pairloss(self, u, y, last_u, last_y, ind):
        u = F.normalize(u)
        last_u = F.normalize(last_u)
        # label_sim = ((y @ y.t()) > 0).float()
        # cos_sim = u @ u.t()
        last_sim = ((y @ last_y.t()) > 0).float()
        last_cos = u @ last_u.t()

        loss = torch.sum(last_sim * torch.log(1 + torch.exp(1/2 *(1 - last_cos))))/torch.sum(last_sim) # only the positive pair 
        return loss
    
    def cos_eps_loss(self, u, y, ind):
        K = self.bit
        # m = 0.0
        # l = 1 - 2 * self.d_max / K
        u_norm = F.normalize(u)
        centers_norm = F.normalize(self.hash_center)
        cos_sim = torch.matmul(u_norm, torch.transpose(centers_norm, 0, 1)) # batch x n_class
        s = (y @ self.label_center.t()).float() # batch x n_class
        cos_sim = K ** 0.5 * cos_sim
        p = torch.softmax(cos_sim, dim=1)
        loss = s * torch.log(p) + (1-s) * torch.log(1-p)
        loss = torch.mean(loss)
        return -loss

    def generate_center(self, bit, n_class, l):
        hash_centers = np.load(f'./tmp_file/MDSH/init_{n_class}_{bit}.npy')
        hash_centers = hash_centers[l]
        Z = torch.from_numpy(hash_centers).float()
        return Z
