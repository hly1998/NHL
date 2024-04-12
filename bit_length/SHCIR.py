import torch
from utils.tools import *
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as NN
import random

from tqdm import tqdm
from scipy.special import comb

def shcir_config(config):
    config["m"] = 16
    config["alpha"] = 1
    config["beta"] = 0.1
    config["beta2"] = 0.01
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

class SHCIRLoss(NN.Module):
    def __init__(self, config, bit):
        super(SHCIRLoss, self).__init__()
        l = config["n_class"]
        device = config["device"]
        self.config = config
        self.bit = bit
        # if 'zero' in config['remarks']: 
        # self.alpha_pos, self.alpha_neg, self.beta_neg, self.d_min, self.d_max = self.get_margin()
        self.hash_center = self.generate_center(bit, config, device)
        # np.save(config['save_center'], self.hash_center.cpu().numpy())
        self.label_center = torch.from_numpy(
            np.eye(config['n_class'], dtype=np.float32)[np.array([i for i in range(config['n_class'])])]).to(device)

    def forward(self, u1, y, ind, k=0):
        return self.cos_pair(u1, y, ind, k)[0]
    
    def cos_pair(self, u, y, ind, k):
        cos_loss = self.cos_eps_loss(u, y, ind)
        Q_loss = (u.abs() - 1).pow(2).mean()
        
        loss = cos_loss + self.config['lambda'] * Q_loss
        return loss, cos_loss
    
    def cos_eps_loss(self, u, y, ind):
        K = self.bit
        m = 0.0
        u_norm = F.normalize(u)
        centers_norm = F.normalize(self.hash_center)
        cos_sim = torch.matmul(u_norm, torch.transpose(centers_norm, 0, 1)) # batch x n_class
        s = (y.float() @ self.label_center.t()).float() # batch x n_class
        loss = 0
        for i in range(u_norm.shape[0]):
            pos_pair = cos_sim[i][s[i] == 1] # 1
            neg_pair = cos_sim[i][s[i] == 0] # n_class - 1
            pos_eps = torch.exp(K**0.5 * (pos_pair - m))
            neg_eps = torch.exp(K**0.5 * (neg_pair - m))
            batch_loss = torch.log(pos_eps / (pos_eps + torch.sum(neg_eps))) + torch.sum(torch.log(1 - neg_eps / (pos_eps + torch.sum(neg_eps)))) / (self.config['n_class']-1)
            loss += batch_loss
        loss /= u_norm.shape[0]
        return -loss
            
    def get_margin(self):
        L = self.bit
        n_class = self.config['n_class']
        right = (2 ** L) / n_class
        d_min = 0
        d_max = 0
        for j in range(2 * L + 4):
            dim = j
            sum_1 = 0
            sum_2 = 0
            for i in range((dim - 1) // 2 + 1):
                sum_1 += comb(L, i)
            for i in range((dim) // 2 + 1):
                sum_2 += comb(L, i)
            if sum_1 <= right and sum_2 > right:
                d_min = dim
        for i in range(2 * L + 4):
            dim = i
            sum_1 = 0
            sum_2 = 0
            for j in range(dim):
                sum_1 += comb(L, j)
            for j in range(dim - 1):
                sum_2 += comb(L, j)
            if sum_1 >= right and sum_2 < right:
                d_max = dim
        alpha_neg = L - 2 * d_max
        beta_neg = L - 2 * d_min
        alpha_pos = L
        return alpha_pos, alpha_neg, beta_neg, d_min, d_max

    def generate_center(self, bit, config, device):
        hash_centers = np.load(f"./tmp_file/SHCIR/SHCIR_{config['dataset']}_{bit}.npy")
        # self.evaluate_centers(hash_centers)
        Z = torch.from_numpy(hash_centers).float().to(device)
        return Z
    
    def evaluate_centers(self, H):
        dist = []
        for i in range(H.shape[0]):
            for j in range(i+1, H.shape[0]):
                    TF = np.sum(H[i] != H[j])
                    dist.append(TF)
        dist = np.array(dist)
        st = dist.mean() - dist.var() + dist.min()
        print(f"mean is {dist.mean()}; min is {dist.min()}; var is {dist.var()}; max is {dist.max()}")
