# 2024.3.8 结果非常差，不考虑
from utils.tools import *

import os
import torch
from itertools import product
from random import shuffle
from tqdm import tqdm


# torch.multiprocessing.set_sharing_strategy('file_system')


# CNNH(AAAI2014)
# paper [Supervised Hashing for Image Retrieval via Image Representation Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/download/8137/8861)
# code[CNNH-pytorch](https://github.com/heheqianqian/CNNH)
# [CNNH] epoch:20, bit:48, dataset:cifar10-1, MAP:0.134, Best MAP: 0.134
# [CNNH] epoch:80, bit:48, dataset:nuswide_21, MAP:0.386, Best MAP: 0.386

def cnnh_config(config):
    config["T"] = 10
    config["H_save_path"] = "save/CNNH/",
    return config


class CNNHLoss(torch.nn.Module):
    def __init__(self, config, train_labels, bit):

        super(CNNHLoss, self).__init__()
        S = (train_labels @ train_labels.t() > 0).float() * 2 - 1
        # load H if exists
        save_full_path = "%sH_T(%d)_bit(%d)_dataset(%s).pt" % (
            config["H_save_path"], config["T"], bit, config["dataset"])
        if os.path.exists(save_full_path):
            print("loading ", save_full_path)
            self.H = torch.load(save_full_path).to(config["device"])
        else:
            self.H = self.stage_one(config["num_train"], bit, config["T"], S, config["H_save_path"], config["dataset"],
                                    config["device"])

    def stage_one(self, n, q, T, S, H_save_path, dataset, device):

        # if not os.path.exists(H_save_path):
        #     os.makedirs(H_save_path)

        H = 2 * torch.rand((n, q)).to(device) - 1
        L = H @ H.t() - q * S
        permutation = list(product(range(n), range(q)))
        for t in range(T):
            H_temp = H.clone()
            L_temp = L.clone()
            shuffle(permutation)
            for i, j in tqdm(permutation):
                # formula 7
                g_prime_Hij = 4 * L[i, :] @ H[:, j]
                g_prime_prime_Hij = 4 * (H[:, j].t() @ H[:, j] + H[i, j].pow(2) + L[i, i])
                # formula 6
                d = (-g_prime_Hij / g_prime_prime_Hij).clamp(min=-1 - H[i, j], max=1 - H[i, j])
                # formula 8
                L[i, :] = L[i, :] + d * H[:, j].t()
                L[:, i] = L[:, i] + d * H[:, j]
                L[i, i] = L[i, i] + d * d

                H[i, j] = H[i, j] + d

            if L.pow(2).mean() >= L_temp.pow(2).mean():
                H = H_temp
                L = L_temp
            # save_full_path = "%sH_T(%d)_bit(%d)_dataset(%s).pt" % (H_save_path, t + 1, bit, dataset)
            # torch.save(H.sign().cpu(), save_full_path)
            # print("[CNNH stage 1][%d/%d] reconstruction loss:%.7f ,H save in %s" % (
            #     t + 1, T, L.pow(2).mean().item(), save_full_path))
        return H.sign()

    def forward(self, u, y, ind, config):
        loss = (u - self.H[ind]).pow(2).mean()
        return loss
