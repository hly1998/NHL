from utils.tools import *
import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')

def lcdsh_config(config):
    config["lambda"] = 3
    return config

class LCDSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(LCDSHLoss, self).__init__()

    def forward(self, u, y, ind, config):
        s = 2 * (y @ y.t() > 0).float() - 1
        inner_product = u @ u.t() * 0.5
        inner_product = inner_product.clamp(min=-50, max=50)
        L1 = torch.log(1 + torch.exp(-s * inner_product)).mean()

        b = u.sign()
        inner_product_ = b @ b.t() * 0.5
        L2 = (inner_product.sigmoid() - inner_product_.sigmoid()).pow(2).mean()

        return L1 + config["lambda"] * L2
