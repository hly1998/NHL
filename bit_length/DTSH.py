from utils.tools import *
# from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

def dtsh_config(config):
    config["alpha"] = 5
    config["lambda"] = 1
    return config

class DTSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DTSHLoss, self).__init__()

    def forward(self, u, y, ind, config):

        inner_product = u @ u.t()
        s = y @ y.t() > 0
        count = 0

        loss1 = 0
        for row in range(s.shape[0]):
            # if has positive pairs and negative pairs
            if s[row].sum() != 0 and (~s[row]).sum() != 0:
                count += 1
                theta_positive = inner_product[row][s[row] == 1]
                theta_negative = inner_product[row][s[row] == 0]
                triple = (theta_positive.unsqueeze(1) - theta_negative.unsqueeze(0) - config["alpha"]).clamp(min=-100,
                                                                                                             max=50)
                loss1 += -(triple - torch.log(1 + torch.exp(triple))).mean()

        if count != 0:
            loss1 = loss1 / count
        else:
            loss1 = 0

        loss2 = config["lambda"] * (u - u.sign()).pow(2).mean()

        return loss1 + loss2
