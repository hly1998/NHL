from utils.tools import *
import torch
import time

def dhn_config(config):
    config["alpha"] = 0.1
    return config

class DHNLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DHNLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        s = (y @ self.Y.t() > 0).float()
        inner_product = u @ self.U.t() * 0.5

        likelihood_loss = (1 + (-inner_product.abs()).exp()).log() + inner_product.clamp(min=0) - s * inner_product

        likelihood_loss = likelihood_loss.mean()

        # quantization_loss = config["alpha"] * (u.abs() - 1).abs().mean()
        quantization_loss = config["alpha"] * (u.abs() - 1).cosh().log().mean()

        return likelihood_loss + quantization_loss

