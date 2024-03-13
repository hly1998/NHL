from utils.tools import *
import torch
import torch.optim as optim
import time

def dbdh_config(config):
    config["alpha"] = 0.1
    config["p"] = 2
    return config

class DBDHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DBDHLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind, config):
        u = u.clamp(min=-1, max=1)
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        s = (y @ self.Y.t() > 0).float()
        inner_product = u @ self.U.t() * 0.5

        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product

        likelihood_loss = likelihood_loss.mean()

        if config["p"] == 1:
            quantization_loss = config["alpha"] * u.mean(dim=1).abs().mean()
        else:
            quantization_loss = config["alpha"] * u.mean(dim=1).pow(2).mean()

        return likelihood_loss + quantization_loss

