# 跑不了，不考虑
from utils.tools import *
import torch
import torch.optim as optim
import time

# torch.multiprocessing.set_sharing_strategy('file_system')

def dfh_config(config):
    config["m"] = 3
    config["mu"] = 0.1
    config["vul"] = 1
    config["nta"] = 1
    config["eta"] = 0.5
    return config

class DFHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DFHLoss, self).__init__()
        self.U = torch.zeros(bit, config["num_train"]).float().to(config["device"])
        self.Y = torch.zeros(config["n_class"], config["num_train"]).float().to(config["device"])

        # Relax_center
        self.V = torch.zeros(bit, config["n_class"]).to(config["device"])

        # Center
        self.C = self.V.sign().to(config["device"])

        T = 2 * torch.eye(self.Y.size(0)) - torch.ones(self.Y.size(0))
        TK = self.V.size(0) * T
        self.TK = torch.FloatTensor(torch.autograd.Variable(TK, requires_grad=False)).to(config["device"])


    def forward(self, u, y, ind, config):

        self.U[:, ind] = u.t().data
        self.Y[:, ind] = y.t()

        b = (config["mu"] * self.C @ y.t() + u.t()).sign()

        self.Center_gradient(torch.autograd.Variable(self.V, requires_grad=True),
                             torch.autograd.Variable(y, requires_grad=False),
                             torch.autograd.Variable(b, requires_grad=False), config)

        s = (y @ self.Y > 0).float()
        inner_product = u @ self.U * 0.5
        inner_product = inner_product.clamp(min=-100, max=50)
        metric_loss = ((1 - s) * torch.log(1 + torch.exp(config["m"] + inner_product))
                       + s * torch.log(1 + torch.exp(config["m"] - inner_product))).mean()
        # metric_loss = (torch.log(1 + torch.exp(theta)) - S * theta).mean()  # Without Margin
        quantization_loss = (b - u.t()).pow(2).mean()
        loss = metric_loss + config["eta"] * quantization_loss
        return loss

    def Center_gradient(self, V, batchy, batchb, config):
        alpha = 0.03
        for i in range(200):
            intra_loss = (V @ batchy.t() - batchb).pow(2).mean()
            inter_loss = (V.t() @ V - self.TK).pow(2).mean()
            quantization_loss = (V - V.sign()).pow(2).mean()

            loss = intra_loss + config["vul"] * inter_loss + config["nta"] * quantization_loss

            loss.backward()

            if i in (149, 179):
                alpha = alpha * 0.1

            V.data = V.data - alpha * V.grad.data

            V.grad.data.zero_()
        self.V = V
        self.C = self.V.sign()