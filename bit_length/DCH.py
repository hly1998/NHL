from utils.tools import *
# from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

# torch.multiprocessing.set_sharing_strategy('file_system')

def dch_config(config):
    config["gamma"] = 20.0
    config["lambda"] = 0.1
    return config

class DCHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DCHLoss, self).__init__()
        self.gamma = config["gamma"]
        self.lambda1 = config["lambda"]
        self.K = bit
        self.one = torch.ones((config["batch_size"], bit)).to(config["device"])

    def d(self, hi, hj):
        inner_product = hi @ hj.t()
        norm = hi.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ hj.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        cos = inner_product / norm.clamp(min=0.0001)
        # formula 6
        return (1 - cos.clamp(max=0.99)) * self.K / 2

    def forward(self, u, y, ind, config):
        s = (y @ y.t() > 0).float()

        if (1 - s).sum() != 0 and s.sum() != 0:
            # formula 2
            positive_w = s * s.numel() / s.sum()
            negative_w = (1 - s) * s.numel() / (1 - s).sum()
            w = positive_w + negative_w
        else:
            # maybe |S1|==0 or |S2|==0
            w = 1

        d_hi_hj = self.d(u, u)
        # formula 8
        cauchy_loss = w * (s * torch.log(d_hi_hj / self.gamma) + torch.log(1 + self.gamma / d_hi_hj))
        # formula 9
        quantization_loss = torch.log(1 + self.d(u.abs(), self.one) / self.gamma)
        # formula 7
        loss = cauchy_loss.mean() + self.lambda1 * quantization_loss.mean()

        return loss


# def dch_train_val(config, bit, net):
#     specific_config(config)
#     device = config["device"]
#     train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
#     config["num_train"] = num_train
#     net = config["net"](bit).to(device)

#     optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

#     criterion = DCHLoss(config, bit)

#     Best_mAP = 0

#     for epoch in range(config["epoch"]):

#         current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

#         print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
#             config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

#         net.train()

#         train_loss = 0
#         for image, label, ind in train_loader:
#             image = image.to(device)
#             label = label.to(device)

#             optimizer.zero_grad()
#             u = net(image)

#             loss = criterion(u, label.float(), ind, config)
#             train_loss += loss.item()

#             loss.backward()
#             optimizer.step()

#         train_loss = train_loss / len(train_loader)

#         print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

#         if (epoch + 1) % config["test_map"] == 0:
#             Best_mAP_before = Best_mAP
#             mAP, Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
#             logging.info(f"{net.__class__.__name__} epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
#             if mAP > Best_mAP_before:
#                 count = 0
#             else:
#                 if count == config['stop_iter']:
#                     break
#                 count += 1
