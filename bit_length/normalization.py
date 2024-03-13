import torch
from torch import nn

def fowardbn(x, gam, beta, ):
    momentum = 0.1
    eps = 1e-05
    running_mean = 0
    running_var = 1
    running_mean = (1 - momentum) * running_mean + momentum * x.mean(dim=0)
    running_var = (1 - momentum) * running_var + momentum * x.var(dim=0)
    mean = x.mean(dim=0)
    var = x.var(dim=0,unbiased=False)
    # bnmiddle_buffer = (input - mean) / ((var + eps) ** 0.5).data
    x_hat = (x - mean) / torch.sqrt(var + eps)
    out = gam * x_hat + beta
    cache = (x, gam, beta, x_hat, mean, var, eps)
    return out, cache

model2 = nn.BatchNorm1d(5)
input1 = torch.randn(3, 5, requires_grad=True)
input2 = input1.clone().detach().requires_grad_()
x = model2(input1)
print(x)
out, cache = fowardbn(input2, model2.weight, model2.bias)
print(out)