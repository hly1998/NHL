# 2024.2.22正则化方案
import torch

class SpaceAlignment(torch.nn.Module):
    # 蒸馏损失函数
    def __init__(self, bit_list=[16,32,64]):
        super(SpaceAlignment, self).__init__()
        self.bit_list = bit_list

    def forward(self, X_list):
        losses = []
        for idx in range(len(X_list)-1):
            align_loss = self.mse_loss(X_list[idx], X_list[idx+1][:,:self.bit_list[idx]])
            losses.append(align_loss)
        losses = torch.stack(losses)
        return losses
    
    def mse_loss(self, matrix1, matrix2):
        # 这里计算的是MSE距离 -> 考虑到有些模型输出的表征不是二值
        # 计算两个矩阵对应行之间的差异
        difference = matrix1 - matrix2
        # 计算差异的平方
        squared_difference = difference ** 2
        # 对每一行的平方差求和
        sum_squared_difference = torch.sum(squared_difference, dim=1)
        # 对每一行的和求平均值，得到MSE
        mse = sum_squared_difference / matrix1.size(1)
        return mse