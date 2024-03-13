import torch.nn as nn
from torchvision import models
# import torch.utils.model_zoo as model_zoo
import torch
# import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, pretrained=True):
        super(AlexNet, self).__init__()

        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            # nn.ReLU(inplace=True),
            # nn.Linear(4096, hash_bit),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6) 
        x = self.hash_layer(x) # output_size:4096
        return x

class VGG16(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16, self).__init__()

        self.vgg = models.vgg16(pretrained=pretrained)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])
        self.hash_layer = nn.Sequential(nn.Linear(4096, 1024),
                                        nn.ReLU(),
                                        )
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.vgg.features(x)
        x = x.view(x.size(0), -1)
        x = self.vgg.classifier(x)
        x = self.hash_layer(x) # output_size:1024
        return x

class ResNet(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(ResNet, self).__init__()
        model_resnet = models.resnet50(pretrained=pretrained)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)
        
        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x, is_feat=False):
        x = self.feature_layers(x)
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        x = self.avgpool(feat4)
        x = x.view(x.size(0), -1)
        x = self.hash_layer(x)
        if is_feat:
            return feat1, feat2, feat3, feat4, x
        else:
            return x

class MoCo(nn.Module):
    # 用于MDSH或SHCIR模型
    def __init__(self, config, hash_bit):
        super(MoCo, self).__init__()
        self.m = config['mome']
        self.encoder_q = ResNet(hash_bit)
        self.encoder_k = ResNet(hash_bit)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x):
        encode_x = self.encoder_q(x)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            encode_x2 = self.encoder_k(x)
        return encode_x, encode_x2


class ResNet_f(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet_f, self).__init__()
        model_resnet = models.resnet50(pretrained=pretrained)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)

    def forward(self, x):
        x = self.feature_layers(x)
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        x = self.avgpool(feat4)
        x = x.view(x.size(0), -1)
        return x # output_size: 2048

class MoCo_RML(nn.Module):
    # 用于MDSH或SHCIR模型
    def __init__(self, config):
        super(MoCo_RML, self).__init__()
        self.m = config['mome']
        self.encoder_q = ResNet_f()
        self.encoder_k = ResNet_f()
        self.bit_list = config["bit_list"]
        self.hash_layer_q = nn.Linear(2048, self.bit_list[-1])
        self.hash_layer_k = nn.Linear(2048, self.bit_list[-1])
        self.hash_layer_q.weight.data.normal_(0, 0.01)
        self.hash_layer_k.bias.data.fill_(0.0)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.hash_layer_q.parameters(), self.hash_layer_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.hash_layer_q.parameters(), self.hash_layer_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x):
        encode_x = self.encoder_q(x)
        outputs_x = [torch.matmul(encode_x, self.hash_layer_q.weight[:bit, :].t()) + self.hash_layer_q.bias[:bit] for bit in self.bit_list]
        with torch.no_grad():
            self._momentum_update_key_encoder()
            encode_x2 = self.encoder_k(x)
            outputs_x2 = [torch.matmul(encode_x2, self.hash_layer_k.weight[:bit, :].t()) + self.hash_layer_k.bias[:bit] for bit in self.bit_list]
        # return encode_x, encode_x2
        output = [(o_x, o_x2) for o_x, o_x2 in zip(outputs_x, outputs_x2)]
        return output

class RML_layer(nn.Module):
    # 2024.2.3
    def __init__(self, in_feature, bit_list=[8,16,32,64,128]):
        super(RML_layer, self).__init__()
        self.hash_layers = nn.ModuleList([nn.Linear(in_feature, bit) for bit in bit_list])
        for hash_layer in self.hash_layers:
            hash_layer.weight.data.normal_(0, 0.01)
            hash_layer.bias.data.fill_(0.0)
        self.bn_finals = nn.ModuleList([nn.BatchNorm1d(bit) for bit in bit_list])

    def forward(self, x):
        xs = [hash_layer(x) for hash_layer in self.hash_layers]
        outputs = [bn_final(xt) for bn_final, xt in zip(self.bn_finals, xs)]
        return outputs

class RML_E_layer(nn.Module):
    # 2024.2.3
    def __init__(self, in_feature, bit_list=[8,16,32,64,128]):
        super(RML_E_layer, self).__init__()
        self.bit_list = bit_list
        self.hash_layer = nn.Linear(in_feature, bit_list[-1])
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
        outputs = [torch.matmul(x, self.hash_layer.weight[:bit, :].t()) + self.hash_layer.bias[:bit] for bit in self.bit_list]
        return outputs

class AllNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AllNorm1d, self).__init__()
        # 初始化参数 gamma 和 beta
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        # 为了数值稳定性而添加的小值
        self.eps = eps
        # 用于运行时均值和方差的动量
        self.momentum = momentum
        # 用于存储运行时均值和方差
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self, input):
        running_mean = self.running_mean.to(input.device)
        running_var = self.running_var.to(input.device)

        if self.training:
            # batch_mean = input.mean(dim=0)
            # batch_var = input.var(dim=0, unbiased=False)
            batch_mean = input.mean()
            batch_var = input.var(unbiased=False)
            self.running_mean = (1 - self.momentum) * running_mean.detach() + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * running_var.detach() + self.momentum * batch_var
            # 标准化
            input_normalized = (input - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # 使用运行时均值和方差进行标准化
            input_normalized = (input - running_mean) / torch.sqrt(running_var + self.eps)

        # 应用缩放和偏移
        return self.gamma * input_normalized + self.beta

class RML_E_layer_batchnrom(nn.Module):
    # 2024.2.25 
    # 为每个长度增加一个加和的变量
    def __init__(self, in_feature, bit_list=[16,32,64]):
        super(RML_E_layer_batchnrom, self).__init__()
        self.bit_list = bit_list
        self.hash_layer = nn.Linear(in_feature, bit_list[-1])
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        self.norm_finals = nn.ModuleList([nn.BatchNorm1d(bit) for bit in bit_list])
        
    def forward(self, x):
        outputs = [torch.matmul(x, self.hash_layer.weight[:bit, :].t()) + self.hash_layer.bias[:bit] for bit in self.bit_list]
        outputs_norm = [norm_final(output) for norm_final, output in zip(self.norm_finals, outputs)]
        return outputs, outputs_norm

class RML_E_layer_norm(nn.Module):
    # 2024.2.27
    # 测试，让每个bias是独立的
    def __init__(self, in_feature, bit_list=[16,32,64]):
        super(RML_E_layer_norm, self).__init__()
        self.bit_list = bit_list
        self.hash_layer = nn.Linear(in_feature, bit_list[-1])
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(bit)) for bit in bit_list])
        
    def forward(self, x):
        outputs = [torch.matmul(x, self.hash_layer.weight[:bit, :].t()) for bit in self.bit_list]
        outputs_norm = [output + bias for bias, output in zip(self.biases, outputs)]
        return outputs, outputs_norm

class RML_E_layer_allnorm(nn.Module):
    # 2024.2.25 
    # 为每个长度增加一个加和的变量
    def __init__(self, in_feature, bit_list=[16,32,64]):
        super(RML_E_layer_allnorm, self).__init__()
        self.bit_list = bit_list
        self.hash_layer = nn.Linear(in_feature, bit_list[-1])
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        self.norm_finals = nn.ModuleList([AllNorm1d(bit) for bit in bit_list])
        
    def forward(self, x):
        outputs = [torch.matmul(x, self.hash_layer.weight[:bit, :].t()) + self.hash_layer.bias[:bit] for bit in self.bit_list]
        outputs_norm = [norm_final(output) for norm_final, output in zip(self.norm_finals, outputs)]
        return outputs, outputs_norm