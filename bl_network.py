import torch.nn as nn
from torchvision import models
import torch
from torch.autograd import Function

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
        feat = self.avgpool(feat4)
        feat = feat.view(feat.size(0), -1)
        x = self.hash_layer(feat)
        if is_feat:
            return feat, x
        else:
            return x

class ViT_B(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(ViT_B, self).__init__()
        self.net = models.vit_b_16(pretrained=True)
        # self.net = models.vit_l_16(pretrained=True)
        self.net.heads = nn.Linear(768, hash_bit)
        # self.net.heads = nn.Linear(1024, hash_bit)

    def forward(self, x, is_feat=False):
        x = self.net(x)
        return x

class MoCo(nn.Module):
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

class ViT_B_f(nn.Module):
    def __init__(self):
        super(ViT_B_f, self).__init__()
        self.vit = models.vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()

    def forward(self, x):
        x = self.vit(x)        
        return x

class MoCo_RML(nn.Module):
    def __init__(self, config):
        super(MoCo_RML, self).__init__()
        self.m = config['mome']
        self.encoder_q = ResNet_f()
        self.encoder_k = ResNet_f()
        self.bit_list = config["bit_list"]
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
        # outputs_x = [torch.matmul(encode_x, self.hash_layer_q.weight[:bit, :].t()) + self.hash_layer_q.bias[:bit] for bit in self.bit_list]
        with torch.no_grad():
            self._momentum_update_key_encoder()
            encode_x2 = self.encoder_k(x)
            # outputs_x2 = [torch.matmul(encode_x2, self.hash_layer_k.weight[:bit, :].t()) + self.hash_layer_k.bias[:bit] for bit in self.bit_list]
        # return encode_x, encode_x2
        return (encode_x, encode_x2)
        # output = [(o_x, o_x2) for o_x, o_x2 in zip(outputs_x, outputs_x2)]
        # return output

class MoCo_RML_head(nn.Module):
    def __init__(self, in_feature, bit_list, config):
        super(MoCo_RML_head, self).__init__()
        self.m = config['mome']
        self.bit_list = bit_list
        self.hash_layer_q = nn.Linear(in_feature, self.bit_list[-1])
        self.hash_layer_k = nn.Linear(in_feature, self.bit_list[-1])
        self.hash_layer_q.weight.data.normal_(0, 0.01)
        self.hash_layer_k.bias.data.fill_(0.0)
        for param_q, param_k in zip(self.hash_layer_q.parameters(), self.hash_layer_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.hash_layer_q.parameters(), self.hash_layer_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    def forward(self, x):
        (encode_x, encode_x2) = x
        # encode_x = self.encoder_q(x)
        outputs_x = [torch.matmul(encode_x, self.hash_layer_q.weight[:bit, :].t()) + self.hash_layer_q.bias[:bit] for bit in self.bit_list]
        with torch.no_grad():
            self._momentum_update_key_encoder()
            outputs_x2 = [torch.matmul(encode_x2, self.hash_layer_k.weight[:bit, :].t()) + self.hash_layer_k.bias[:bit] for bit in self.bit_list]
        output = [(o_x, o_x2) for o_x, o_x2 in zip(outputs_x, outputs_x2)]
        return output


class hash(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class RML_E_layer(nn.Module):
    def __init__(self, in_feature, bit_list=[8,16,32,64,128]):
        super(RML_E_layer, self).__init__()
        self.bit_list = bit_list
        self.hash_layer = nn.Linear(in_feature, bit_list[-1])
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
        outputs = [torch.matmul(x, self.hash_layer.weight[:bit, :].t()) + self.hash_layer.bias[:bit] for bit in self.bit_list]
        return outputs

class ResNetClass(nn.Module):
    def __init__(self, label_size, pretrained=True):
        super(ResNetClass, self).__init__()
        self.model_resnet = models.resnet50(pretrained=pretrained)
        self.model_resnet.fc = nn.Linear(self.model_resnet.fc.in_features, label_size)
        self.BN = nn.BatchNorm1d(label_size, momentum=0.1)

    def forward(self, x):
        feat = self.model_resnet(x)
        return self.BN(feat)