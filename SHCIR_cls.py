import torch
from utils.tools import *
from bl_network import ResNetClass
# import os
import torch.optim as optim
import time
import numpy as np
torch.multiprocessing.set_sharing_strategy('file_system')  # multiprocessing to read files
import torch.nn as nn
import random
from tqdm import tqdm

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_config():
    config = {
        # "remarks": "OurLossWithPair",
        "seed": 60,
        "optim_parms":{
            "lr": 5e-3,
            "momentum": 0.9,
            "weight_decay": 1e-4,
        },
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        # "dataset": "imagenet",
        # "dataset": "cifar-1",
        # "dataset": "nuswide_21",
        # "dataset": "cifar10",
        "dataset": "imagenet",
        "test_map": 2,
        "stop_iter": 7,
        "epoch": 1000,
        "device": torch.device('cuda:2'),
        "n_gpu": torch.cuda.device_count(),
        "max_norm": 5.0,
        "info": "SHCIR_cls"
    }
    config = config_dataset(config)
    return config

class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()
        self.CELoss = torch.nn.CrossEntropyLoss()

    def forward(self, probs, labels, w):
        labels = labels.float()
        # print("label:", labels)
        # labels = torch.argmax(torch.Tensor(labels)).long()
        # print("label:", labels)

        celoss = self.CELoss(probs, labels)
        Q_loss = (w.abs()-1).pow(2).mean()
        return celoss + Q_loss

def top_k_accuracy(output, target, k=1):
    with torch.no_grad():
        _, predicted = torch.max(output.data, 1)
        _, target = torch.max(target, 1)
        total_correct = (predicted == target).sum().item()
        total = target.shape[0]
    return total_correct, total

def test_val(config, model, test_loader, device):
    model.eval()
    acc = 0
    total = 0
    with torch.no_grad():
        for img, label, ind in tqdm(test_loader):
            img = img.to(device)
            label = label.to(device)
            preds = model(img)
            temp_acc,  temp_batch = top_k_accuracy(preds, label, k=1)
            acc += temp_acc
            total += temp_batch
    return 100 * acc / total

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    print(f"lr is {lr}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_val(config):
    device = config['device']
    # torch.cuda.set_device(device)
    # if config['dataset'] == 'imagenet':
    #     train_loader, test_loader, database_loader, num_train, num_test, num_database = get_imagenet_data(config)
    # else:
    #     train_loader, test_loader, database_loader, num_train, num_test, num_database = get_data(config)
    train_loader, test_loader, database_loader, num_train, num_test, num_database = get_data(config)
    net = ResNetClass(config['n_class']).to(device)
    optimizer = optim.SGD(net.parameters(), lr=config['optim_parms']['lr'], weight_decay=config['optim_parms']['weight_decay'])
    config['num_train'] = num_train

    Best_acc = 0
    print('finish load config')

    count = 0
    # print(f"config: {str(config)}")
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    cross_entropy_loss = myLoss()

    for epoch in range(config['epoch']):
        current_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
        print(
            f"{epoch + 1}/{config['epoch']} {current_time} dataset: {config['dataset']} training...")
        # adjust_learning_rate(optimizer, epoch, config['optim_parms']['lr'])
        train_loss = 0
        train_acc = 0
        total = 0
        net.train()
        for img, label, ind in tqdm(train_loader):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            probs = net(img)
            loss = cross_entropy_loss(probs, label, net.model_resnet.fc.weight)
            # print("prob", probs)
            # print("label", label)
            temp_acc, temp_batch= top_k_accuracy(probs, label, k=1)
            train_acc += temp_acc
            total += temp_batch
            if config['n_gpu'] > 1:
                loss = loss.mean()
            train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config['max_norm'])
            optimizer.step()

        train_loss /= len(train_loader)
        train_acc /= total
        print(f"train loss: {train_loss}, train accuracy: {100 * train_acc}")

        if (epoch + 1) % config['test_map'] == 0:
            acc = test_val(config, net, test_loader, device)
            if acc > Best_acc:
                Best_acc = acc
                count = 0
                # print(f'save in ./results/class_model')
                # torch.save(net.state_dict(), f'./results/class_model/{config["cls_model"]}_{config["dataset"]}_model_w_{config["optim_parms"]["lr"]}.pt')
                net.eval()
                with torch.no_grad():
                    W = net.model_resnet.fc.weight.cpu().numpy()
                np.save(f'./tmp_file/SHCIR/{config["cls_model"]}_{config["dataset"]}_class_head.npy', W)
            else:
                if count == config['stop_iter']:
                    print(f"valid acc: {Best_acc}")
                    end_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
                    # with open(f'./results/class_model/map_result.txt', 'a') as f:
                    #     f.write('valid: ' + str(Best_acc) + '\t' + 'start time: ' + str(start_time) +
                    #             '\t' + 'end_time:' + str(end_time) + str(config) +'\n')
                    break
                count += 1
            print(
                f"{epoch + 1}/{config['epoch']} {current_time} dataset: {config['dataset']} Best acc: {Best_acc}, current acc: {acc}")
        if (epoch + 1) == config['epoch']:
            print(f"valid acc: {Best_acc}")
            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            net.eval()
            with torch.no_grad():
                W = net.fc.weight.cpu().numpy()
            np.save(f'./tmp_file/SHCIR/{config["cls_model"]}_{config["dataset"]}_class_head.npy', W)


if __name__ == '__main__':
    config = get_config()
    best_result = 0
    config['cls_model'] = 'ResNet'
    setup_seed(config['seed'])
    train_val(config)
