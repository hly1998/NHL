from utils.tools import *
import torch
import random
import logging
import psutil
from bit_length.model_grand import grand_train_val
# from network import ResNet
from bl_network import ResNet_f, RML_layer, RML_E_layer, ResNet, RML_E_layer_norm, RML_E_layer_batchnrom, RML_E_layer_allnorm, MoCo, MoCo_RML
import argparse
import wandb
from bit_length.MDSH import mdsh_config
import os

# count = psutil.cpu_count()
# print(f"逻辑cpu的数量是{count}")
# p = psutil.Process()
# p.cpu_affinity(list(random.sample(range(1, count), 6)))
# torch.multiprocessing.set_sharing_strategy('file_system')

os.environ["WANDB_MODE"] = "offline"

def get_argparser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='imagenet', help='可以选择的有imagenet, cifar10-1, cifar10-2, nuswide_21')
        parser.add_argument('--optimizer', type=str, default='Adam')
        parser.add_argument('--net', type=str, default='ResNet_RML')
        parser.add_argument('--mode', type=str, default='simple')
        parser.add_argument('--info', type=str, default='CSQ')
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--weight_decay', type=float, default=10 ** -5)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--epoch', type=int, default=1500)
        parser.add_argument('--test_map', type=int, default=10)
        parser.add_argument('--stop_iter', type=int, default=5, help='控制在效果没有提升多少次后停止运行')
        parser.add_argument('--device', type=int, default=1)
        parser.add_argument('--bit', type=int, default=64)
        parser.add_argument('--distill', action='store_true', help='是否采用自蒸馏')
        parser.add_argument('--distill_weight', type=float, default=0.1)
        parser.add_argument('--analysis', action='store_true', help='是否去分析梯度情况')
        parser.add_argument('--heuristic_weight', action='store_true', help='是否采用启发式权重作用于损失目标，如果是，需要修改下面的值')
        parser.add_argument('--heuristic_weight_value', nargs='+', type=float, default=[2.0, 1.5, 1.0])
        parser.add_argument('--step_update', action='store_true', help='是否采用依次更新的方式')
        parser.add_argument('--step_update_value', nargs='+', type=int, default=[30, 60])
        parser.add_argument('--space', action='store_true', help='是否采用空间对齐')
        parser.add_argument('--space_weight', type=float, default=0.01)
        # parser.add_argument('--norm', action='store_true', help='是否采用norm')
        parser.add_argument('--norm', type=str, default='no', help='no:不用norm, batch:batchnorm, bias:分离的bias')
        parser.add_argument('--wandb', action='store_true', help='是否使用wandb记录结果')
        return parser

def get_config(args):
    optimizer_map = {
        'SGD': torch.optim.SGD,
        'ASGD': torch.optim.ASGD,
        'Adam': torch.optim.Adam,
        'Adamax': torch.optim.Adamax,
        'Adagrad': torch.optim.Adagrad,
        'Adadelta': torch.optim.Adadelta,
        'Rprop': torch.optim.Rprop,
        'RMSprop': torch.optim.RMSprop
    }
    net_map = {
        "ResNet": ResNet,
        'ResNet_RML': ResNet_f,
    }
    config = {
        "optimizer": {"type": optimizer_map[args.optimizer], "optim_params": {"lr": args.lr, "weight_decay": args.weight_decay}},
        "info": args.info,
        "net": net_map[args.net],
        "mode": args.mode, # simple: 直接训练 RML: 多任务形式 - RML_E: 共享参数
        "dataset": args.dataset,
        "epoch": args.epoch,
        "distill": args.distill,
        "distill_weight": args.distill_weight,
        "analysis": args.analysis,
        "heuristic_weight": args.heuristic_weight,
        "heuristic_weight_value": args.heuristic_weight_value,
        "step_update": args.step_update,
        "step_update_value": args.step_update_value,
        "space": args.space,
        "space_weight": args.space_weight,
        "norm": args.norm,
        "wandb": args.wandb,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "test_map": args.test_map,
        "device": torch.device("cuda:{}".format(args.device)),
        "stop_iter": args.stop_iter,
        "bit_list": [8,16,32,64,128],
        # "bit_list": [64,],
        "in_feature": 2048
    }
    config = config_dataset(config)
    return config

if __name__ == "__main__":
    argparser = get_argparser()
    args = argparser.parse_args()
    config = get_config(args)
    print(config)
    if config["info"] == "MDSH":
        config = mdsh_config(config)

    if config["mode"] == "simple":
        for bit in config["bit_list"]:
            if config["wandb"]:
                wandb.init(
                    # reinit=True,
                    # set the wandb project where this run will be logged
                    project="RML",
                    name=f"{config['info']}_{config['dataset']}_{config['mode']}_bit:{bit}",
                    # track hyperparameters and run metadata
                    config={
                        "learning_rate": args.lr,
                        "architecture": "CNN",
                        "test_map": args.test_map,
                        "stop_iter": args.stop_iter,
                        "optimizer": args.optimizer,
                        "net": args.net
                    },
                )
            if config["info"] == "MDSH":
                net = MoCo(config, bit).to(config["device"])
            else:
                net = config["net"](bit).to(config["device"])
            logging.basicConfig(filename=f"bl_logs/{config['info']}_{config['mode']}_{config['dataset']}_{str(bit)}.log", level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # if config["info"] in ["CSQ", "DCH", "DTSH"]:
            #     grand_train_val(config, bit, net)
            grand_train_val(config, bit, net)
            if config["wandb"]:
                wandb.finish()
    elif config["mode"] in ['RML', 'RML_E']:
        if config["wandb"]:
            wandb.init(
                # set the wandb project where this run will be logged
                # reinit=True,
                project="RML",
                name=f"{config['info']}_{config['dataset']}_{config['mode']}_d:{config['distill']}_a:{config['analysis']}",
                # track hyperparameters and run metadata
                config={
                    "ana": config['analysis'],
                    "learning_rate": args.lr,
                    "architecture": "CNN",
                    "test_map": args.test_map,
                    "stop_iter": args.stop_iter,
                    "optimizer": args.optimizer,
                    "net": args.net,
                    "distill": args.distill
                },
            )
        if config["mode"] == "RML":
            backbone = config["net"]()
            head = RML_layer(config["in_feature"])
            net = torch.nn.Sequential(backbone, head).to(config["device"])
        elif config["mode"] == "RML_E":
            if config["info"] == "MDSH":
                net = MoCo_RML(config).to(config["device"])
            else:
                backbone = config["net"]()
                # head = RML_E_layer(config["in_feature"])
                if config["norm"]=='bias':
                    head = RML_E_layer_norm(config["in_feature"])
                elif config["norm"]=='batch':
                    head = RML_E_layer_batchnrom(config["in_feature"])
                elif config["norm"]=='all':
                    head = RML_E_layer_allnorm(config["in_feature"])
                elif config["norm"]=='no':
                    head = RML_E_layer(config["in_feature"])
                net = torch.nn.Sequential(backbone, head).to(config["device"])
        logging.basicConfig(filename=f"bl_logs/{config['info']}_{config['mode']}_distill:{config['distill']}_ana:{config['analysis']}_{config['dataset']}.log", level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # if config["info"] in ["CSQ", "DCH", "DTSH"]:
        #     grand_train_val(config, 64, net) # 随机指定一个bit，这里bit无作用
        grand_train_val(config, 64, net) # 随机指定一个bit，这里bit无作用
        if config["wandb"]:
            wandb.finish()
    

