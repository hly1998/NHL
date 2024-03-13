# import numpy as np
# import torch.utils.data as util_data
# from torchvision import transforms
# import torch
# from PIL import Image
# from tqdm import tqdm
# import torchvision.datasets as dsets
# import os
# import json
# import logging.config
# import random

# # # 设置随机数种子
# # seed = 1234
# # torch.manual_seed(seed) # 为CPU设置随机种子
# # torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
# # np.random.seed(seed)  # Numpy module.
# # random.seed(seed)  # Python random module.	
# # # torch.backends.cudnn.benchmark = False
# # torch.backends.cudnn.deterministic = True

# def config_dataset(config):
#     if config["dataset"] == 'cifar100':
#         config["topK"] = 500
#         config["n_class"] = 100
#     elif "cifar" in config["dataset"]:
#         config["topK"] = -1
#         config["n_class"] = 10
#     elif config["dataset"] in ["nuswide_21", "nuswide_21_m"]:
#         config["topK"] = 5000
#         config["n_class"] = 21
#     elif config["dataset"] == "nuswide_81_m":
#         config["topK"] = 5000
#         config["n_class"] = 81
#     elif config["dataset"] == "coco":
#         config["topK"] = 5000
#         config["n_class"] = 80
#     elif config["dataset"] == "imagenet":
#         config["topK"] = 1000
#         config["n_class"] = 100
#     elif config["dataset"] == "mirflickr":
#         config["topK"] = -1
#         config["n_class"] = 38
#     elif config["dataset"] == "voc2012":
#         config["topK"] = -1
#         config["n_class"] = 20

#     config["data_path"] = "/dataset/" + config["dataset"] + "/"
#     if config["dataset"] == "imagenet":
#         config["data_path"] = "/data/lyhe/BNN/DeepHash-pytorch-master/data/imagenet/"
#     if config["dataset"] == "nuswide_21":
#         config["data_path"] = "/data/lyhe/BNN/DeepHash-pytorch-master/data/nuswide_21/"
#     if config["dataset"] in ["nuswide_21_m", "nuswide_81_m"]:
#         config["data_path"] = "/dataset/nus_wide_m/"
#     if config["dataset"] == "coco":
#         config["data_path"] = "/dataset/COCO_2014/"
#     if config["dataset"] == "voc2012":
#         config["data_path"] = "/dataset/"
#     config["data"] = {
#         "train_set": {"list_path": "./data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
#         "database": {"list_path": "./data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
#         "test": {"list_path": "./data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
#     return config

# class ImageList(object):

#     def __init__(self, data_path, image_list, transform):
#         self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
#         self.transform = transform

#     def __getitem__(self, index):
#         path, target = self.imgs[index]
#         img = Image.open(path).convert('RGB')
#         img = self.transform(img)
#         return img, target, index

#     def __len__(self):
#         return len(self.imgs)

# class ImageList_special(object):
#     # 使用特殊的采样方法，具体地，让一个batch中的同类出现次数更多
#     # n_positive: 同类样本出现次数
#     def __init__(self, data_path, image_list, transform, num_classes=100, n_positive=8):
#         # self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
#         self.n_positive = n_positive
#         self.imgs = [data_path + val.split()[0] for val in image_list]
#         self.targets = [np.array([int(la) for la in val.split()[1:]]) for val in image_list]
#         self.transform = transform
#         self.cls_positive = [[] for i in range(num_classes)]
#         for i in range(len(self.imgs)):
#             cls_idx = np.argmax(self.targets[i])
#             self.cls_positive[cls_idx].append(i)

#     def __getitem__(self, index):
#         path = self.imgs[index]
#         target = self.targets[index]
#         # 随机找到n_positive个图片
#         cls_idx = np.argmax(target)
#         idxs = np.random.choice(self.cls_positive[cls_idx], self.n_positive, replace=False)
#         imgs = []
#         targets = []
#         for idx in idxs:
#             img = Image.open(self.imgs[idx]).convert('RGB')
#             img = self.transform(img)
#             imgs.append(img)
#             targets.append(self.targets[idx])
#         imgs = np.stack(imgs)
#         targets = np.stack(targets)
#         return imgs, targets, index

#     def __len__(self):
#         return len(self.imgs)

# def image_transform(resize_size, crop_size, data_set):
#     if data_set == "train_set":
#         step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
#     else:
#         step = [transforms.CenterCrop(crop_size)]
#     return transforms.Compose([transforms.Resize(resize_size)]
#                               + step +
#                               [transforms.ToTensor(),
#                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                     std=[0.229, 0.224, 0.225])
#                                ])

# def image_transform_v9(resize_size, crop_size, data_set):
#     if data_set == "train_set":
#         return transforms.Compose([transforms.Resize(resize_size),
#                                    transforms.RandomCrop(crop_size),
#                                    transforms.ToTensor()])
#     else:
#         step = [transforms.CenterCrop(crop_size)]
#         return transforms.Compose([transforms.Resize(resize_size)]
#                                 + step +
#                                 [transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                         std=[0.229, 0.224, 0.225])
#                                 ])

# def image_transform_ours(resize_size, crop_size, data_set):
#     if data_set == "train_set":
#         return transforms.Compose([transforms.Resize(resize_size),
#                                    transforms.RandomHorizontalFlip(),
#                                    transforms.RandomCrop(crop_size),
#                                    transforms.ToTensor()])
#     else:
#         step = [transforms.CenterCrop(crop_size)]
#         return transforms.Compose([transforms.Resize(resize_size)]
#                                 + step +
#                                 [transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                         std=[0.229, 0.224, 0.225])
#                                 ])

# class MyCIFAR10(dsets.CIFAR10):
#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]
#         img = Image.fromarray(img)
#         img = self.transform(img)
#         target = np.eye(10, dtype=np.int8)[np.array(target)]
#         return img, target, index

# class MyCIFAR100(dsets.CIFAR100):
#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]
#         img = Image.fromarray(img)
#         img = self.transform(img)
#         target = np.eye(100, dtype=np.int8)[np.array(target)]
#         return img, target, index


def cifar_dataset(config):
    batch_size = config["batch_size"]

    train_size = 500
    test_size = 100

    if config["dataset"] == "cifar10-2":
        train_size = 5000
        test_size = 1000

    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # cifar_dataset_root = '/dataset/cifar/'
    cifar_dataset_root = '/data/lyhe/KD/CIBHash-main/data/cifar10/'
    # Dataset
    train_dataset = MyCIFAR10(root=cifar_dataset_root,
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root=cifar_dataset_root,
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR10(root=cifar_dataset_root,
                                 train=False,
                                 transform=transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if config["dataset"] == "cifar10":
        # test:1000, train:5000, database:54000
        pass
    elif config["dataset"] == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))
    elif config["dataset"] == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4)

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]

def cifar100_dataset(config):
    # 2024.1.4 新增一个数据集
    batch_size = config["batch_size"]

    train_size = 100
    test_size = 50

    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # cifar_dataset_root = '/dataset/cifar/'
    cifar_dataset_root = '/data/lyhe/BNN/DeepHash-pytorch-master/data/cifar100/'
    # Dataset
    train_dataset = MyCIFAR100(root=cifar_dataset_root,
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR100(root=cifar_dataset_root,
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR100(root=cifar_dataset_root,
                                 train=False,
                                 transform=transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(100):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    # if config["dataset"] == "cifar10":
    #     # test:1000, train:5000, database:54000
    #     pass
    # elif config["dataset"] == "cifar10-1":
    #     # test:1000, train:5000, database:59000
    #     database_index = np.concatenate((train_index, database_index))
    # elif config["dataset"] == "cifar10-2":
    #     # test:10000, train:50000, database:50000
    #     database_index = train_index
    database_index = np.concatenate((train_index, database_index))
    
    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4)

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]
        

# def get_data(config):
#     def _init_fn(worker_id):
#         np.random.seed(int(8888)+worker_id)

#     if config["dataset"] == 'cifar100':
#         return cifar100_dataset(config)

#     if "cifar" in config["dataset"]:
#         return cifar_dataset(config)

#     dsets = {}
#     dset_loaders = {}
#     data_config = config["data"]

#     for data_set in ["train_set", "test", "database"]:
#         if config["info"] in ["MBHash_v10_v3","Ours_v4"] and data_set == 'train_set':
#             # 2023.12.11 特殊的采样形式
#             print("run MBHash_v10_v3 or Ours_v4")
#             dsets[data_set] = ImageList_special(config["data_path"],
#                                         open(data_config[data_set]["list_path"]).readlines(),
#                                         transform=image_transform_v9(config["resize_size"], config["crop_size"], data_set), n_positive=config['n_positive'])
#         elif config["info"] in ["MBHash_v9", "MBHash_v10", "MBHash_v10_v2", "MBHash_v10_v3", "MBHash_v11", "MBHash_v12", "MBHash_v13", "MBHash_v14", "MBHash_v15", "MBHash_v16", "Ours", "Ours_v3", "Ours_v4"]:
#             print("run MBHash_v9 or MBHash_v10 or MBHash_v10_v2 or MBHash_v10_v3 or MBHash_v11 or MBHash_v12 or MBHash_v13 or MBHash_v14 or MBHash_v15 or MBHash_v16 or Ours or Ours_v3 or Ours_v4")
#             dsets[data_set] = ImageList(config["data_path"],
#                                         open(data_config[data_set]["list_path"]).readlines(),
#                                         transform=image_transform_v9(config["resize_size"], config["crop_size"], data_set))
#         elif  config["info"] in ["Ours_v2"]:
#             print("run Ours_v2")
#             dsets[data_set] = ImageList(config["data_path"],
#                                         open(data_config[data_set]["list_path"]).readlines(),
#                                         transform=image_transform_ours(config["resize_size"], config["crop_size"], data_set))
#         else:
#             dsets[data_set] = ImageList(config["data_path"],
#                                         open(data_config[data_set]["list_path"]).readlines(),
#                                         transform=image_transform(config["resize_size"], config["crop_size"], data_set))

#         # 用于测试，取少量数据
#         # dsets[data_set] = ImageList(config["data_path"],
#         #                             open(data_config[data_set]["list_path"]).readlines()[:500],
#         #                             transform=image_transform(config["resize_size"], config["crop_size"], data_set))
#         print(data_set, len(dsets[data_set]))
        
#         dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
#                                                       batch_size=data_config[data_set]["batch_size"],
#                                                       shuffle= (data_set == "train_set") , num_workers=4)

#         # dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
#         #                                               batch_size=data_config[data_set]["batch_size"],
#         #                                               shuffle= (data_set == "train_set") , num_workers=4, worker_init_fn=_init_fn)

#     return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
#            len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])


# def compute_result(dataloader, net, device):
#     bs, clses = [], []
#     net.eval()
#     for img, cls, _ in tqdm(dataloader):
#         clses.append(cls)
#         out = net(img.to(device))
#         if isinstance(out, tuple):
#             bs.append(out[0].data.cpu())
#         else:
#             bs.append((out).data.cpu())
#         # bs.append((net(img.to(device))).data.cpu())
#     return torch.cat(bs).sign(), torch.cat(clses)

# # def compute_result_MDSH(dataloader, net, device):
# #     bs, clses = [], []
# #     net.eval()
# #     for img, cls, _ in tqdm(dataloader):
# #         clses.append(cls)
# #         out = net(img.to(device))
# #         if isinstance(out, tuple):
# #             bs.append(out[0].data.cpu())
# #         else:
# #             bs.append((out).data.cpu())
# #         # bs.append((net(img.to(device))).data.cpu())
# #     return torch.cat(bs).sign(), torch.cat(clses)

# def compute_result_MDSH(dataloader, net, device, T, label_vector):
#     bs, clses = [], []
#     net.eval()
#     for img, cls, _ in tqdm(dataloader):
#         clses.append(cls)
#         bs.append((net(img.to(device), T, label_vector)[0]).data.cpu())
#     return torch.cat(bs).sign(), torch.cat(clses)

# def CalcHammingDist(B1, B2):
#     q = B2.shape[1]
#     distH = 0.5 * (q - np.dot(B1, B2.transpose()))
#     return distH

# def CalcTopMap(rB, qB, retrievalL, queryL, topk):
#     num_query = queryL.shape[0]
#     topkmap = 0
#     for iter in tqdm(range(num_query)):
#         gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
#         hamm = CalcHammingDist(qB[iter, :], rB)
#         ind = np.argsort(hamm)
#         gnd = gnd[ind]

#         tgnd = gnd[0:topk]
#         tsum = np.sum(tgnd).astype(int)
#         if tsum == 0:
#             continue
#         count = np.linspace(1, tsum, tsum)

#         tindex = np.asarray(np.where(tgnd == 1)) + 1.0
#         topkmap_ = np.mean(count / (tindex))
#         topkmap = topkmap + topkmap_
#     topkmap = topkmap / num_query
#     return topkmap


# def CalcTopMap_for_v13(rB, qB, retrievalL, queryL, topk, weights):
#     # 10.22 一点都没改，因为先去吃饭了，先跑一个不改的看看效果
#     num_query = queryL.shape[0]
#     topkmap = 0
#     # 获取全体database所对应的weight
#     rB_weight = weights
#     for iter in tqdm(range(num_query)):
#         gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
#         hamm = CalcHammingDist(qB[iter, :], rB)
#         ind = np.argsort(hamm)
#         gnd = gnd[ind]

#         tgnd = gnd[0:topk]
#         tsum = np.sum(tgnd).astype(int)
#         if tsum == 0:
#             continue
#         count = np.linspace(1, tsum, tsum)

#         tindex = np.asarray(np.where(tgnd == 1)) + 1.0
#         topkmap_ = np.mean(count / (tindex))
#         topkmap = topkmap + topkmap_
#     topkmap = topkmap / num_query
#     return topkmap


# # faster but more memory
# def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):
#     num_query = queryL.shape[0]
#     num_gallery = retrievalL.shape[0]
#     topkmap = 0
#     prec = np.zeros((num_query, num_gallery))
#     recall = np.zeros((num_query, num_gallery))
#     for iter in tqdm(range(num_query)):
#         gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
#         hamm = CalcHammingDist(qB[iter, :], rB)
#         ind = np.argsort(hamm)
#         gnd = gnd[ind]

#         tgnd = gnd[0:topk]
#         tsum = np.sum(tgnd).astype(int)
#         if tsum == 0:
#             continue
#         count = np.linspace(1, tsum, tsum)
#         all_sim_num = np.sum(gnd)

#         prec_sum = np.cumsum(gnd)
#         return_images = np.arange(1, num_gallery + 1)

#         prec[iter, :] = prec_sum / return_images
#         recall[iter, :] = prec_sum / all_sim_num

#         assert recall[iter, -1] == 1.0
#         assert all_sim_num == prec_sum[-1]

#         tindex = np.asarray(np.where(tgnd == 1)) + 1.0
#         topkmap_ = np.mean(count / (tindex))
#         topkmap = topkmap + topkmap_
#     topkmap = topkmap / num_query
#     index = np.argwhere(recall[:, -1] == 1.0)
#     index = index.squeeze()
#     prec = prec[index]
#     recall = recall[index]
#     cum_prec = np.mean(prec, 0)
#     cum_recall = np.mean(recall, 0)

#     return topkmap, cum_prec, cum_recall

# # https://github.com/chrisbyd/DeepHash-pytorch/blob/master/validate.py
# def validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset):
#     device = config["device"]
#     # print("calculating test binary code......")
#     # if config["info"] in ("MDSH", "MDSH_XNOR", "MDSH_ReactNet"):
#     #     print("eval by MDSH or MDSH_XNOR or MDSH_ReactNet")
#     #     tst_binary, tst_label = compute_result_MDSH(test_loader, net, device, None, None)
#     #     # print("calculating dataset binary code.......")
#     #     trn_binary, trn_label = compute_result_MDSH(dataset_loader, net, device, None, None)
#     # else:
#     #     tst_binary, tst_label = compute_result(test_loader, net, device=device)
#     #     # print("calculating dataset binary code.......")
#     #     trn_binary, trn_label = compute_result(dataset_loader, net, device=device)
    
#     tst_binary, tst_label = compute_result(test_loader, net, device=device)
#     # print("calculating dataset binary code.......")
#     trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

#     if "pr_curve_path" not in  config:
#         # if config["info"] in ("MBHash_v13",):
#         #     print("weight distance calculate for MBHash_v13")
#         #     mAP = CalcTopMap_for_v13(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), config["topK"])
#         # else:
#         mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), config["topK"])
#     else:
#         # need more memory
#         mAP, cum_prec, cum_recall = CalcTopMapWithPR(tst_binary.numpy(), tst_label.numpy(),
#                                                      trn_binary.numpy(), trn_label.numpy(),
#                                                      config["topK"])
#         index_range = num_dataset // 100
#         index = [i * 100 - 1 for i in range(1, index_range + 1)]
#         max_index = max(index)
#         overflow = num_dataset - index_range * 100
#         index = index + [max_index + i for i in range(1, overflow + 1)]
#         c_prec = cum_prec[index]
#         c_recall = cum_recall[index]

#         pr_data = {
#             "index": index,
#             "P": c_prec.tolist(),
#             "R": c_recall.tolist()
#         }
#         os.makedirs(os.path.dirname(config["pr_curve_path"]), exist_ok=True)
#         with open(config["pr_curve_path"], 'w') as f:
#             f.write(json.dumps(pr_data))
#         print("pr curve save to ", config["pr_curve_path"])

#     if mAP > Best_mAP:
#         Best_mAP = mAP
#         if "save_path" in config:
#             save_path = os.path.join(config["save_path"], f'{config["dataset"]}_{bit}bits_{mAP}')
#             os.makedirs(save_path, exist_ok=True)
#             print("save in ", save_path)
#             np.save(os.path.join(save_path, "tst_label.npy"), tst_label.numpy())
#             np.save(os.path.join(save_path, "tst_binary.npy"), tst_binary.numpy())
#             np.save(os.path.join(save_path, "trn_binary.npy"), trn_binary.numpy())
#             np.save(os.path.join(save_path, "trn_label.npy"), trn_label.numpy())
#             torch.save(net.state_dict(), os.path.join(save_path, "{}_model.pt".format(net.__class__.__name__)))
#     # print(f"{config['info']} epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
#     print(f"epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
#     print(config)
#     return mAP, Best_mAP

# # 2023.5.16 加入对优化器的调整

# __optimizers = {
#     'SGD': torch.optim.SGD,
#     'ASGD': torch.optim.ASGD,
#     'Adam': torch.optim.Adam,
#     'Adamax': torch.optim.Adamax,
#     'Adagrad': torch.optim.Adagrad,
#     'Adadelta': torch.optim.Adadelta,
#     'Rprop': torch.optim.Rprop,
#     'RMSprop': torch.optim.RMSprop
# }

# def adjust_optimizer(optimizer, epoch, config):
#     """Reconfigures the optimizer according to epoch and config dict"""
#     def modify_optimizer(optimizer, setting):
#         if 'optimizer' in setting:
#             optimizer = __optimizers[setting['optimizer']](
#                 optimizer.param_groups)
#             logging.debug('OPTIMIZER - setting method = %s' %
#                           setting['optimizer'])
#         for param_group in optimizer.param_groups:
#             for key in param_group.keys():
#                 if key in setting:
#                     logging.debug('OPTIMIZER - setting %s = %s' %
#                                   (key, setting[key]))
#                     param_group[key] = setting[key]
#         return optimizer

#     if callable(config):
#         optimizer = modify_optimizer(optimizer, config(epoch))
#     else:
#         for e in range(epoch + 1):  # run over all epochs - sticky setting
#             if e in config:
#                 optimizer = modify_optimizer(optimizer, config[e])

#     return optimizer


#####################################################################

import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import os
import json
import logging.config
import random
from torchmetrics import RetrievalMAP

def config_dataset(config):
    if "cifar100" in config["dataset"]:
        config["topK"] = 500
        config["n_class"] = 100
    elif "cifar" in config["dataset"]:
        config["topK"] = -1
        config["n_class"] = 10
    elif config["dataset"] in ["nuswide_21", "nuswide_21_m"]:
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_81_m":
        config["topK"] = 5000
        config["n_class"] = 81
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] in ("imagenet", "imagenet-2"):
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "mirflickr":
        config["topK"] = -1
        config["n_class"] = 38
    elif config["dataset"] == "voc2012":
        config["topK"] = -1
        config["n_class"] = 20

    # config["data_path"] = "/dataset/" + config["dataset"] + "/"
    # if config["dataset"] == "imagenet":
    #     config["data_path"] = "/root/autodl-tmp/imagenet/"
    # if config["dataset"] == "nuswide_21":
    #     config["data_path"] = "/root/autodl-tmp/nuswide_21/"
    # # if config["dataset"] in ["nuswide_21_m", "nuswide_81_m"]:
    # #     config["data_path"] = "/dataset/nus_wide_m/"
    # # if config["dataset"] == "coco":
    # #     config["data_path"] = "/dataset/COCO_2014/"
    # # if config["dataset"] == "voc2012":
    # #     config["data_path"] = "/dataset/"
    # config["data"] = {
    #     "train_set": {"list_path": "/root/autodl-tmp/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
    #     "database": {"list_path": "/root/autodl-tmp/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
    #     "test": {"list_path": "/root/autodl-tmp/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    # return config

    config["data_path"] = "/dataset/" + config["dataset"] + "/"
    if config["dataset"] == "imagenet":
        config["data_path"] = "/root/autodl-tmp/data/imagenet/"
    if config["dataset"] == "imagenet-2":
        config["data_path"] = "/root/autodl-tmp/data/imagenet/"
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "/data/lyhe/data/nuswide_21/"
    if config["dataset"] in ["nuswide_21_m", "nuswide_81_m"]:
        config["data_path"] = "/dataset/nus_wide_m/"
    if config["dataset"] == "coco":
        config["data_path"] = "/dataset/COCO_2014/"
    if config["dataset"] == "voc2012":
        config["data_path"] = "/dataset/"
    # config["data"] = {
    #     "train_set": {"list_path": "./data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
    #     "database": {"list_path": "./data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
    #     "test": {"list_path": "./data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    config["data"] = {
        "train_set": {"list_path": config["data_path"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": config["data_path"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": config["data_path"] + "/test.txt", "batch_size": config["batch_size"]}}
    # if config["dataset"] == "imagenet-2":
    #     config["dataset"] = "imagenet" # 注意这里还是要改回保证路径是对的
    #     config["data"]["train_set"] = {"list_path": "./data/" + config["dataset"] + "/train-2.txt", "batch_size": config["batch_size"]}
    #     config["data"]["database"] = {"list_path": "./data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]}
    #     config["data"]["test"] = {"list_path": "./data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}
    return config

class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

class ImageList_special(object):
    # 使用特殊的采样方法，具体地，让一个batch中的同类出现次数更多
    # n_positive: 同类样本出现次数
    def __init__(self, data_path, image_list, transform, num_classes=100, n_positive=8):
        # self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.n_positive = n_positive
        self.imgs = [data_path + val.split()[0] for val in image_list]
        self.targets = [np.array([int(la) for la in val.split()[1:]]) for val in image_list]
        self.transform = transform
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(len(self.imgs)):
            cls_idx = np.argmax(self.targets[i])
            self.cls_positive[cls_idx].append(i)

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.targets[index]
        # 随机找到n_positive个图片
        cls_idx = np.argmax(target)
        idxs = np.random.choice(self.cls_positive[cls_idx], self.n_positive, replace=False)
        imgs = []
        targets = []
        for idx in idxs:
            img = Image.open(self.imgs[idx]).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
            targets.append(self.targets[idx])
        imgs = np.stack(imgs)
        targets = np.stack(targets)
        return imgs, targets, index

    def __len__(self):
        return len(self.imgs)

def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])

def image_transform_v9(resize_size, crop_size, data_set):
    if data_set == "train_set":
        return transforms.Compose([transforms.Resize(resize_size),
                                   transforms.RandomCrop(crop_size),
                                   transforms.ToTensor()])
    else:
        step = [transforms.CenterCrop(crop_size)]
        return transforms.Compose([transforms.Resize(resize_size)]
                                + step +
                                [transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                ])

def image_transform_ours(resize_size, crop_size, data_set):
    if data_set == "train_set":
        return transforms.Compose([transforms.Resize(resize_size),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop(crop_size),
                                   transforms.ToTensor()])
    else:
        step = [transforms.CenterCrop(crop_size)]
        return transforms.Compose([transforms.Resize(resize_size)]
                                + step +
                                [transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                ])

class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index

class MyCIFAR10_special(dsets.CIFAR10):

    def __init__(self, transform, num_classes=10, n_positive=8):
        # self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.n_positive = n_positive
        # self.imgs = self.data
        # self.targets = self.targets
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(len(self.data)):
            cls_idx = self.targets[i]
            self.cls_positive[cls_idx].append(i)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # 随机找到n_positive个图片
        cls_idx = target
        idxs = np.random.choice(self.cls_positive[cls_idx], self.n_positive, replace=False)
        imgs = []
        targets = []
        for idx in idxs:
            img = Image.fromarray(self.data[idx])
            # img = Image.open(self.imgs[idx]).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
            targets.append(self.targets[idx])
        imgs = np.stack(imgs)
        targets = np.stack(targets)
        targets = np.eye(10, dtype=np.int8)[np.array(targets)]
        return imgs, targets, index


class MyCIFAR100(dsets.CIFAR100):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(100, dtype=np.int8)[np.array(target)]
        return img, target, index

class MyCIFAR100_special(dsets.CIFAR100):
    def __init__(self, num_classes=100, n_positive=8, **kwargs):
        super().__init__(**kwargs)
        # self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.n_positive = n_positive
        # self.imgs = self.data
        # self.targets = self.targets
        self.cls_positive = [[] for i in range(num_classes)]
        # print("len data:", len(self.data))
        for i in range(10000):
            cls_idx = self.targets[i]
            self.cls_positive[cls_idx].append(i)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # 随机找到n_positive个图片
        cls_idx = target
        idxs = np.random.choice(self.cls_positive[cls_idx], self.n_positive, replace=False)
        imgs = []
        targets = []
        for idx in idxs:
            img = Image.fromarray(self.data[idx])
            # img = Image.open(self.imgs[idx]).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
            targets.append(self.targets[idx])
        imgs = np.stack(imgs)
        targets = np.stack(targets)
        targets = np.eye(100, dtype=np.int8)[np.array(targets)]
        return imgs, targets, index


def cifar100_dataset_for_ours(config):
    batch_size = config["batch_size"]

    train_size = 100
    test_size = 50

    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # cifar_dataset_root = '/dataset/cifar/'
    cifar_dataset_root = '/data/lyhe/BNN/DeepHash-pytorch-master/data/cifar100/'
    # Dataset
    
    explore_train_dataset = MyCIFAR100(root=cifar_dataset_root,
                              train=True,
                              transform=transform,
                              download=True)
    
    aggregation_train_dataset = MyCIFAR100_special(root=cifar_dataset_root,
                              train=True,
                              transform=transform,
                              download=True, 
                              n_positive=config['n_positive'])
    
    test_dataset = MyCIFAR100(root=cifar_dataset_root,
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR100(root=cifar_dataset_root,
                                 train=False,
                                 transform=transform)

    X = np.concatenate((explore_train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(explore_train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(100):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    database_index = np.concatenate((train_index, database_index))
    
    explore_train_dataset.data = X[train_index]
    explore_train_dataset.targets = L[train_index]
    aggregation_train_dataset.data = X[train_index]
    aggregation_train_dataset.targets = L[train_index]
    

    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", explore_train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    explore_train_loader = torch.utils.data.DataLoader(dataset=explore_train_dataset,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=4)
    
    aggregation_train_loader = torch.utils.data.DataLoader(dataset=aggregation_train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=64,
                                              shuffle=False,
                                              num_workers=4)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=64,
                                                  shuffle=False,
                                                  num_workers=4)

    return explore_train_loader, aggregation_train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]

def get_data(config):
    def _init_fn(worker_id):
        np.random.seed(int(8888)+worker_id)

    if "cifar100" in config["dataset"] and config["info"]=="Ours_v5":
        return cifar100_dataset_for_ours(config)
    elif "cifar100" in config["dataset"]:
        return cifar100_dataset(config)
    elif "cifar" in config["dataset"]:
        return cifar_dataset(config)
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test", "database"]:
        if config["info"] in ["Ours_v5"] and data_set == 'train_set':
            print("run Ours_v5")
            dsets["explore_set"] = ImageList(config["data_path"],
                                        open(data_config[data_set]["list_path"]).readlines(),
                                        transform=image_transform_v9(config["resize_size"], config["crop_size"], data_set))
            dsets[data_set] = ImageList_special(config["data_path"],
                                        open(data_config[data_set]["list_path"]).readlines(),
                                        transform=image_transform_v9(config["resize_size"], config["crop_size"], data_set), n_positive=config['n_positive'])
                    
        elif config["info"] in ["MBHash_v10_v3","Ours_v4"] and data_set == 'train_set':
            # 2023.12.11 特殊的采样形式
            print("run MBHash_v10_v3 or Ours_v4")
            dsets[data_set] = ImageList_special(config["data_path"],
                                        open(data_config[data_set]["list_path"]).readlines(),
                                        transform=image_transform_v9(config["resize_size"], config["crop_size"], data_set), n_positive=config['n_positive'])
        elif config["info"] in ["MBHash_v9", "MBHash_v10", "MBHash_v10_v2", "MBHash_v10_v3", "MBHash_v11", "MBHash_v12", "MBHash_v13", "MBHash_v14", "MBHash_v15", "MBHash_v16", "Ours", "Ours_v3", "Ours_v4", "Ours_v5"]:
            print("run MBHash_v9 or MBHash_v10 or MBHash_v10_v2 or MBHash_v10_v3 or MBHash_v11 or MBHash_v12 or MBHash_v13 or MBHash_v14 or MBHash_v15 or MBHash_v16 or Ours or Ours_v3 or Ours_v4 or Ours_v5")
            dsets[data_set] = ImageList(config["data_path"],
                                        open(data_config[data_set]["list_path"]).readlines(),
                                        transform=image_transform_v9(config["resize_size"], config["crop_size"], data_set))
        elif  config["info"] in ["Ours_v2"]:
            print("run Ours_v2")
            dsets[data_set] = ImageList(config["data_path"],
                                        open(data_config[data_set]["list_path"]).readlines(),
                                        transform=image_transform_ours(config["resize_size"], config["crop_size"], data_set))
        else:
            dsets[data_set] = ImageList(config["data_path"],
                                        open(data_config[data_set]["list_path"]).readlines(),
                                        transform=image_transform(config["resize_size"], config["crop_size"], data_set))

        print(data_set, len(dsets[data_set]))

        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle= (data_set == "train_set") , num_workers=4)
    if config["info"] in ["Ours_v5"]:
        dset_loaders["explore_set"] = util_data.DataLoader(dsets["explore_set"],
                                                      batch_size=64,
                                                      shuffle=True , num_workers=4)
        return dset_loaders["explore_set"], dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])
    else:
        return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
            len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])


# def compute_result(dataloader, net, device):
#     bs, clses = [], []
#     net.eval()
#     for img, cls, _ in tqdm(dataloader):
#         clses.append(cls)
#         # clses.append(cls.to(device).float())
#         out = net(img.to(device))
#         if isinstance(out, tuple):
#             bs.append(out[0].data.cpu())
#             # bs.append(out[0].data)
#         else:
#             bs.append((out).data.cpu())
#             # bs.append((out).data)
#         # bs.append((net(img.to(device))).data.cpu())
#     return torch.cat(bs).sign(), torch.cat(clses)

def compute_result(dataloader, net, config, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        # clses.append(cls)
        clses.append(cls.to(device).float())
        if config["info"] == "MDSH":
            out = net(img.to(device))[0]
        else:
            out = net(img.to(device))
        if isinstance(out, tuple):
            # bs.append(out[0].data.cpu())
            bs.append(out[0].data)
        else:
            # bs.append((out).data.cpu())
            bs.append((out).data)
        # bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        print(gnd, gnd.shape)
        print(ind, ind.shape)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

import torch
from tqdm import tqdm

# Make sure to use PyTorch operations inside the CalcHammingDist function
def CalcHammingDistGPU(B1, B2):
    q = B2.size(1)
    # print(B1.shape, B2.shape)
    distH = 0.5 * (q - torch.mm(B1, B2.t()))
    return distH

def CalcTopMapGPU(rB, qB, retrievalL, queryL, topk, device):
    num_query = queryL.size(0)
    topkmap = 0.0
    for iter in tqdm(range(num_query)):
        gnd = (torch.mm(queryL[iter, :].unsqueeze(0), retrievalL.t()) > 0).float().squeeze(0)
        hamm = CalcHammingDistGPU(qB[iter, :].unsqueeze(0), rB)
        ind = torch.argsort(hamm).squeeze(0)
        gnd = gnd[ind]
        tgnd = gnd[:topk]
        tsum = torch.sum(tgnd).int()
        if tsum == 0:
            continue
        count = torch.linspace(1, tsum.item(), tsum.item()).to(device)
        tindex = (tgnd == 1).nonzero(as_tuple=False).squeeze() + 1.0
        topkmap_ = torch.mean(count / tindex.float())
        topkmap += topkmap_.item()  # Convert to Python scalar

    topkmap = topkmap / num_query
    return topkmap

# faster but more memory
def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):
    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    recall = np.zeros((num_query, num_gallery))
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1, num_gallery + 1)

        prec[iter, :] = prec_sum / return_images
        recall[iter, :] = prec_sum / all_sim_num

        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()
    prec = prec[index]
    recall = recall[index]
    cum_prec = np.mean(prec, 0)
    cum_recall = np.mean(recall, 0)

    return topkmap, cum_prec, cum_recall

# https://github.com/chrisbyd/DeepHash-pytorch/blob/master/validate.py
def validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset):
    device = config["device"]
    tst_binary, tst_label = compute_result(test_loader, net, config, device=device)
    trn_binary, trn_label = compute_result(dataset_loader, net, config, device=device)

    # mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), config["topK"])
    mAP = CalcTopMapGPU(trn_binary, tst_binary, trn_label, tst_label, config["topK"], device)
    
    if mAP > Best_mAP:
        Best_mAP = mAP
        if "save_path" in config:
            save_path = os.path.join(config["save_path"], f'{config["dataset"]}_{bit}bits_{mAP}')
            os.makedirs(save_path, exist_ok=True)
            print("save in ", save_path)
            np.save(os.path.join(save_path, "tst_label.npy"), tst_label.numpy())
            np.save(os.path.join(save_path, "tst_binary.npy"), tst_binary.numpy())
            np.save(os.path.join(save_path, "trn_binary.npy"), trn_binary.numpy())
            np.save(os.path.join(save_path, "trn_label.npy"), trn_label.numpy())
            torch.save(net.state_dict(), os.path.join(save_path, "{}_model.pt".format(net.__class__.__name__)))
    # print(f"{config['info']} epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
    print(f"epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
    print(config)
    return mAP, Best_mAP

################## 2024.2.3 用于计算bit length ########################

def compute_result_RML(dataloader, net, device, bit_list, config):
    bss, clses = [], []
    for _ in bit_list:
        bss.append([])
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        # clses.append(cls)
        clses.append(cls.to(device).float())
        # outputs, outputs_norm = net(img.to(device))
        outputs = net(img.to(device))
        # for idx, output in enumerate(outputs_norm):
        for idx, output in enumerate(outputs):
            # bss[idx].append(output.data.cpu())
            if config["info"] == "MDSH":
                bss[idx].append(output[0].data)
            else:
                bss[idx].append(output.data)
        outputs = [torch.cat(bs).sign() for bs in bss]
    return outputs, torch.cat(clses)

def validate_RML(config, Best_mAP_list, test_loader, dataset_loader, net, epoch):
    # 2024.2.3
    device = config["device"]
    bit_list = config["bit_list"]
    # print(bit_list)
    tst_outputs, tst_label = compute_result_RML(test_loader, net, device, bit_list, config)
    trn_outputs, trn_label = compute_result_RML(dataset_loader, net, device, bit_list, config)
    # mAPs = [CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), config["topK"]) for trn_binary, tst_binary in zip(trn_outputs, tst_outputs)]
    mAPs = [CalcTopMapGPU(trn_binary, tst_binary, trn_label, tst_label, config["topK"], device) for trn_binary, tst_binary in zip(trn_outputs, tst_outputs)]
    print(mAPs)
    for idx, (mAP, Best_mAP) in enumerate(zip(mAPs, Best_mAP_list)):
        print(mAP)
        if mAP > Best_mAP:
            Best_mAP_list[idx] = mAP
    print(f"epoch:{epoch + 1} dataset:{config['dataset']} MAP:{str(mAPs)} Best MAP: {str(Best_mAP_list)}")
    print(config)
    return mAPs, Best_mAP_list

################## 2024.2.3 用于计算bit length ########################

# 2023.5.16 加入对优化器的调整

__optimizers = {
    'SGD': torch.optim.SGD,
    'ASGD': torch.optim.ASGD,
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'Adagrad': torch.optim.Adagrad,
    'Adadelta': torch.optim.Adadelta,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop
}

def adjust_optimizer(optimizer, epoch, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):
        if 'optimizer' in setting:
            optimizer = __optimizers[setting['optimizer']](
                optimizer.param_groups)
            logging.debug('OPTIMIZER - setting method = %s' %
                          setting['optimizer'])
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    logging.debug('OPTIMIZER - setting %s = %s' %
                                  (key, setting[key]))
                    param_group[key] = setting[key]
        return optimizer

    if callable(config):
        optimizer = modify_optimizer(optimizer, config(epoch))
    else:
        for e in range(epoch + 1):  # run over all epochs - sticky setting
            if e in config:
                optimizer = modify_optimizer(optimizer, config[e])

    return optimizer