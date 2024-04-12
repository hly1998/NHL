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
from tqdm import tqdm


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
    cifar_dataset_root = '{your root}/data/cifar10/'
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

def config_dataset(config):
    if "cifar" in config["dataset"]:
        config["topK"] = -1
        config["n_class"] = 10
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
    config["data_path"] = "/dataset/" + config["dataset"] + "/"
    if config["dataset"] == "imagenet":
        config["data_path"] = "{your root}/data/imagenet/"
    if config["dataset"] == "imagenet-2":
        config["data_path"] = "{your root}/data/imagenet/"
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "{your root}/data/nuswide_21/"
    if config["dataset"] == "coco":
        config["data_path"] = "{your root}/data/coco/"
    config["data"] = {
        "train_set": {"list_path": config["data_path"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": config["data_path"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": config["data_path"] + "/test.txt", "batch_size": config["batch_size"]}}
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

class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index

def get_data(config):
    if "cifar" in config["dataset"]:
        return cifar_dataset(config)
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))

        print(data_set, len(dsets[data_set]))

        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle= (data_set == "train_set") , num_workers=4)
    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
            len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])

def compute_result(dataloader, net, config, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls.to(device).float())
        if config["info"] == "MDSH":
            out = net(img.to(device))[0]
        elif config["info"] == "CIBHash":
            img[0] = img[0].to(device)
            img[1] = img[1].to(device)
            out = net(img)
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

def validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset):
    device = config["device"]
    tst_binary, tst_label = compute_result(test_loader, net, config, device=device)
    trn_binary, trn_label = compute_result(dataset_loader, net, config, device=device)

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

def compute_result_RML(dataloader, net, device, bit_list, config):
    bss, clses = [], []
    tuple_flag = False
    for _ in bit_list:
        bss.append([])
    if isinstance(net, tuple):
        (net, head) = net
        net.eval()
        head.eval()
        tuple_flag = True
    else:
        net.eval()
    for img, cls, _ in tqdm(dataloader):
        # clses.append(cls)
        clses.append(cls.to(device).float())
        # outputs, outputs_norm = net(img.to(device))
        if tuple_flag:
            feature = net(img.to(device))
            outputs = head(feature)
        else:
            outputs = net(img.to(device))
        # for idx, output in enumerate(outputs_norm):
        for idx, output in enumerate(outputs):
            # bss[idx].append(output.data.cpu())
            if config["info"] == "MDSH":
                bss[idx].append(output[0].data)
            else:
                bss[idx].append(output.data)
        # outputs = [torch.cat(bs).sign() for bs in bss]
    outputs = [torch.cat(bs).sign() for bs in bss]
    return outputs, torch.cat(clses)

def validate_RML(config, Best_mAP_list, test_loader, dataset_loader, net, epoch):
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