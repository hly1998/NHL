# 2024.2.3 针对设计loss形式的模型，可采用一套统一的训练方案

from utils.tools import *
import torch
import time
import logging
from bit_length.CSQ import CSQLoss, csq_config
from bit_length.DCH import DCHLoss, dch_config
from bit_length.DTSH import DTSHLoss, dtsh_config
from bit_length.CNNH import CNNHLoss, cnnh_config
from bit_length.DBDH import DBDHLoss, dbdh_config
from bit_length.DFH import DFHLoss, dfh_config
from bit_length.DHN import DHNLoss, dhn_config
from bit_length.DPN import DPNLoss, dpn_config
from bit_length.DHD import DHDLoss, dhd_config, Augmentation
from bit_length.MDSH import MDSHLoss
from bit_length.distill import Ours_Distill
import torch.nn.functional as F
import kornia.augmentation as Kg
import wandb



grads = {}

def save_grad(name):
    def hook(grad):
        bit =  grad.shape[-1]
        grads[bit] = grad
    return hook

def grand_train_val(config, bit, net):
    # 特殊参数
    if config["info"] == "DTSH":
        config = dtsh_config(config)
    elif config["info"] == "DCH":
        config = dch_config(config)
    elif config["info"] == "CSQ":
        config = csq_config(config)
    elif config["info"] == "CNNH":
        config = cnnh_config(config)
    elif config["info"] == "DBDH":
        config = dbdh_config(config)
    elif config["info"] == "DFH":
        config = dfh_config(config)
    elif config["info"] == "DHN":
        config = dhn_config(config)
    elif config["info"] == "DPN":
        config = dpn_config(config)
    # elif config["info"] == "MDSH":
    #     config = mdsh_config(config)
    elif config["info"] == "DHD":
        config = dhd_config(config)
    
    if config["distill"]:
        distill_criterion = Ours_Distill()

    device = config["device"]
    # analysis = config["analysis"]
    heuristic_weight_value = torch.tensor(config['heuristic_weight_value']).to(device)
    analysis_weight = torch.ones(len(config["bit_list"])).to(device)
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    Best_mAP = 0
    Best_mAP_list = np.zeros(len(config["bit_list"]))
    if config["mode"] == "simple":
        if config["info"] == "DTSH":
            criterion = DTSHLoss(config, bit)
        elif config["info"] == "DCH":
            criterion = DCHLoss(config, bit)
        elif config["info"] == "CSQ":
            criterion = CSQLoss(config, bit)
        elif config["info"] == "CNNH":
            clses = []
            for _, cls, _ in tqdm(train_loader):
                clses.append(cls)
            train_labels = torch.cat(clses).to(device).float()
            criterion = CNNHLoss(config, train_labels, bit)
        elif config["info"] == "DBDH":
            criterion = DBDHLoss(config, bit)
        elif config["info"] == "DFH":
            criterion = DFHLoss(config, bit)
        elif config["info"] == "DHN":
            criterion = DHNLoss(config, bit)
        elif config["info"] == "DPN":
            criterion = DPNLoss(config, bit)
        elif config["info"] == "DHD":
            AugS = Augmentation(config["resize_size"], 1.0)
            AugT = Augmentation(config["resize_size"], config["transformation_scale"])
            Crop = torch.nn.Sequential(Kg.CenterCrop(config["crop_size"]))
            Norm = torch.nn.Sequential(Kg.Normalize(mean=torch.as_tensor([0.485, 0.456, 0.406]), std=torch.as_tensor([0.229, 0.224, 0.225])))
            criterion = DHDLoss(config, bit)
        elif config["info"] == "MDSH":
            criterion = MDSHLoss(config, bit)
        
        for epoch in range(config["epoch"]):
            current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
            print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
                config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

            net.train()

            train_loss = 0
            test_loss = 0
            for image, label, ind in train_loader:
                image = image.to(device)
                label = label.to(device)
                # print("image", image.shape)
                optimizer.zero_grad()
                if config["info"] == 'DHD':
                    Is = Norm(Crop(AugS(image)))
                    It = Norm(Crop(AugT(image)))
                    # print("I shape:", Is.shape, It.shape)
                    Xt = net(It)
                    Xs = net(Is)
                    u = (Xs, Xt)
                    # print("u shape:", u[0].shape, u[1].shape)
                else:
                    u = net(image)
                if config["info"] == 'MDSH':
                    loss = criterion(u, label.float(), ind, epoch)
                else:
                    loss = criterion(u, label.float(), ind, config)
                # print("1111")
                train_loss += loss.item()

                loss.backward()
                optimizer.step()

            train_loss = train_loss / len(train_loader)

            # 增加一项，对测试集计算loss
            with torch.no_grad():
                net.eval()
                for image, label, ind in test_loader:
                    image = image.to(device)
                    label = label.to(device)
                    u = net(image)
                    if config["info"] == 'MDSH':
                        loss = criterion(u, label.float(), ind, epoch)
                    else:
                        loss = criterion(u, label.float(), ind, config)
                    test_loss += loss.item()
                test_loss = test_loss / len(test_loader)

            print("\b\b\b\b\b\b\b train loss:%.3f, test loss:%.3f" % (train_loss, test_loss))
            if (epoch + 1) % config["test_map"] == 0:
                Best_mAP_before = Best_mAP
                mAP, Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
                logging.info(f"{net.__class__.__name__} epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
                if config["wandb"]:
                    wandb.log({"mAP": mAP, "best mAP": Best_mAP, "epoch": epoch+1, "loss": train_loss})
                if mAP > Best_mAP_before:
                    count = 0
                else:
                    if count == config['stop_iter']:
                        break
                    count += 1
            else:
                if config["wandb"]:
                    wandb.log({"epoch": epoch+1, "train_loss": train_loss, "test_loss": test_loss}, )

    elif config["mode"] in ['RML', 'RML_E']:
        if config["info"] == "DTSH":
            criterions = [DTSHLoss(config, bit) for bit in config["bit_list"]]
        elif config["info"] == "DCH":
            criterions = [DCHLoss(config, bit) for bit in config["bit_list"]]
        elif config["info"] == "CSQ":
            criterions = [CSQLoss(config, bit) for bit in config["bit_list"]]
        elif config["info"] == "CNNH":
            clses = []
            for _, cls, _ in tqdm(train_loader):
                clses.append(cls)
            train_labels = torch.cat(clses).to(device).float()
            criterions = [CNNHLoss(config, train_labels, bit) for bit in config["bit_list"]]
        elif config["info"] == "DBDH":
            criterions = [DBDHLoss(config, bit) for bit in config["bit_list"]]
        elif config["info"] == "DFH":
            criterions = [DFHLoss(config, bit) for bit in config["bit_list"]]
        elif config["info"] == "DHN":
            criterions = [DHNLoss(config, bit) for bit in config["bit_list"]]
        elif config["info"] == "DPN":
            criterions = [DPNLoss(config, bit) for bit in config["bit_list"]]
        elif config["info"] == "DHD":
            AugS = Augmentation(config["resize_size"], 1.0)
            AugT = Augmentation(config["resize_size"], config["transformation_scale"])
            Crop = torch.nn.Sequential(Kg.CenterCrop(config["crop_size"]))
            Norm = torch.nn.Sequential(Kg.Normalize(mean=torch.as_tensor([0.485, 0.456, 0.406]), std=torch.as_tensor([0.229, 0.224, 0.225])))
            criterions = [DHDLoss(config, bit) for bit in config["bit_list"]]
        elif config["info"] == "MDSH":
            criterions = [MDSHLoss(config, bit) for bit in config["bit_list"]]

        step = 0
        for epoch in range(config["epoch"]):
            current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
            print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
                config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

            net.train()

            train_loss = np.zeros(len(criterions))
            test_loss = np.zeros(len(criterions))
            distill_loss_record = 0
            preserve_loss_record = 0
            # 初始化权重，仅会在analysis模式下起到作用
            ws = [1.0, 1.0, 1.0, 1.0, 1.0]
            for image, label, ind in train_loader:
                step += 1
                image = image.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                outputs = net(image)
                if config["info"] == 'DHD':
                    Is = Norm(Crop(AugS(image)))
                    It = Norm(Crop(AugT(image)))
                    Xts = net(It)
                    Xss = net(Is)
                    outputs = [(Xs, Xt) for Xs, Xt in zip(Xts, Xss)]
                else:
                    outputs = net(image)
                if config['analysis']:
                    # 在这里完成梯度相关的计算，获取权重
                    w_grad = []
                    for idx, (criterion, output_norm) in enumerate(zip(criterions, outputs)):
                        # obj = criterion(output_norm, label.float(), ind, config)
                        if config["info"] == 'MDSH':
                            obj = criterion(output_norm, label.float(), ind, epoch)
                        else:
                            obj = criterion(output_norm, label.float(), ind, config)
                        train_loss[idx] += obj.item()
                        optimizer.zero_grad(set_to_none=True)
                        # if idx < len(criterions):
                        #     obj.backward(retain_graph=True)
                        # else:
                        #     obj.backward()
                        obj.backward(retain_graph=True)
                        for group in optimizer.param_groups:
                            w_grad.append(group['params'][-2].grad.flatten())
                    # 梯度的余弦值
                    grad_simis = [[] for _ in config["bit_list"][:-1]]
                    # 梯度的点积
                    grad_dots = [[] for _ in config["bit_list"][:-1]]
                    # 梯度对应维度的长度
                    grad_norm = [[] for _ in config["bit_list"][:-1]]
                    for idx, shotest_bit in enumerate(config['bit_list']):
                        if idx == len(config['bit_list'])-1:
                            break
                        wgrads_short = w_grad[idx]
                        for wgh in w_grad[idx+1:]:
                            grad_simis[idx].append(F.cosine_similarity(wgrads_short[:2048*shotest_bit].unsqueeze(0), wgh[:2048*shotest_bit].unsqueeze(0)).item())
                            grad_dots[idx].append(torch.dot(wgrads_short[:2048*shotest_bit], wgh[:2048*shotest_bit]).item())
                        for wgh in w_grad:
                            grad_norm[idx].append(wgh[:2048*shotest_bit].norm().item())
                    # 计算新权重的值
                    # alpha = [1.0, 1.0, 1.0, 1.0, 1.0]
                    alpha = [[1.0,] for _ in config["bit_list"][:-1]]
                    beta = 1.0
                    for idx, (grad_simi, grad_dot, grad_no) in enumerate(zip(grad_simis, grad_dots, grad_norm)):
                        short_norm = grad_no[0]
                        for grad_s, grad_d, grad_n in zip(grad_simi, grad_dot, grad_no[idx+1:]):
                            if grad_d < 0:
                                if (- grad_d) > (beta * short_norm * short_norm / len(config["bit_list"])):
                                    al = (beta * short_norm * short_norm / len(config["bit_list"])) / (- grad_d) 
                                    alpha[idx].append(al)
                                else:
                                    alpha[idx].append(1.0)
                            else:
                                alpha[idx].append(1.0)
                    # 构建ws
                    # 先处理 alpha
                    for i, alp in enumerate(alpha):
                        anchor = alp[1]
                        for j, al in enumerate(alp[2:]):
                            if al > anchor:
                                alpha[i][j+2] = anchor
                    # 获取ws
                    ws = [1.0]
                    for idx in range(len(config["bit_list"][:-1])):
                        w = 1.0
                        for jdx in range(idx+1):
                            # print(jdx, idx+1-jdx)
                            w = w * alpha[jdx][idx+1-jdx]
                        ws.append(w)
                    # print("------ ws:", ws)
                    # 一种更高效的方式
                    # ws = [1.0]
                    # for idx, _, in enumerate(config["bit_list"][1:]):
                    #     ws.append(alpha[idx][1]*ws[idx])
                    # print(ws)
                        
                    total_sum = sum(ws)
                    ws = [x*len(config["bit_list"]) / total_sum for x in ws]
                    losses = [w*criterion(output, label.float(), ind, config) for w, criterion, output in zip(ws, criterions, outputs)]
                    for idx, loss in enumerate(losses):
                        train_loss[idx] += loss.item()
                    obj = torch.stack(losses).sum()
                    if config["distill"]:
                        # 如果是DHD或者MDSH，需要处理一下outputs
                        if config["info"] in ('DHD', 'MDSH'):
                            outputs = [o1 for (o1, _) in outputs]
                        distill_loss, preserve_loss = distill_criterion(outputs)
                        distill_loss_record = distill_loss_record + distill_loss.item()
                        preserve_loss_record = preserve_loss_record + preserve_loss.item()
                        obj = obj + config["distill_weight"]*(distill_loss + preserve_loss)
                    obj.backward()
                    optimizer.step()
                else:
                    losses = [criterion(output, label.float(), ind, config) for criterion, output in zip(criterions, outputs)]
                    for idx, loss in enumerate(losses):
                        train_loss[idx] += loss.item()
                    obj = torch.stack(losses).sum()
                    # 是否考虑蒸馏
                    if config["distill"]:
                        # 如果是DHD或者MDSH，需要处理一下outputs
                        if config["info"] in ('DHD', 'MDSH'):
                            outputs = [o1 for (o1, _) in outputs]
                        distill_loss, preserve_loss = distill_criterion(outputs)
                        distill_loss_record = distill_loss_record + distill_loss.item()
                        preserve_loss_record = preserve_loss_record + preserve_loss.item()
                        obj = obj + config["distill_weight"]*(distill_loss + preserve_loss)
                        # obj = obj + config["distill_weight"]*distill_loss
                        # obj = obj + config["distill_weight"]*preserve_loss
                    obj.backward()
                    optimizer.step()
                    
            train_loss = train_loss / len(train_loader)
            if config["distill"]:
                distill_loss_record = distill_loss_record / len(train_loader)
                preserve_loss_record = preserve_loss_record / len(train_loader)
            with torch.no_grad():
                net.eval()
                for image, label, ind in test_loader:
                    step += 1
                    image = image.to(device)
                    label = label.to(device)
                    outputs = net(image)
                    for idx, (criterion, output_norm) in enumerate(zip(criterions, outputs)):
                        # obj = criterion(output_norm, label.float(), ind, config)
                        if config["info"] == 'MDSH':
                            obj = criterion(output_norm, label.float(), ind, epoch)
                        else:
                            obj = criterion(output_norm, label.float(), ind, config)
                        test_loss[idx] += obj.item()
            test_loss = test_loss / len(test_loader)

            if config["distill"]:
                print("\b\b\b\b\b\b\b train loss:%.3f, test loss:%.3f distill loss:%.5f preserve loss:%.5f" % (train_loss.sum(), test_loss.sum(), distill_loss_record, preserve_loss_record))
            else:
                print("\b\b\b\b\b\b\b train loss:%.3f, test loss:%.3f" % (train_loss.sum(), test_loss.sum()))

            if (epoch + 1) % config["test_map"] == 0:
                Best_mAP_list_before = Best_mAP_list.copy()
                mAP_list, Best_mAP_list = validate_RML(config, Best_mAP_list, test_loader, dataset_loader, net, epoch)
                logging.info(f"{net.__class__.__name__} epoch:{epoch + 1} dataset:{config['dataset']} MAP:{str(mAP_list)} Best MAP: {str(Best_mAP_list)}")
                if config["wandb"]:
                    log_dic = {}
                    for idx, bit in enumerate(config["bit_list"]):
                        log_dic["mAP_b{}".format(bit)] = mAP_list[idx]
                        log_dic["best mAP_b{}".format(bit)] = Best_mAP_list[idx]
                        log_dic["train_loss_b{}".format(bit)] = train_loss[idx]
                        log_dic["test_loss_b{}".format(bit)] = test_loss[idx]
                    log_dic["total train_loss"] = train_loss.sum()
                    log_dic["total test_loss"] = train_loss.sum()
                    log_dic["epoch"] = epoch+1
                    wandb.log(log_dic)
                    # wandb.log({"mAP0": mAP_list[0], "best mAP0": Best_mAP_list[0], "train_loss0": train_loss[0],
                    #         "mAP1": mAP_list[1], "best mAP1": Best_mAP_list[1], "train_loss1": train_loss[1],
                    #         "mAP2": mAP_list[2], "best mAP2": Best_mAP_list[2], "train_loss2": train_loss[2],
                    #         "loss": train_loss, "epoch": epoch+1}, )
                for idx, bit in enumerate(config["bit_list"]):
                    if mAP_list[idx] > Best_mAP_list_before[idx]:
                        print("better results get")
                        count = 0
                count += 1
                if count == config['stop_iter']:
                    break
            else:
                if config["wandb"]:
                    # wandb.log({"train_loss0": train_loss[0], "train_loss1": train_loss[1], "train_loss2": train_loss[2],
                    #         "loss": train_loss, "epoch": epoch+1}, )
                    log_dic = {}
                    for idx, bit in enumerate(config["bit_list"]):
                        log_dic["train_loss_b{}".format(bit)] = train_loss[idx]
                        log_dic["test_loss_b{}".format(bit)] = test_loss[idx]
                    log_dic["total train_loss"] = train_loss.sum()
                    log_dic["total test_loss"] = train_loss.sum()
                    log_dic["epoch"] = epoch+1
                    wandb.log(log_dic)
        # if config['analysis']:
        #     for hook in hooks:
        #         hook.remove()