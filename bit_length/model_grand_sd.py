from utils.tools import *
import torch
import time
import logging
from bit_length.CSQ import CSQLoss, csq_config
from bit_length.DCH import DCHLoss, dch_config
from bit_length.DTSH import DTSHLoss, dtsh_config
from bit_length.DBDH import DBDHLoss, dbdh_config
from bit_length.DHN import DHNLoss, dhn_config
from bit_length.DPN import DPNLoss, dpn_config
from bit_length.MDSH import MDSHLoss
from bit_length.DSH import DSHLoss, dsh_config
from bit_length.LCDSH import LCDSHLoss, lcdsh_config
from bit_length.SHCIR import SHCIRLoss, shcir_config
from bit_length.distill import Ours_Distill, Ours_Distill_per_one, Ours_Distill_all
import torch.nn.functional as F
import wandb
import time


grads = {}

def save_grad(name):
    def hook(grad):
        bit =  grad.shape[-1]
        grads[bit] = grad
    return hook

def grand_train_val(config, bit, net, head=None):
    # 特殊参数
    if config["info"] == "DTSH":
        config = dtsh_config(config)
    elif config["info"] == "DCH":
        config = dch_config(config)
    elif config["info"] == "CSQ":
        config = csq_config(config)
    elif config["info"] == "DBDH":
        config = dbdh_config(config)
    elif config["info"] == "DHN":
        config = dhn_config(config)
    elif config["info"] == "DPN":
        config = dpn_config(config)
    elif config["info"] == "DSH":
        config = dsh_config(config)
    elif config["info"] == "LCDSH":
        config = lcdsh_config(config)
    elif config["info"] == "SHCIR":
        config = shcir_config(config)
    if config["distill"]:
        distill_criterion_all = Ours_Distill_all(bit_list=config["bit_list"])
        distill_criterion = Ours_Distill(bit_list=config["bit_list"])
        distill_per_one_criterion = Ours_Distill_per_one()
    device = config["device"]
    # analysis = config["analysis"]
    heuristic_weight_value = torch.tensor(config['heuristic_weight_value']).to(device)
    analysis_weight = torch.ones(len(config["bit_list"])).to(device)
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    if config['mode'] == 'simple':
        optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    else:
        optimizer = config["optimizer"]["type"]([*net.parameters(), *head.parameters()], **(config["optimizer"]["optim_params"]))
    Best_mAP = 0
    Best_mAP_list = np.zeros(len(config["bit_list"]))
    total_time = 0
    if config["mode"] == "simple":
        if config["info"] == "DTSH":
            criterion = DTSHLoss(config, bit)
        elif config["info"] == "DCH":
            criterion = DCHLoss(config, bit)
        elif config["info"] == "CSQ":
            criterion = CSQLoss(config, bit)
        elif config["info"] == "DBDH":
            criterion = DBDHLoss(config, bit)
        elif config["info"] == "DFH":
            criterion = DFHLoss(config, bit)
        elif config["info"] == "DHN":
            criterion = DHNLoss(config, bit)
        elif config["info"] == "DPN":
            criterion = DPNLoss(config, bit)
        elif config["info"] == "MDSH":
            criterion = MDSHLoss(config, bit)
        elif config["info"] == "DSH":
            criterion = DSHLoss(config, bit)
        elif config["info"] == "LCDSH":
            criterion = LCDSHLoss(config, bit)
        elif config["info"] == "SHCIR":
            criterion = SHCIRLoss(config, bit)
            
        for epoch in range(config["epoch"]):
            current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
            print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
                config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

            net.train()
            start_time = time.time()

            train_loss = 0
            test_loss = 0
            for image, label, ind in train_loader:
                if config["info"] == 'CIBHash':
                    image[0] = image[0].to(device)
                    image[1] = image[1].to(device)
                else:
                    image = image.to(device)
                label = label.to(device)
                # print("image", image.shape)
                optimizer.zero_grad()
                u = net(image)
                if config["info"] == 'MDSH':
                    loss = criterion(u, label.float(), ind, epoch)
                else:
                    loss = criterion(u, label.float(), ind, config)
                train_loss += loss.item()

                loss.backward()
                optimizer.step()
                # for name, param in net.named_parameters():
                #     # print(param, name)
                #     if 'hash_layer' in name:
                #         print(param[0], name)
            
            total_time += time.time() - start_time

            train_loss = train_loss / len(train_loader)
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
                    wandb.log({"mAP": mAP, "best mAP": Best_mAP, "epoch": epoch+1, "loss": train_loss, "total_time": total_time})
                if mAP > Best_mAP_before:
                    count = 0
                else:
                    if count == config['stop_iter']:
                        break
                    count += 1
            else:
                if config["wandb"]:
                    wandb.log({"epoch": epoch+1, "train_loss": train_loss, "test_loss": test_loss, "total_time": total_time}, )

    elif config["mode"] in ['RML_E']:
        if config["info"] == "DTSH":
            criterions = [DTSHLoss(config, bit) for bit in config["bit_list"]]
        elif config["info"] == "DCH":
            criterions = [DCHLoss(config, bit) for bit in config["bit_list"]]
        elif config["info"] == "CSQ":
            criterions = [CSQLoss(config, bit) for bit in config["bit_list"]]
            criterions = [DBDHLoss(config, bit) for bit in config["bit_list"]]
        elif config["info"] == "DHN":
            criterions = [DHNLoss(config, bit) for bit in config["bit_list"]]
        elif config["info"] == "DPN":
            criterions = [DPNLoss(config, bit) for bit in config["bit_list"]]
        elif config["info"] == "MDSH":
            criterions = [MDSHLoss(config, bit) for bit in config["bit_list"]]
        elif config["info"] == "DSH":
            criterions = [DSHLoss(config, bit) for bit in config["bit_list"]]
        elif config["info"] == "LCDSH":
            criterions = [LCDSHLoss(config, bit) for bit in config["bit_list"]]
        elif config["info"] == "SHCIR":
            criterions = [SHCIRLoss(config, bit) for bit in config["bit_list"]]
        step = 0
        for epoch in range(config["epoch"]):
            current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
            print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
                config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

            net.train()
            head.train()
            start_time = time.time()

            train_loss = np.zeros(len(criterions))
            test_loss = np.zeros(len(criterions))
            distill_loss_record = 0
            preserve_loss_record = 0
            ws = [1.0, 1.0, 1.0, 1.0, 1.0]
            anti_rate = 0
            for image, label, ind in train_loader:
                step += 1
                image = image.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                feature = net(image)
                
                if config['analysis']:
                    if config["info"] == 'MDSH':
                        outputs = head([f.detach() for f in feature])
                    else:
                        outputs = head(feature.detach())
                    w_grad = []
                    for idx, (criterion, output_norm) in enumerate(zip(criterions, outputs)):
                        if config["info"] == 'MDSH':
                            obj = criterion(output_norm, label.float(), ind, epoch)
                        else:
                            obj = criterion(output_norm, label.float(), ind, config)
                        if config["distill"]:
                            if idx < len(config["bit_list"])-1:
                                if config['info']=='MDSH':
                                    obj = obj + config["distill_weight"] * distill_per_one_criterion((output_norm[0], outputs[idx+1][0]))
                                else:
                                    obj = obj + config["distill_weight"] * distill_per_one_criterion((output_norm, outputs[idx+1]))
                        train_loss[idx] += obj.item()
                        optimizer.zero_grad(set_to_none=True)
                        obj.backward(retain_graph=True)
                        for group in optimizer.param_groups:
                            if config["info"] == 'MDSH':
                                w_grad.append(group['params'][-4].grad.flatten())
                            else:
                                w_grad.append(group['params'][-2].grad.flatten())
                    grad_simis = [[] for _ in config["bit_list"][:-1]]
                    grad_dots = [[] for _ in config["bit_list"][:-1]]
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

                    alpha = [[1.0,] for _ in config["bit_list"][:-1]]
                    beta = 1.0
                    for idx, (grad_simi, grad_dot, grad_no) in enumerate(zip(grad_simis, grad_dots, grad_norm)):
                        short_norm = grad_no[0]
                        for grad_s, grad_d, grad_n in zip(grad_simi, grad_dot, grad_no[idx+1:]):
                            if grad_d < 0:
                                if (- grad_d) > (beta * short_norm * short_norm / (len(config["bit_list"])-1)):
                                    al = (beta * short_norm * short_norm / (len(config["bit_list"])-1)) / (- grad_d) 
                                    alpha[idx].append(al)
                                else:
                                    alpha[idx].append(1.0)
                            else:
                                alpha[idx].append(1.0)

                    for i, alp in enumerate(alpha):
                        anchor = alp[1]
                        for j, al in enumerate(alp[2:]):
                            if al > anchor:
                                alpha[i][j+2] = anchor

                    ws = [1.0]
                    for idx in range(len(config["bit_list"][:-1])):
                        w = 1.0
                        for jdx in range(idx+1):
                            w = w * alpha[jdx][idx+1-jdx]
                        ws.append(w)
                    if ws[1] != 1:
                        anti_rate += 1
                        
                    total_sum = sum(ws)
                    ws = [x*len(config["bit_list"]) / total_sum for x in ws]
                    outputs = head(feature)
                    if config["distill"]:
                        if config['info'] == 'MDSH':
                            outputs_d = [o1 for (o1, _) in outputs]
                        distill_losses = distill_criterion(outputs_d)
                        distill_loss_record = distill_loss_record + torch.stack(distill_losses).sum().item()
                        losses = [w*(criterion(output, label.float(), ind, config)+config["distill_weight"]*disitll_loss) for w, criterion, output, disitll_loss  in zip(ws, criterions, outputs, distill_losses)]
                    else:
                        losses = [w*(criterion(output, label.float(), ind, config)) for w, criterion, output  in zip(ws, criterions, outputs)]
                    for idx, loss in enumerate(losses):
                        train_loss[idx] += loss.item()
                    obj = torch.stack(losses).sum()
                    obj.backward()
                    optimizer.step()
                else:
                    outputs = head(feature)
                    if config["info"] == 'MDSH':
                        losses = [criterion(output, label.float(), ind, epoch) for criterion, output in zip(criterions, outputs)]
                    else:
                        losses = [criterion(output, label.float(), ind, config) for criterion, output in zip(criterions, outputs)]
                    for idx, loss in enumerate(losses):
                        train_loss[idx] += loss.item()
                    obj = torch.stack(losses).sum()
                    if config["distill"]:
                        if config["info"]=='MDSH':
                            outputs = [o1 for (o1, _) in outputs]
                        distill_loss, preserve_loss = distill_criterion_all(outputs)
                        distill_loss_record = distill_loss_record + distill_loss.item()
                        preserve_loss_record = preserve_loss_record + preserve_loss.item()
                        # obj = obj + config["distill_weight"]*(distill_loss + preserve_loss)
                        obj = obj + config["distill_weight"] * distill_loss
                        # obj = obj + config["distill_weight"]*preserve_loss
                    obj.backward()
                    optimizer.step()

            total_time += time.time() - start_time
            train_loss = train_loss / len(train_loader)
            anti_rate = anti_rate  / len(train_loader)
            print("anti_rate", anti_rate)
            if config["distill"]:
                distill_loss_record = distill_loss_record / len(train_loader)
                preserve_loss_record = preserve_loss_record / len(train_loader)

            with torch.no_grad():
                net.eval()
                head.eval()
                for image, label, ind in test_loader:
                    step += 1
                    image = image.to(device)
                    label = label.to(device)
                    if config["info"] in ["Bihalf", "GreedyHash", "CIBHash"]:
                        outputs = net(image, True)
                    else:
                        feature = net(image)
                        outputs = head(feature)
                    for idx, (criterion, output_norm) in enumerate(zip(criterions, outputs)):
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
                mAP_list, Best_mAP_list = validate_RML(config, Best_mAP_list, test_loader, dataset_loader, (net, head), epoch)
                logging.info(f"{net.__class__.__name__} epoch:{epoch + 1} dataset:{config['dataset']} distill_weight:{config['distill_weight']} MAP:{str(mAP_list)} Best MAP: {str(Best_mAP_list)} total_time:{str(total_time)}")
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
                    log_dic["total_time"] = total_time
                    wandb.log(log_dic)
                for idx, bit in enumerate(config["bit_list"]):
                    if mAP_list[idx] > Best_mAP_list_before[idx]:
                        print("better results get")
                        count = 0
                count += 1
                if count == config['stop_iter']:
                    break
            else:
                if config["wandb"]:
                    log_dic = {}
                    for idx, bit in enumerate(config["bit_list"]):
                        log_dic["train_loss_b{}".format(bit)] = train_loss[idx]
                        log_dic["test_loss_b{}".format(bit)] = test_loss[idx]
                    log_dic["total train_loss"] = train_loss.sum()
                    log_dic["total test_loss"] = train_loss.sum()
                    log_dic["epoch"] = epoch+1
                    log_dic["total_time"] = total_time
                    wandb.log(log_dic)