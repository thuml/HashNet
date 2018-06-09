import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
import torch.utils.data as util_data
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable

optim_dict = {"SGD": optim.SGD}

def image_classification_predict(loader, model, test_10crop=True, gpu=True, softmax_param=1.0):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader['test'+str(i)]) for i in range(10)]
        for i in range(len(loader['test0'])):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            if gpu:
                for j in range(10):
                    inputs[j] = Variable(inputs[j].cuda())
                labels = Variable(labels.cuda())
            else:
                for j in range(10):
                    inputs[j] = Variable(inputs[j])
                labels = Variable(labels)
            outputs = []
            for j in range(9):
                _, predict_out = model(inputs[j])
                outputs.append(nn.Softmax()(softmax_param * predict_out))
            outputs_center = model(inputs[9])
            outputs.append(nn.Softmax()(softmax_param * outputs_center))
            softmax_outputs = sum(outputs)
            outputs = outputs_center
            if start_test:
                all_output = outputs.data.float()
                all_softmax_output = softmax_outputs.data.cpu().float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    else:
        iter_val = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_val.next()
            inputs = data[0]
            if gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)
            _, outputs = model(inputs)
            softmax_outputs = nn.Softmax()(softmax_param * outputs)
            if start_test:
                all_output = outputs.data.cpu().float()
                all_softmax_output = softmax_outputs.data.cpu().float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.cpu().float()), 0)
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    _, predict = torch.max(all_output, 1)
    return all_softmax_output, predict, all_output, all_label

def image_classification_test(loader, model, test_10crop=True, gpu=True):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader['test'+str(i)]) for i in range(10)]
        for i in range(len(loader['test0'])):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            if gpu:
                for j in range(10):
                    inputs[j] = Variable(inputs[j].cuda())
                labels = Variable(labels.cuda())
            else:
                for j in range(10):
                    inputs[j] = Variable(inputs[j])
                labels = Variable(labels)
            outputs = []
            for j in range(10):
                _, predict_out = model(inputs[j])
                outputs.append(nn.Softmax()(predict_out))
            outputs = sum(outputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    else:
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            if gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)       
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label) / float(all_label.size()[0])
    return accuracy


def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["train_set1"] = prep.image_train( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
    prep_dict["train_set2"] = prep.image_train( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
    if prep_config["test_10crop"]:
        prep_dict["database"] = prep.image_test_10crop( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
        prep_dict["test"] = prep.image_test_10crop( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
    else:
        prep_dict["database"] = prep.image_test( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
        prep_dict["test"] = prep.image_test( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
               
    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    dsets["train_set1"] = ImageList(open(data_config["train_set1"]["list_path"]).readlines(), \
                                transform=prep_dict["train_set1"])
    dset_loaders["train_set1"] = util_data.DataLoader(dsets["train_set1"], \
            batch_size=data_config["train_set1"]["batch_size"], \
            shuffle=True, num_workers=4)
    dsets["train_set2"] = ImageList(open(data_config["train_set2"]["list_path"]).readlines(), \
                                transform=prep_dict["train_set2"])
    dset_loaders["train_set2"] = util_data.DataLoader(dsets["train_set2"], \
            batch_size=data_config["train_set2"]["batch_size"], \
            shuffle=True, num_workers=4)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["database"+str(i)] = ImageList(open(data_config["database"]["list_path"]).readlines(), \
                                transform=prep_dict["database"]["val"+str(i)])
            dset_loaders["database"+str(i)] = util_data.DataLoader(dsets["database"+str(i)], \
                                batch_size=data_config["database"]["batch_size"], \
                                shuffle=False, num_workers=4)
            dsets["test"+str(i)] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"]["val"+str(i)])
            dset_loaders["test"+str(i)] = util_data.DataLoader(dsets["test"+str(i)], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=4)

    else:
        dsets["database"] = ImageList(open(data_config["database"]["list_path"]).readlines(), \
                                transform=prep_dict["database"])
        dset_loaders["database"] = util_data.DataLoader(dsets["database"], \
                                batch_size=data_config["database"]["batch_size"], \
                                shuffle=False, num_workers=4)
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = util_data.DataLoader(dsets["test"], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=4)

    hash_bit = config["hash_bit"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["type"](**net_config["params"])

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_network = base_network.cuda()

    ## collect parameters
    parameter_list = [{"params":base_network.feature_layers.parameters(), "lr":1}, \
                      {"params":base_network.hash_layer.parameters(), "lr":10}]
 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]


    ## train   
    len_train1 = len(dset_loaders["train_set1"]) - 1
    len_train2 = len(dset_loaders["train_set2"]) - 1
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    for i in range(config["num_iterations"]):
        if i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
                "iter_{:05d}_model.pth.tar".format(i)))

        ## train one iter
        base_network.train(True)
        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train1 == 0:
            iter1 = iter(dset_loaders["train_set1"])
        if i % len_train2 == 0:
            iter2 = iter(dset_loaders["train_set2"])
        inputs1, labels1 = iter1.next()
        inputs2, labels2 = iter2.next()
        if use_gpu:
            inputs1, inputs2, labels1, labels2 = \
                Variable(inputs1).cuda(), Variable(inputs2).cuda(), \
                Variable(labels1).cuda(), Variable(labels2).cuda()
        else:
            inputs1, inputs2, labels1, labels2 = Variable(inputs1), \
                Variable(inputs2), Variable(labels1), Variable(labels2)
           
        inputs = torch.cat((inputs1, inputs2), dim=0)
        outputs = base_network(inputs)
        similarity_loss = loss.pairwise_loss(outputs.narrow(0,0,inputs1.size(0)), \
                                 outputs.narrow(0,inputs1.size(0),inputs2.size(0)), \
                                 labels1, labels2, \
                                 sigmoid_param=config["loss"]["sigmoid_param"], \
                                 l_threshold=config["loss"]["l_threshold"], \
                                 class_num=config["loss"]["class_num"])

        total_loss = config["loss"]["l_weight"] * similarity_loss
        total_loss.backward()
        print("Iter: {:05d}, l_loss: {:.3f}".format(i, similarity_loss.float().data[0]))
        config["out_file"].write("Iter: {:05d}, l_loss: {:.3f}".format(i, \
                    similarity_loss.float().data[0]))
        optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('gpu_id', type=str, nargs='?', default='0', help="device id to run")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 

    # train config  
    config = {}
    config["num_iterations"] = 10000
    config["snapshot_interval"] = 3000
    config["dataset"] = "coco"
    config["hash_bit"] = 48
    config["output_path"] = "../snapshot/"+config["dataset"]+"_"+str(config["hash_bit"])+"bit_resnet_0.0003"
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")

    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
    config["network"] = {"type":network.ResNetFc, "params":{"name":"ResNet50", "hash_bit":config["hash_bit"], "use_hashnet":True} }
    config["prep"] = {"test_10crop":True, "resize_size":256, "crop_size":224}
    config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"step", \
                           "lr_param":{"init_lr":0.0003, "gamma":0.5, "step":2000} }
    config["loss"] = {"l_weight":1.0, "l_threshold":15.0, "sigmoid_param":10.0/config["hash_bit"]}

    if config["dataset"] == "imagenet":
        config["data"] = {"train_set1":{"list_path":"../data/imagenet/train.txt", "batch_size":36}, \
                          "train_set2":{"list_path":"../data/imagenet/train.txt", "batch_size":36}, \
                          "database":{"list_path":"../data/imagenet/database.txt", "batch_size":4}, \
                          "test":{"list_path":"../data/imagenet/test.txt", "batch_size":4}}
        config["loss"]["class_num"] = 100.0
    elif config["dataset"] == "nus_wide":
        config["data"] = {"train_set1":{"list_path":"../data/nus_wide/train.txt", "batch_size":36}, \
                          "train_set2":{"list_path":"../data/nus_wide/train.txt", "batch_size":36}, \
                          "database":{"list_path":"../data/nus_wide/database.txt", "batch_size":4}, \
                          "test":{"list_path":"../data/nus_wide/test.txt", "batch_size":4}}
        config["loss"]["class_num"] = 5.0
    elif config["dataset"] == "coco":
        config["data"] = {"train_set1":{"list_path":"../data/coco/train.txt", "batch_size":36}, \
                          "train_set2":{"list_path":"../data/coco/train.txt", "batch_size":36}, \
                          "database":{"list_path":"../data/coco/database.txt", "batch_size":4}, \
                          "test":{"list_path":"../data/coco/test.txt", "batch_size":4}}
        config["loss"]["class_num"] = 1.0
    train(config)
