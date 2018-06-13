import scipy as sp
import sys
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

def save_code_and_label(params, path):
    database_code = params['database_code']
    validation_code = params['test_code']
    database_labels = params['database_labels']
    validation_labels = params['test_labels']
    np.save(path + "_database_code.npy", database_code)
    np.save(path + "_database_labels.npy", database_labels)
    np.save(path + "_test_code.npy", validation_code)
    np.save(path + "_test_labels.npy", validation_labels)


def mean_average_precision(params, R):
    database_code = params['database_code']
    validation_code = params['test_code']
    database_labels = params['database_labels']
    validation_labels = params['test_labels']
    query_num = validation_code.shape[0]

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []
    
    for i in range(query_num):
        label = validation_labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R+1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
    
    return np.mean(np.array(APx))
        
def code_predict(loader, model, name, test_10crop=True, gpu=True):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader[name+str(i)]) for i in range(10)]
        for i in range(len(loader[name+'0'])):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            if gpu:
                for j in range(10):
                    inputs[j] = Variable(inputs[j].cuda())
            else:
                for j in range(10):
                    inputs[j] = Variable(inputs[j])
            outputs = []
            for j in range(10):
                outputs.append(model(inputs[j]))
            outputs = sum(outputs) / 10.0
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    else:
        iter_val = iter(loader[name])
        for i in range(len(loader[name])):
            data = iter_val.next()
            inputs = data[0]
            labels = data[1]
            if gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)
            outputs = model(inputs)
            if start_test:
                all_output = outputs.data.cpu().float()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.cpu().float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    return torch.sign(all_output), all_label

def predict(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
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
    ## set base network
    base_network = torch.load(config["snapshot_path"])

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_network = base_network.cuda()

    database_codes, database_labels = code_predict(dset_loaders, base_network, "database", test_10crop=prep_config["test_10crop"], gpu=use_gpu)
    test_codes, test_labels = code_predict(dset_loaders, base_network, "test", test_10crop=prep_config["test_10crop"], gpu=use_gpu)

    return {"database_code":database_codes.numpy(), "database_labels":database_labels.numpy(), \
            "test_code":test_codes.numpy(), "test_labels":test_labels.numpy()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, default='coco', help="dataset name")
    parser.add_argument('--hash_bit', type=int, default=48, help="number of hash code bits")
    parser.add_argument('--prefix', type=str, help="save path prefix")
    parser.add_argument('--snapshot', type=str, help="model path prefix")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 

    # train config  
    config = {}
    config["dataset"] = args.dataset 
    config["snapshot_path"] = "../snapshot/"+config["dataset"]+"_"+str(args.hash_bit)+"bit_"+args.prefix+"/"+args.snapshot+"_model.pth.tar"
    config["output_path"] = "../snapshot/"+config["dataset"]+"_"+str(args.hash_bit)+"bit_"+args.prefix

    config["prep"] = {"test_10crop":False, "resize_size":256, "crop_size":224}
    if config["dataset"] == "imagenet":
        config["data"] = {"database":{"list_path":"../data/imagenet/database.txt", "batch_size":16}, \
                          "test":{"list_path":"../data/imagenet/test.txt", "batch_size":16}}
        config["R"] = 1000
    elif config["dataset"] == "nus_wide":
        config["data"] = {"database":{"list_path":"../data/nus_wide/database.txt", "batch_size":16}, \
                          "test":{"list_path":"../data/nus_wide/test.txt", "batch_size":16}}
        config["R"] = 5000
    elif config["dataset"] == "coco":
        config["data"] = {"database":{"list_path":"../data/coco/database.txt", "batch_size":16}, \
                          "test":{"list_path":"../data/coco/test.txt", "batch_size":16}}
        config["R"] = 5000
    code_and_label = predict(config)

    mAP = mean_average_precision(code_and_label, config["R"])
    print(config["snapshot_path"])
    print ("MAP: "+ str(mAP))
    print("saving ...")
    save_code_and_label(code_and_label, osp.join(config["output_path"], args.snapshot))
    print("saving done")


