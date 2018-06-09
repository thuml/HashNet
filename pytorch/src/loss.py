import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def pairwise_loss(outputs1, outputs2, label1, label2, sigmoid_param=1.0, l_threshold=15.0, class_num=1.0):
    similarity = Variable(torch.mm(label1.data.float(), label2.data.float().t()) > 0).float()
    dot_product = torch.mm(outputs1, outputs2.t())
    exp_product = torch.exp(sigmoid_param*dot_product)
    mask_dot = dot_product.data > l_threshold
    mask_exp = dot_product.data <= l_threshold
    mask_positive = similarity.data > 0
    mask_negative = similarity.data <= 0
    mask_dp = mask_dot & mask_positive
    mask_dn = mask_dot & mask_negative
    mask_ep = mask_exp & mask_positive
    mask_en = mask_exp & mask_negative

    dot_loss = dot_product * (1-similarity)
    exp_loss = torch.log(1+exp_product) - similarity * dot_product
    #loss = torch.sum(torch.masked_select(exp_loss, Variable(mask_exp))) + torch.sum(torch.masked_select(dot_loss, Variable(mask_dot)))
    loss = (torch.sum(torch.masked_select(exp_loss, Variable(mask_ep))) + torch.sum(torch.masked_select(dot_loss, Variable(mask_dp)))) * class_num + torch.sum(torch.masked_select(exp_loss, Variable(mask_en))) + torch.sum(torch.masked_select(dot_loss, Variable(mask_dn)))

    return loss / (outputs1.size(0) * outputs2.size(0))

def quantization_loss(outputs):
    return torch.sum(-torch.log(torch.cosh(torch.abs(outputs) - 1))) / outputs.size(0)
