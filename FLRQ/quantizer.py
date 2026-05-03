import torch
import torch.nn as nn
import argparse
import os
import json
import numpy as np

from r1_sketch import *


def find_params_group(x, bit = 4, group_size = 128, weight=False):
    dev = x.device

    maxq = torch.tensor(2 ** bit - 1)
    maxq = maxq.to(dev)

    shape = x.shape

    # use per channel
    x = x.flatten(1)

    assert x.size(0) % 128 == 0
    xmin = x.unfold(1, group_size, group_size).min(dim=-1).values
    xmax = x.unfold(1, group_size, group_size).max(dim=-1).values

    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1

    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)

    return  scale, zero


def get_scale_fp4(self, input, bits, mantissa_bit, bias):
    
    M = mantissa_bit
    E = bits - 1 - M
    bias = bias.float()
    maxval = (2 - 2 ** (-M)) * 2 ** (
            2**E - 1 - bias
        )
    minval = -maxval

    input = torch.min(torch.max(input, minval), maxval)

    input_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(input)) + bias)), 1.0)

    return input, 2.0 ** (input_log_scales - M - bias)

def quantize_lora_group(x, scale, zero, bit = 4,group=128, L = 0, R = 0, rank=0):

    dev = x.device
    maxq = torch.tensor(2 ** bit - 1)
    maxq = maxq.to(dev)

    shape = x.shape
    x_unfolded = x.unfold(1, group, group)
    q = torch.round(x_unfolded / scale.unsqueeze(-1) + zero.unsqueeze(-1))
    q = torch.clamp(q, 0, maxq)

    x_dequantized = (q - zero.unsqueeze(-1)) * scale.unsqueeze(-1)
    
    x_dequantized_folded = x_dequantized.reshape(shape)
    if rank == 0:
        return x_dequantized_folded

    return x_dequantized_folded + torch.matmul(L.T,R)


def quant_sketch_save_full_process_res(W, feat_scale, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128):

    qscale, qzero = find_params_group(W, bit = bit , group_size = groupsize,weight=True) 
    Reduce = quantize_lora_group(
        W, qscale, qzero, group=groupsize
    )
    res = W - Reduce
    feat_scale = feat_scale.cuda()
    res_scale_T = torch.diag(feat_scale) @ res.T

    W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch(res_scale_T, bit, ratio = ratio, fix_rank = fix_rank)


    if srank != 0:
        lora_res = torch.matmul(r1_L.T,r1_R)
        lora_res = torch.diag(feat_scale.float()).inverse().half() @ lora_res
        Reduce = Reduce + lora_res.T
    else:
        Reduce = Reduce
    
    return Reduce,srank


def quant_lora_scale(W, feat_scale, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128):

    feat_scale = feat_scale.cuda()

    W_scale_T = torch.diag(feat_scale) @ W.T

    W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch(W_scale_T, bit, ratio = ratio, fix_rank = fix_rank)

    if srank != 0:
        lora_W = torch.matmul(r1_L.T,r1_R)
        lora_W = torch.diag(feat_scale.float()).inverse().to(lora_W.dtype) @ lora_W
    else:
        Reduce = 0
        
    if srank != 0:
        res = W - lora_W.T
    else:
        res = W


    Reduce = pseudo_quantize_tensor(res, bit=bit,q_group_size=groupsize)
    if srank != 0:
        Reduce = Reduce + lora_W.T
    else:
        Reduce = Reduce
    
    return Reduce,srank




def get_scale_lora(W, feat_scale, bit = 4, fix_rank = 0, ratio = 0.1,groupsize = 128):

    feat_scale = feat_scale.cuda()

    W_scale_T = torch.diag(feat_scale) @ W.T

    W2,r1_L,r1_R,max_0,max_now,srank = get_best_sketch(W_scale_T, bit, ratio = ratio, fix_rank = fix_rank)

    if srank != 0:
        lora_W = torch.matmul(r1_L.T,r1_R)
        lora_W = torch.diag(feat_scale.float()).inverse().to(lora_W.dtype) @ lora_W
    else:
        lora_W = 0
    
    return lora_W,srank



# core quantization method (simulated quantization)
def pseudo_quantize_tensor(
    w, bit=8, q_group_size=-1
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)


    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0


    w = (
        torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
    ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w