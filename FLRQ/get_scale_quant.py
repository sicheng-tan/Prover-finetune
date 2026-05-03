import torch
import torch.nn as nn
import argparse
import os
import json

from quantizer import *
from make_clip import *
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2DecoderLayer



SCALE_CLAMP_MIN = 1e-4


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)

@torch.no_grad()
def quant_sketch(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos):

    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
    
    mean_feat = mean_feat.pow(2.4)
    mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)
    scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()

    qweight,srank = quant_lora_scale(layer.weight, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)

    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*16
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    return qweight


@torch.no_grad()
def quant_sketch_clip(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos):

    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
    

    #0.8 - 37.94,45.18,30.92
    #1.2 - 37.33,44.64,30.69
    #1.4 - 37.18,44.81,30.68
    #1.8 - 37.04,44.97,30.49
    #2.4 - 37.13,44.58,30.42
    mean_feat = mean_feat.pow(2.4)
    mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)
    scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()

    lora_W,srank = get_scale_lora(layer.weight, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
    if (srank!=0):
        res = layer.weight - lora_W.T
    else:
        res = layer.weight
        lora_W = torch.zeros_like(layer.weight)
    clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=20, max_shrink=0.5, n_sample_token=512)
    q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)
    if (srank!=0):
        qweight = q_res + lora_W.T
    else:
        qweight = q_res
    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*16
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    return qweight
    

@torch.no_grad()
def quant_sketch_clip_iter_for(layer, input_feat, w_bit, group_size, fix_rank, ratio, quant_infos):

    mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
    
    mean_feat = mean_feat.pow(2.4)
    mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)
    scales = mean_feat / (mean_feat.max() * mean_feat.min()).sqrt()
    print("scale absmax = ",torch.abs(scales).max().item(),scales.dtype)
    lora_W,srank = get_scale_lora(layer.weight, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
    if (srank!=0):
        res = layer.weight - lora_W.T
    else:
        res = layer.weight
        lora_W = torch.zeros_like(layer.weight)
    clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=40, max_shrink=0.8, n_sample_token=512)
    q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)

    iter = 0
    if w_bit == 4:
        iter = 1
    elif w_bit == 3:
        iter = 5
    elif w_bit == 2: 
        iter = 20
    
    max_shrinks = [0.5,0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5,0.5]
    input_feat = input_feat.cuda()
    scale_org = input_feat @ layer.weight.T

    best_error = float("inf")
    best_lora = None
    best_iter = 0
    best_rank = 0
    best_qres = None
    for i in range(iter):
        w_res = layer.weight - q_res
        lora_W,srank = get_scale_lora(w_res, scales, fix_rank = fix_rank, bit = w_bit, ratio = ratio, groupsize = group_size)
        if (srank!=0):
            res = layer.weight - lora_W.T
        else:
            res = layer.weight
            lora_W = torch.zeros_like(layer.weight.T)
        iter_grid = 20
        if i == iter-1:
            iter_grid = 20
        clip_res = auto_clip_lora(res, lora_W, input_feat, w_bit, group_size, n_grid=iter_grid, max_shrink=max_shrinks[i], n_sample_token=512)
        q_res = pseudo_quantize_tensor(clip_res, bit=w_bit,q_group_size=group_size)


        scale_out = input_feat @ (q_res.T + lora_W)
        loss = (scale_org.to(input_feat.device) - scale_out.to(input_feat.device)).float().pow(2).sum().item()
        if loss < best_error:
            best_error = loss
            best_iter = i
            best_rank = srank
            best_lora = lora_W.T         
            best_qres = q_res

    if (best_rank!=0):
        qweight = best_qres + best_lora
    else:
        qweight = q_res

    quant_infos["lora_rank"] = quant_infos["lora_rank"] + srank
    quant_infos["lora_size"] =  quant_infos["lora_size"] + srank * (layer.weight.size(0) + layer.weight.size(1))*16
    quant_infos["total_size"] = quant_infos["total_size"] + layer.weight.size(0) * layer.weight.size(1)*16
    quant_infos["quant_size"] = quant_infos["quant_size"] + layer.weight.size(0) * layer.weight.size(1)*w_bit
    quant_infos["layer_cnt"] = quant_infos["layer_cnt"] + 1
    return qweight




@torch.no_grad()
def scale_quant_layer(layer, input_feat, quant_infos, w_bit=4, fix_rank = 0, ratio = 0.1, group_size = 128):
    if isinstance(layer, OPTDecoderLayer):
        layer.self_attn.q_proj.weight = nn.Parameter(quant_sketch_clip_iter_for(layer.self_attn.q_proj , input_feat["self_attn.q_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.self_attn.k_proj.weight = nn.Parameter(quant_sketch_clip_iter_for(layer.self_attn.k_proj , input_feat["self_attn.k_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.self_attn.v_proj.weight = nn.Parameter(quant_sketch_clip_iter_for(layer.self_attn.v_proj , input_feat["self_attn.v_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.self_attn.out_proj.weight = nn.Parameter(quant_sketch_clip_iter_for(layer.self_attn.out_proj , input_feat["self_attn.out_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.fc1.weight = nn.Parameter(quant_sketch_clip_iter_for(layer.fc1 , input_feat["fc1"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.fc2.weight = nn.Parameter(quant_sketch_clip_iter_for(layer.fc2 , input_feat["fc2"] , w_bit, group_size, fix_rank, ratio, quant_infos))
    elif isinstance(layer, (LlamaDecoderLayer, Qwen2DecoderLayer)):
      # attention input
      #ppl is very high at llama2 2b quantization, o_proj and down_proj can't do clip.
        layer.self_attn.q_proj.weight = nn.Parameter(quant_sketch_clip_iter_for(layer.self_attn.q_proj , input_feat["self_attn.q_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.self_attn.k_proj.weight = nn.Parameter(quant_sketch_clip_iter_for(layer.self_attn.k_proj , input_feat["self_attn.k_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.self_attn.v_proj.weight = nn.Parameter(quant_sketch_clip_iter_for(layer.self_attn.v_proj , input_feat["self_attn.v_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.mlp.gate_proj.weight = nn.Parameter(quant_sketch_clip_iter_for(layer.mlp.gate_proj , input_feat["mlp.gate_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        if(w_bit==2):
            layer.mlp.down_proj.weight = nn.Parameter(quant_sketch(layer.mlp.down_proj , input_feat["mlp.down_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos))
            layer.self_attn.o_proj.weight = nn.Parameter(quant_sketch(layer.self_attn.o_proj , input_feat["self_attn.o_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        else:
            layer.mlp.down_proj.weight = nn.Parameter(quant_sketch_clip_iter_for(layer.mlp.down_proj , input_feat["mlp.down_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos))
            layer.self_attn.o_proj.weight = nn.Parameter(quant_sketch_clip_iter_for(layer.self_attn.o_proj , input_feat["self_attn.o_proj"] , w_bit, group_size, fix_rank, ratio, quant_infos))      
    elif isinstance(layer, BloomBlock):
        layer.self_attention.query_key_value.weight = nn.Parameter(quant_sketch_clip_iter_for(layer.self_attention.query_key_value , input_feat["self_attention.query_key_value"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.self_attention.dense.weight = nn.Parameter(quant_sketch_clip_iter_for(layer.self_attention.dense , input_feat["self_attention.dense"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.mlp.dense_h_to_4h.weight = nn.Parameter(quant_sketch_clip_iter_for(layer.mlp.dense_h_to_4h , input_feat["mlp.dense_h_to_4h"] , w_bit, group_size, fix_rank, ratio, quant_infos))
        layer.mlp.dense_4h_to_h.weight = nn.Parameter(quant_sketch_clip_iter_for(layer.mlp.dense_4h_to_h , input_feat["mlp.dense_4h_to_h"] , w_bit, group_size, fix_rank, ratio, quant_infos))
    return layer