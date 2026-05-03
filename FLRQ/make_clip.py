import torch
import torch.nn as nn
import gc
from quantizer import *


@torch.no_grad()
def auto_clip_lora(
    w, lora_W ,input_feat, n_bit, group_size = 0, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]

    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    lora_W = lora_W.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        lw = lora_W[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        org_w = lw + w
        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * org_w).sum(dim=-1)  # co, n_token, n_group
        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor(cur_w, bit=n_bit,q_group_size=group_size)
            #q_w = pseudo_quantize_tensor0(cur_w, bit=n_bit,group_size=group_size)
            cur_out = (input_feat * (q_w + lw)).sum(dim=-1)
            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)
    best_max_val = best_max_val.squeeze(1)
    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    w_all = w_all.reshape(*best_max_val.shape[:2], -1)
    w_all = torch.clamp(w_all, -best_max_val, best_max_val)

    w_all = w_all.reshape(org_w_shape)
    return w_all