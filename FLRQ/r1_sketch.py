import torch
import torch.nn as nn
import argparse
import os
import json
import math


from numpy import random
qtype = torch.bfloat16
def find_max_abs_value(A):
    abs_A = torch.abs(A)
    max_abs_value = torch.max(abs_A)
    return max_abs_value


def compute_r1sketch(A,iter = 1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, n = A.shape
    
    # Generates a random normal distribution vector x of length n
    x_numpy = random.normal(loc=0, scale=1, size=(n))
    
    x = torch.from_numpy(x_numpy)
    x = x.to(torch.float64)
    x = x.to(device)
    y = torch.matmul(A, x)

    for i in range(iter):
        tmp = torch.matmul(A.T, y)
        y = torch.matmul(A, tmp)

    A_L = y
    A_R = torch.matmul(A.T, A_L)

    normP = torch.norm(A_L, p=2)
    normQ = torch.norm(A_R, p=2)

    Var_AL = normQ/(normP*normP)
    Var_AR = 1.0/normQ

    A_R = A_R*Var_AR
    A_L = A_L*Var_AL
    return A_L, A_R


def get_best_sketch(weights, bits, ratio=0.01, max_sketch_iter = 2, fix_rank = 0):
    row = weights.size(0)
    col = weights.size(1)
    min_rank = min(row,col)

    weight_cp = weights
    if weights.dtype == torch.float16 or weights.dtype == torch.bfloat16:
        weights = weights.to(torch.float64)

    
    max_absW_0 = find_max_abs_value(weights)
    max_absW_iter = find_max_abs_value(weights)

    skethc_L = []
    skethc_R = []

    max_iter = {}
    max_ptr = 8#int(min_rank*ratio*(bits+0.001)/32.0)
    min_ptr = 0#max(max_ptr//4,4)
    VS_L = None
    VS_R = None
    VS_L_16 = None
    VS_R_16 = None
    work_rank = 0
    if fix_rank != 0:
        work_rank = fix_rank
        for i in range(0,work_rank):
            r1_L,r1_R = compute_r1sketch(weights,max_sketch_iter)
            r1_matrix = torch.outer(r1_L, r1_R)
            weights = weights - r1_matrix
            max_absW = find_max_abs_value(weights)
            max_iter[i] = max_absW
            skethc_L.append(r1_L)
            skethc_R.append(r1_R)
            VS_L = torch.vstack(skethc_L[:work_rank])
            VS_R = torch.vstack(skethc_R[:work_rank])
        max_now = max_iter[work_rank-1]
    else:
        for i in range(0,min_rank):
            r1_L,r1_R = compute_r1sketch(weights,max_sketch_iter)
            r1_matrix = torch.outer(r1_L, r1_R)
            weights = weights - r1_matrix
            max_absW = find_max_abs_value(weights)
            max_iter[i] = max_absW
            P = max_absW_0/max_absW
            K = 1.0+(16.0*(i+1)*(row+col)/(1.0*bits*row*col))
            Q = (bits + math.log(P,2) )/(1.0*bits)
            skethc_L.append(r1_L)
            skethc_R.append(r1_R)
            if(i>=max_ptr):
                if(((max_iter[i-max_ptr]-max_iter[i])/max_iter[i-max_ptr])<0.1):
                    work_rank = i-max_ptr//2
                    break
            if(K>(1.0+ratio)):
                work_rank = i
                break
            if (K >= Q and i > max_sketch_iter+min_ptr):
                work_rank = i - 1
                break
            else:
                work_rank = i
        if(work_rank == 0):
            work_rank = 1
        if (max_absW_0 - max_iter[work_rank-1])/max_absW_0 < 0.1:
            work_rank = min_ptr
            max_now = max_absW_0

        if(work_rank>=1):
            VS_L = torch.vstack(skethc_L[:work_rank])
            VS_R = torch.vstack(skethc_R[:work_rank])
            max_now = max_iter[work_rank-1]

        if(work_rank!=0 and max_absW_0<=max_iter[work_rank-1]):
            work_rank = 0
            VS_L.zero_()
            VS_R.zero_()
            max_now = max_absW_0

    if work_rank!=0:
        VS_L_16 = VS_L.to(qtype)
        VS_R_16 = VS_R.to(qtype)
        weight_cp = weight_cp - torch.matmul(VS_L_16.T,VS_R_16)
        max_now = find_max_abs_value(weight_cp)
    return weight_cp,VS_L_16,VS_R_16,max_absW_0,max_now,work_rank




def get_best_sketch_svd(weights, bits, ratio=0.01, max_sketch_iter = 4, fix_rank = 0):
    row = weights.size(0)
    col = weights.size(1)
    min_rank = min(row,col)

    weight_cp = weights
    if weights.dtype == torch.float16 or weights.dtype == torch.bfloat16:
        weights = weights.to(torch.float32)

    
    max_absW_0 = find_max_abs_value(weights)
    max_absW_iter = find_max_abs_value(weights)
    U, s, Vt = torch.linalg.svd(weights, full_matrices=True)
    skethc_L = []
    skethc_R = []

    max_iter = {}
    #print(f"max_absW0: {max_absW_0}, rank: {0}")

    max_iter = {}
    max_ptr = int(min_rank*ratio*(bits+0.001)/32.0)
    min_ptr = 0#max(max_ptr//4,4)
    VS_L = None
    VS_R = None
    VS_L_16 = None
    VS_R_16 = None
    work_rank = 0
    if fix_rank != 0:
        work_rank = fix_rank
        for i in range(0,work_rank):
            r1_L = U[:, i].reshape(-1)* s[i]
            r1_R = Vt[i, :].reshape(-1)
            r1_matrix = torch.outer(r1_L, r1_R)
            weights = weights - r1_matrix
            max_absW = find_max_abs_value(weights)
            max_iter[i] = max_absW
            skethc_L.append(r1_L)
            skethc_R.append(r1_R)
            VS_L = torch.vstack(skethc_L[:work_rank])
            VS_R = torch.vstack(skethc_R[:work_rank])
        max_now = max_iter[work_rank-1]


    #plot_weight_histogram(weights, "W2")
    if work_rank!=0:
        VS_L_16 = VS_L.to(qtype)
        VS_R_16 = VS_R.to(qtype)
        weight_cp = weight_cp - torch.matmul(VS_L_16.T,VS_R_16)
        max_now = find_max_abs_value(weight_cp)
    return weight_cp,VS_L_16,VS_R_16,max_absW_0,max_now,work_rank
