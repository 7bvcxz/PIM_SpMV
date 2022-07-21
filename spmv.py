import torch
import cvxpy as cvp
import numpy as np
import random
import time

import argparse

# Parse 만들기 => 그런데, general하게 만들 것.

num_row = 0
num_idx = 0
k = 0
sparsity = 0
nnz = 0
bound_ratio = 0
nnz_per_cluster = 0
nnz_per_cluster_h = 0
nnz_per_cluster_l = 0

def init(layer_, sparsity_, k_, bound_ratio_):
    global num_row
    global num_idx
    global k
    global sparsity
    global nnz
    global bound_ratio
    global nnz_per_cluster

    num_row = layer.shape()[0]
    num_idx = layer.shape()[1]
    k = k_
    sparsity = sparsity_
    nnz = torch.count_nonzero(layer, dim=1) * 1.0
    bound_ratio = bound_ratio_
    nnz_per_cluster = int(num_row * num_idx * (1-sparsity) / k)
    nnz_per_cluster_h = int(nnz_per_cluster * (1+bound_ratio))
    nnz_per_cluster_l = int(nnz_per_cluster * (1-bound_ratio))

def prune_layer(layer, sparsity):
    layer_ = layer.abs().reshape(-1).sort()
    threshold = layer_.values[int(num_idx*num_row*sparsity)-1].item()
    return (layer.abs() > threshold)*1.0

def sequence_row_equal(layer, sparsity, nnz, k, nnz_k, col_k, row_k):
    k_i = 0
    for row_i in range(num_row):
        if row_i > int(num_row * (k_i+1) / k):
            k_i = k_i + 1
        nnz_k[k_i] = nnz_k[k_i] + nnz[row_i]
        col_k[k_i] = col_k[k_i] + layer[row_i]
        row_k[k_i] = row_k[k_i] + 1

    return nnz_k, col_k, row_k

def sequence_row_threshold(layer, sparsity, nnz, k, nnz_k, col_k, row_k, bound_ratio):
    k_i = 0
    for row_i in range(num_row):
        if nnz_k[k_i] + nnz[row_i] > nnz_per_cluster_h:
            k_i = k_i + 1
        nnz_k[k_i] = nnz_k[k_i] + nnz[row_i]
        col_k[k_i] = torch.logical_or(col_k[k_i], layer[row_i])*1
        row_k[k_i] = row_k[k_i] + 1

    return nnz_k, col_k, row_k

def sequence_nnz_threshold(layer, sparsity, nnz, k, nnz_k, col_k, row_k):
    k_i = 0
    for row_i in range(num_row):
        if nnz_k[k_i] + nnz[row_i] > nnz_per_cluster:
            front_nnz = nnz_per_cluster - nnz_k[k_i]
            nnz_k[k_i] = int(num_row * num_idx * (1-sparsity) / 32)
            col_k[k_i] = torch.logical_or(col_k[k_i], layer[row_i])*2
            row_k[k_i] = row_k[k_i] + 1
            k_i = k_i + 1
            nnz_k[k_i] = nnz_k[k_i] + (nnz[row_i] - front_nnz)
            col_k[k_i] = torch.logical_or(col_k[k_i], layer[row_i])*1 # just doing for now
            row_k[k_i] = row_k[k_i] + 1
        else:
            nnz_k[k_i] = nnz_k[k_i] + nnz[row_i]
            col_k[k_i] = torch.logical_or(col_k[k_i], layer[row_i])*1
            row_k[k_i] = row_k[k_i] + 1

    return nnz_k, col_k, row_k

def space_a(layer, sparsity, nnz, k, nnz_k, col_k, row_k):
    score = torch.zeros(k)
    for row_i in range(num_row):
        print("iter : ", row_i+1, end='\t')
        for k_i in range(k):
            if nnz_k[k_i] + nnz[row_i] > nnz_per_cluster:
                score[k_i] = -1 * (nnz_k[k_i] + nnz[row_i] - nnz_per_cluster)
            else:
                overlap = torch.count_nonzero(torch.logical_and(col_k, layer[row_i])*1).item()
                score[k_i] = max(overlap / nnz[row_i], 1 / (nnz_k[k_i] + nnz[row_i]))
        #print(torch.argmax(score), " ", score)
        k_i = torch.argmax(score)
    
        print("row ", row_i, "   \t→ cluster ", k_i.item())
        nnz_k[k_i] = nnz_k[k_i] + nnz[row_i]
        col_k[k_i] = torch.logical_or(col_k[k_i], layer[row_i])*1
        row_k[k_i] = row_k[k_i] + 1 

    return nnz_k, col_k, row_k

############################### Weight - Balanced - K - Means ################################
def init_centroid(layer):
    tmp = random.sample(range(num_row), k)
    for k_idx in range(k):
        centroid[k_idx] = layer[tmp[k_idx]]
    return centroid

def AssignRows(layer, sparsity, nnz, k, y_, centroid_, bound_ratio):
    sT_s = (centroid_ * centroid_).sum(axis=1)
    #print("1. sT_s shape : ", sT_s.shape)

    x_s = torch.matmul(layer, centroid_.T)
    #print("2. x_s shape : ", x_s.shape)

    a = x_s.T * (-2)
    #print("3. a shape : ", a.shape)

    for k_idx in range(k):
        a[k_idx] = a[k_idx] + sT_s[k_idx]
    a = a*nnz
    #print("4. a shape : ", a.shape)

    ###### Now no torch.Tensor.... Only Numpy!! ######
    a = a.numpy()
    nnz = nnz.numpy()

    C = np.ones((k, 1))
    D = np.ones((num_row, 1))

    y_ = cvp.Variable((k, num_row))
    obj = cvp.Minimize(cvp.sum(cvp.multiply(y_,a)))
    constraints = [nnz_per_cluster_l <= cvp.matmul(y_, nnz),
                   cvp.matmul(y_, nnz) <= nnz_per_cluster_h,
                   0 <= y_,
                   cvp.matmul(y_.T, C) == D]

    prob = cvp.Problem(obj, constraints)
    result = prob.solve()

    y_ = torch.Tensor(y_.value)
    print("Assignment y_ : ", y_)

    return y_
    
def Update(layer, nnz, k, y_, centroid_):
    centroid = torch.zeros((k, num_idx)) - 1
    flag = 1
    for k_idx in range(k):
        tot_nnz = torch.matmul(y_[k_idx], nnz) 
        tot_nnz_loc = torch.matmul(y_[k_idx], nnz.reshape(-1, 1)*layer)
        centroid[k_idx] = tot_nnz_loc / tot_nnz
        if torch.equal(centroid[k_idx], centroid_[k_idx]) == False and flag == 1:
            flag = 0
            print("Centroid changed...")
    if flag == 1:
        print("Centroid set Ended!")

    return centroid, flag

def w_b_k_means(layer, sparsity, nnz, k, nnz_k, col_k, row_k, bound_ratio):
    centroid = init_centroid(layer, k)
    y = torch.zeros((k, num_row))
    flag = 0
    cnt_iter = 1
    
    while (flag == 0):
        print("W-B-K-Means iter : ", cnt_iter)
        cnt_iter = cnt_iter + 1
        y = AssignRows(layer, sparsity, nnz, k, y, centroid, bound_ratio)
        centroid, flag = Update(layer, nnz, k, y, centroid)

    nnz_k = torch.zeros(k)
    col_k = torch.zeros((k, num_idx))
    row_k = torch.zeros(k)

    for row_i in range(num_row):
        k_i = torch.argmax(y[row_i])
        nnz_k[k_i] = nnz_k[k_i] + nnz[row_i]
        col_k[k_i] = col_k[k_i] + layer[row_i]
        row_k[k_i] = row_k[k_i] + 1

    return nnz_k, col_k, row_k, centroid

##################################### Max Score ######################################
def SelectKRows(layer, centroid):
    KRows_idx = torch.zeros(k)
    for k_idx in range(k):
        tmp = (layer - centroid_[k_idx])**2
        tmp = tmp.sum(axis=1)
        KRows_idx[k_idx] = torch.argmin(tmp).item()
        layer[torch.argmin(tmp).item()] = torch.zeros(num_idx) - 1
    return KRows_idx

def simil(row0, row1):
    version = 1
    if version==0:
        union = torch.count_nonzero(torch.logical_or(row0, row1)*1)
        intersection = torch.count_nonzero(torch.logical_and(row0, row1)*1)
        return intersection / union
    elif version==1:
        p_union = torch.count_nonzero(row0)
        union = torch.count_nonzero(torch.logical_or(row0, row1)*1)
        intersection = torch.count_nonzero(torch.logical_and(row0, row1)*1)
        score_ = intersection / union
        if union == p_union:  # union not increasing! happy
            return score_
        else: # union incresing... sad (penalty -100)
            return -100 + score_

def InitScore(layer, nnz_k, col_k, row_k, dns_k):
    for k_idx in range(k):
        for row_idx in range(num_row):
            if (layer[row_idx][0] == -1):
                score[k_idx][row_idx] = -444
            elif (nnz_k[k_idx] + torch.count_nonzero(layer[row_idx]) > nnz_per_cluster_h):
                score[k_idx][row_idx] = -333
            else:
                score[k_idx][row_idx] = simil(col_k[k_idx], layer[row_idx])

def Assign1Row(layer, nnz_k, col_k, row_k, dns_k):
    k_i = int(torch.argmax(score) / num_row)
    row_i = int(torch.argmax(score) - k_i * num_row)
    print("score: ", round(score[k_i][row_i].item(), 2), end="      \t")
    if score[k_i][row_i] == -333: # Assign to smallest nnz_k
        k_i = torch.argmin(nnz_k)
        print("??????????????????????????????")
    elif score[k_i][row_i] == -444: # Should not happen..
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("row ", row_i, "   \t→\tcluster", k_i)
    nnz_k[k_i] = nnz_k[k_i] + torch.count_nonzero(layer[row_i]).item()
    col_k[k_i] = torch.logical_or(layer[row_i], col_k[k_i])
    row_k[k_i] = row_k[k_i] + 1
    dns_k[k_i] = dns_k[k_i] + layer[row_i]
    layer[row_i] = torch.zeros(num_idx) - 1
    #print(score)
    for row_idx in range(num_row):
        if (layer[row_idx][0] == -1):
            score[k_i][row_idx] = -444
        elif (nnz_k[k_i] + torch.count_nonzero(layer[row_idx]) > nnz_per_cluster_h):
            score[k_i][row_idx] = -333
        else:
            score[k_i][row_idx] = simil(col_k[k_i], layer[row_idx])
    return layer, nnz_k, col_k, row_k, dns_k

def max_score(layer, sparsity, nnz, k, nnz_k, col_k, row_k, bound_ratio):
    centroid, _, _, _ = w_b_k_means(layer, sparsity, nnz, k, nnz_k, col_k, row_k, bound_ratio):

    layer_ = torch.clone(layer)

    nnz_k = torch.zeros(k)
    col_k = torch.zeros((k, num_idx))
    row_k = torch.zeros(k)

    print("centroid : \n", centroid)
    KRows_idx = SelectKRows(layer_, centroid)
    print(KRows_idx)
    for k_idx in range(k):
        nnz_k[k_idx] = torch.count_nonzero(layer[int(KRows_idx[k_idx])]).item()
        col_k[k_idx] = layer[int(KRows_idx[k_idx])]
        row_k[k_idx] = 1
        dns_k[k_idx] = layer[int(KRows_idx[k_idx])]
        layer_[int(KRows_idx[k_idx])] = torch.zeros(num_idx) - 1
    print("< Selecting k rows Ended! >\n")


def main():
    InitScore(layer_, nnz_k, col_k, row_k, dns_k)
    for i in range(num_row - k):
        print("iter : ", i+1, end='\t')
        layer_, nnz_k, col_k, row_k, dns_k = Assign1Row(layer_, nnz_k, col_k, row_k, dns_k)
    InitScore(layer_, nnz_k, col_k, row_k, dns_k)


def __name__ == "__main__":
    main()
