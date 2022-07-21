print("< Library Loading... >")
import torch
import torchvision.models as models
import cvxpy as cvp
import numpy as np
import random
import time
print("< Library Loaded! >\n")

print("< Model Loading... >")
model = models.resnet50(pretrained=True)
print("< Model Loaded! >\n")

#########################################################################
print("< Configuring Stuff... >")
torch.set_default_dtype(torch.float32)

k = 32
num_idx = 2048
num_row = 1000

sparsity = 0.99
nnz_per_cluster = int(num_idx * num_row * (1-sparsity) / 32)  # 16000

cluster = torch.zeros((k, num_row)) - 1
centroid = torch.zeros((k, num_idx)) - 1
score = torch.zeros((k, num_row)) - 1
y = torch.zeros((k, num_row))
fc = model.fc.weight
print(fc.reshape(-1).shape)
print("< Configure Ended! >\n")

print("< Defining Functions... >")
def prune_layer(layer, sparsity):
    layer_ = layer.abs().reshape(-1).sort()
    threshold = layer_.values[int(num_idx*num_row*sparsity)-1].item()
    return (layer.abs() > threshold)*1.0

print("< Define Functions Ended! >\n")

print("< Pruning Layer... ( sparsity: x", sparsity, ") >")
fc = prune_layer(fc, sparsity)
nnz = torch.count_nonzero(fc, dim=1) * 1.0
print("nnz : ", torch.count_nonzero(fc).item())
print("1000*2048 25% : ", int(1000*2048/4))
print("< Pruning Finished! >\n")

def print_(name, data):
    for i in data.cpu().detach().numpy():
        print(i, end=" ")
    print()

#########################################################################
print("< Start!! >\n")
print("##################################################################")
print("##################################################################")

nnz_k = torch.zeros(k)
col_k = torch.zeros((k, num_idx))
row_k = torch.zeros(k)

k_i = 0
for row_i in range(num_row):
    if row_i > int(num_row * (k_i+1) / k):
        k_i = k_i + 1
    nnz_k[k_i] = nnz_k[k_i] + nnz[row_i]
    col_k[k_i] = col_k[k_i] + fc[row_i]
    row_k[k_i] = row_k[k_i] + 1

col_k = torch.count_nonzero(col_k, dim=1) * 1.0
print_("nnz_k : ", nnz_k)
print_("col_k : ", col_k)
print_("row_k : ", row_k)

tot_k = nnz_k + col_k + row_k
print("sum : ", nnz_k.sum())
print("max : ", max(tot_k))






