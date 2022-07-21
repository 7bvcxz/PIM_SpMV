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

sparsity = 0.95
nnz_per_cluster = int(num_idx * num_row * (1-sparsity) / 32)  # 16000
nnz_per_cluster_l = int(nnz_per_cluster * 0.99)
nnz_per_cluster_h = int(nnz_per_cluster * 1.01)

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

def init_centroid():
    tmp = random.sample(range(num_row), 32)
    for k_idx in range(k):
        centroid[k_idx] = fc[tmp[k_idx]]
    return centroid

def AssignRows(layer, weight, y_, centroid_):
    print("AssignRows!")
    
    sT_s = (centroid_ * centroid_).sum(axis=1)
    #print("1. sT_s shape : ", sT_s.shape)

    x_s = torch.matmul(layer, centroid_.T)
    #print("2. x_s shape : ", x_s.shape)

    a = x_s.T * (-2)
    #print("3. a shape : ", a.shape)

    for k_idx in range(k):
        a[k_idx] = a[k_idx] + sT_s[k_idx]
    a = a*weight
    #print("4. a shape : ", a.shape)

    ###### Now no torch.Tensor.... Only Numpy!! ######
    a = a.numpy()
    weight = weight.numpy()

    C = np.ones((k, 1))
    D = np.ones((num_row, 1))

    y_ = cvp.Variable((k, num_row))
    obj = cvp.Minimize(cvp.sum(cvp.multiply(y_,a)))
    constraints = [nnz_per_cluster_l <= cvp.matmul(y_, weight),
                   cvp.matmul(y_, weight) <= nnz_per_cluster_h,
                   0 <= y_,
                   cvp.matmul(y_.T, C) == D]

    prob = cvp.Problem(obj, constraints)
    result = prob.solve()

    y_ = torch.Tensor(y_.value)
    print("Assignment y_ : ", y_)

    return y_
    
def Update(layer, weight, y_, centroid_):
    print("Update!")
    centroid = torch.zeros((k, num_idx)) - 1
    flag = 1
    for k_idx in range(k):
        tot_weight = torch.matmul(y_[k_idx], weight) 
        tot_weight_loc = torch.matmul(y_[k_idx], weight.reshape(-1, 1)*layer)
        centroid[k_idx] = tot_weight_loc / tot_weight
        if torch.equal(centroid[k_idx], centroid_[k_idx]) == False and flag == 1:
            flag = 0
            print("Centroid changed...")
    if flag == 1:
        print("Centroid set Ended!")

    return centroid, flag

def print_(name, data):
    for i in data.cpu().numpy():
        print(i, end=" ")
    print()

print("< Define Functions Ended! >\n")


print("< Pruning Layer... ( sparsity: x", sparsity, ") >")
fc = prune_layer(fc, sparsity)
nnz = torch.count_nonzero(fc, dim=1) * 1.0
print("nnz : ", torch.count_nonzero(fc).item())
print("1000*2048 25% : ", int(1000*2048/4))
print("< Pruning Finished! >\n")

#########################################################################
print("< Start!! >\n")
print("##################################################################")
print("##################################################################")

print("< Selecting k rows by Weight-Balanced K-Means... >")
print("< Initing Centroid... >")
centroid = init_centroid()
print("< Init Centroid Ended! >\n")
flag = 0
cnt_iter = 1
while (flag == 0):
    print("W-B-K-Means iter : ", cnt_iter)
    cnt_iter = cnt_iter + 1
    y = AssignRows(fc, nnz, y, centroid)
    centroid, flag = Update(fc, nnz, y, centroid)

y = y.T
nnz_k = torch.zeros(k)
col_k = torch.zeros((k, num_idx))
row_k = torch.zeros(k)

for row_i in range(num_row):
    k_i = torch.argmax(y[row_i])
    nnz_k[k_i] = nnz_k[k_i] + nnz[row_i]
    col_k[k_i] = col_k[k_i] + fc[row_i]
    row_k[k_i] = row_k[k_i] + 1

col_k = torch.count_nonzero(col_k, dim=1)
print_("nnz_k", nnz_k)
print_("col_k", col_k)
print_("row_k", row_k)

total = nnz_k + col_k + row_k
print_("total", total)
print("sum : ", nnz_k.sum().item())
print("max : ", max(total))

print("##################################################################")
print("##################################################################")
print("finished!!")
#########################################################################
