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

num_row = 1000
num_idx = 2048
k = 32

sparsity = 0.95
bound = 0.01
nnz_per_cluster = int(num_idx * num_row * (1-sparsity) / k)  # 16000
nnz_per_cluster_l = int(nnz_per_cluster * (1-bound))
nnz_per_cluster_h = int(nnz_per_cluster * (1+bound))

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

def init_centroid():
    tmp = random.sample(range(num_row), 32)
    for k_idx in range(k):
        centroid[k_idx] = fc[tmp[k_idx]]
    return centroid

def AssignRows(layer, weight, y_, cluster_, centroid_):
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
    
def Update(layer, weight, y_, cluster_, centroid_):
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

def SelectKRows(layer, centroid_):
    KRows_idx = torch.zeros(k)
    for k_idx in range(k):
        tmp = (layer - centroid_[k_idx])**2
        tmp = tmp.sum(axis=1)
        KRows_idx[k_idx] = torch.argmin(tmp).item()
        layer[torch.argmin(tmp).item()] = torch.zeros(num_idx) - 1
    return KRows_idx

def simil(row0, row1):
    intersection = torch.count_nonzero(torch.logical_and(row0, row1)*1).item()
    union = torch.count_nonzero(torch.logical_or(row0, row1)*1)
    nB = torch.count_nonzero(row1).item()
    #score_ =  1 / ((nB - intersection)/nB + 1)
    #score_ =  1 / (nB - intersection + 1)
    score_ = intersection   # SpaceA
    #score_ = intersection / union
    return score_

def InitScore(layer, nnz_k, col_k, row_k, dns_k):
    for k_idx in range(k):
        for row_idx in range(num_row):
            if (layer[row_idx][0] == -1):
                score[k_idx][row_idx] = -444
            else:
                score[k_idx][row_idx] = simil(col_k[k_idx], layer[row_idx])

def Assign1Row(layer, nnz_k, col_k, row_k, dns_k):
    k_i = torch.argmax(score).item() // num_row
    row_i = torch.argmax(score).item() % num_row
    print("score: {:.5f}".format(score[k_i][row_i].item()), end="      \t")
    if score[k_i][row_i] == -333: # Assign to smallest nnz_k
        k_i = torch.argmin(nnz_k).item()
        print("over nnz", end='\t')
    elif score[k_i][row_i] == -444: # Should not happen..
        print("ERROR!", end='\t')
    intersection = torch.count_nonzero(torch.logical_and(col_k[k_i], layer[row_i])*1).item()
    nnz_k[k_i] = nnz_k[k_i] + torch.count_nonzero(layer[row_i]).item()
    col_k[k_i] = col_k[k_i] + layer[row_i]
    row_k[k_i] = row_k[k_i] + 1
    dns_k[k_i] = dns_k[k_i] + fc[row_i]
    print("row ", row_i, "   \t→\tcluster", k_i,
          "\tcol ", torch.count_nonzero(col_k[k_i]).item(),
          "\tnnz +", torch.count_nonzero(layer[row_i]).item(),
          "\tinter ", intersection,
          "\tsub ", torch.count_nonzero(layer[row_i]).item() - intersection)
    layer[row_i] = torch.zeros(num_idx) - 1
    #print(score)
    for row_idx in range(num_row):
        if (layer[row_idx][0] == -1):
            score[k_i][row_idx] = -444
        elif (nnz_k[k_i] + torch.count_nonzero(layer[row_idx]) > nnz_per_cluster_h):
            score[k_i][row_idx] = -333
        else:
            score[k_i][row_idx] = simil(col_k[k_i], layer[row_idx])
    for k_idx in range(k):
        score[k_idx][row_i] = -444
    return layer, nnz_k, col_k, row_k, dns_k

def print_(name, data):
    for i in data.cpu().detach().numpy():
        print(i, end=" ")
    print()


print("< Define Functions Ended! >\n")

print("< Pruning Layer... ( sparsity: x", sparsity, ") >")
fc = prune_layer(fc, sparsity)
nnz = torch.count_nonzero(fc, dim=1) * 1.0
print("nnz : ", torch.count_nonzero(fc).item())
print("target nnz : ", int(1000*2048 * (1-sparsity)))
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
    y = AssignRows(fc, nnz, y, cluster, centroid)
    centroid, flag = Update(fc, nnz, y, cluster, centroid)

print("< Select k rows according to W-B-K-Means...>")
y = y.T  # 1000 x 32
nnz_k = torch.zeros(k)
col_k = torch.zeros((k, num_idx))
row_k = torch.zeros(k)
for row_idx in range(num_row):
    k_idx = torch.argmax(y[row_idx])
    nnz_k[k_idx] = nnz_k[k_idx] + torch.count_nonzero(fc[row_idx]).item()
    col_k[k_idx] = col_k[k_idx] + fc[row_idx]
    row_k[k_idx] = row_k[k_idx] + 1

col_k = torch.count_nonzero(col_k, dim=1)
print_("nnz_k : \n", nnz_k)
print_("col_k : \n", col_k)
print_("row_k : \n", row_k)

total = nnz_k + col_k + row_k
print("total : \n", total)
print("max : ", max(total).item())


torch.save(centroid, 'w_b_k_means_centroid.pt')
centroid = torch.load('w_b_k_means_centroid.pt')

fc_ = torch.clone(fc)

nnz_k = torch.zeros(k)
col_k = torch.zeros((k, num_idx))
row_k = torch.zeros(k)
dns_k = torch.zeros((k, num_idx))

print("centroid : \n", centroid)


KRows_idx = SelectKRows(fc_, centroid)
print(KRows_idx)
for k_idx in range(k):
    nnz_k[k_idx] = torch.count_nonzero(fc[int(KRows_idx[k_idx])]).item()
    col_k[k_idx] = fc[int(KRows_idx[k_idx])]
    row_k[k_idx] = 1
    dns_k[k_idx] = fc[int(KRows_idx[k_idx])]
    fc_[int(KRows_idx[k_idx])] = torch.zeros(num_idx) - 1
print("< Selecting k rows Ended! >\n")

##########################################################################
InitScore(fc_, nnz_k, col_k, row_k, dns_k)
for i in range(num_row - k):
    print("iter : ", i+1, end='\t')
    fc_, nnz_k, col_k, row_k, dns_k = Assign1Row(fc_, nnz_k, col_k, row_k, dns_k)
##########################################################################
InitScore(fc_, nnz_k, col_k, row_k, dns_k)
print("score : \n", score)

col_k = torch.count_nonzero(col_k, dim=1)
print_("nnz_k : \n", nnz_k)
print_("col_k : \n", col_k)
print_("row_k : \n", row_k)

total = nnz_k + col_k + row_k
print_("total : \n", total)
print("sum : ", nnz_k.sum().item())
print("max : ", max(total).item())
print("rows : ", row_k.sum().item())

print("##################################################################")
print("##################################################################")
print("finished!!")
