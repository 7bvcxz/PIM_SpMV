print("< Library Loading... >")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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

y = torch.zeros((k, num_row))
fc = model.fc.weight
print(fc.reshape(-1).shape)
print("< Configure Ended! >\n")

print("< Defining Functions... >")
def prune_layer(layer, sparsity):
    layer_ = layer.abs().reshape(-1).sort()
    threshold = layer_.values[int(num_idx*num_row*sparsity)-1].item()
    return (layer.abs() > threshold)*1.0

def print_(name, data):
    for i in data.cpu().detach().numpy():
        print(i, end=" ")
    print()

print("< Define Functions Ended! >\n")


print("< Pruning Layer... ( sparsity: x", sparsity, ") >")
fc = prune_layer(fc, sparsity)
nnz = torch.count_nonzero(fc, dim=1) * 1.0
print("nnz : ", torch.count_nonzero(fc).item())
print("< Pruning Finished! >\n")

#########################################################################
print("< Start!! >\n")
print("##################################################################")
print("##################################################################")

class Clustering(nn.Module):
    def __init__(self, D, K):
        super(Clustering, self).__init__()
        self.D = D.cuda()   # [R, C]
        self.R = D.shape[0]
        self.C = D.shape[1]
        self.K = K
        self.Y = nn.Parameter(torch.zeros((R, K)), requires_grad=True)
        self.sm = nn.Softmax(dim=1)

        torch.nn.init.kaiming_uniform_(self.Y)

    def forward(self):
        y = self.sm(self.Y)
        x = y.view(R, 1, K) * self.D.view(R, C, 1)  # [R, C, K]
        col_k = (1. - torch.prod(1-x, 0)).T # [K, C]
        nnz_k = torch.sum(x, dim=[0, 1]) # [K]

        col_per_cluster = torch.sum(col_k, dim=1)
        nnz_per_cluster = nnz_k

        return y, col_per_cluster, nnz_per_cluster

# fc: [1000, 2048] => [out_dim, in_dim]
D = fc          # [1000, 2048]
R = num_row     # 1000
C = num_idx     # 2048
K = k           # 32

model = Clustering(D, K)
#criterion = nn.MSELoss(reduction='sum')
criterion = nn.MSELoss()
target_col = torch.zeros((K)).cuda()
target_col.data.fill_(C/K)
target_nnz = torch.zeros((K)).cuda()
target_nnz.data.fill_(nnz_per_cluster - 200)

minimum = 77777
optimizer = torch.optim.SGD(model.parameters(), lr=100, momentum=0, weight_decay=0.0015)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
model.cuda()
model.train()

for i in range(4000):
    y, col_per_cluster, nnz_per_cluster = model()
    loss = criterion(nnz_per_cluster + col_per_cluster, target_nnz + target_col)
    loss1 = criterion(nnz_per_cluster, target_nnz)
    loss2 = criterion(col_per_cluster, target_col)
    print(i, ":\t", loss.item(), "\t", loss1.item(), "\t", loss2.item(),
             "\t{:.1f}\t{:.1f}".format(max(nnz_per_cluster).item(), max(col_per_cluster).item()))
    if (max(col_per_cluster + nnz_per_cluster) < minimum):
        minimum = max(col_per_cluster + nnz_per_cluster)
        y_ = y
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

model.eval()

nnz_k = torch.zeros(K)
col_k = torch.zeros((K, C))
row_k = torch.zeros(K)

y_ = torch.argmax(y_, dim=1)
for row_i in range(num_row):
    k_i = y_[row_i]
    nnz_k[k_i] = nnz_k[k_i] + nnz[row_i]
    col_k[k_i] = torch.logical_or(col_k[k_i], D[row_i])
    row_k[k_i] = row_k[k_i] + 1

col_k = torch.count_nonzero(col_k, dim=1) 
tot_k = nnz_k + col_k + row_k
print_("nnz_k", nnz_k)
print_("col_k", col_k)
print_("row_k", row_k)
print_("tot_k", tot_k)
print("max : ", max(tot_k))


