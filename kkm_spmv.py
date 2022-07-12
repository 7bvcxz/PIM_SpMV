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

sparsity = 0.75
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
        y = y.view(R, 1, K) * self.D.view(R, C, 1)
        #print(y)
        return torch.sum(y, dim=[0, 1])


# fc: [1000, 2048] => [out_dim, in_dim]
D = fc          # [1000, 2048]
R = num_row     # 1000
C = num_idx     # 2048
K = k           # 32

model = Clustering(D, K)
criterion = nn.MSELoss()
#criterion = nn.L1Loss()
target_col = torch.zeros((K)).cuda()
target_col.data.fill_(C/K)
target_calc = torch.zeros((K)).cuda()
#target_calc.data.fill_(R*C*0.25/K)
#target_calc.data.fill_(16000)
target_calc.data.fill_(16000)

optimizer = torch.optim.SGD(model.parameters(), lr=30, momentum=0, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
model.cuda()
model.train()

for i in range(1000):
    #col_per_cluster, calc_per_cluster = model()
    calc_per_cluster = model()
    #target_calc.data.fill_(min(calc_per_cluster)-100)
    loss = criterion(calc_per_cluster, target_calc)
    print(i, ":\t", loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

model.eval()
calc_per_cluster = model()
calc_per_cluster = calc_per_cluster.int()
print(calc_per_cluster)
print(max(calc_per_cluster))













