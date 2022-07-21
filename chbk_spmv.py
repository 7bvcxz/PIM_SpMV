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

K = 32
CH = 32
BA = 1
C = 2048
R = 1000
sparsity = 0.99

nnz_per_cluster = int(C * R * (1-sparsity) / K)

y = torch.zeros((K, BA, CH))
fc = model.fc.weight
print("< Configure Ended! >\n")

print("< Defining Functions... >")
def prune_layer(layer, sparsity):
    layer_ = layer.abs().reshape(-1).sort()
    threshold = layer_.values[int(C*R*sparsity)-1].item()
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
print("< Start!! >")
print("##################################################################")
print("##################################################################")

class Clustering(nn.Module):
    def __init__(self, D, K, CH, BA):
        super(Clustering, self).__init__()
        self.D = D.cuda()   # [R, C]
        self.R = D.shape[0]
        self.C = D.shape[1]
        self.K = K
        self.CH = CH
        self.BA = BA
        self.Y = nn.Parameter(torch.zeros((R, CH*BA)), requires_grad=True)
        self.sm = nn.Softmax(dim=1)

        torch.nn.init.kaiming_uniform_(self.Y)

    def forward(self):
        y = self.sm(self.Y).reshape(R, CH, BA)  # [R, CH, BA]
        x = y.view(R, 1, CH, BA) * self.D.view(R, C, 1, 1) # [R, C, CH, BA]

        nnz_ch_ba = torch.sum(x, dim=[0, 1])  # [CH, BA]
        max_nnz_ch = torch.max(nnz_ch_ba, dim=1).values

        col_ch_ba = 1 - torch.prod(1-x, 0)
        col_ch = 1 - torch.prod(1 - col_ch_ba, 2)  # [C, CH]
        num_col_ch = torch.sum(col_ch, dim=0)
        row_ch = torch.sum(y, dim=[0, 2])
        
        max_cmd = torch.max(max_nnz_ch + num_col_ch + row_ch)

        return y, max_cmd, max_nnz_ch, num_col_ch, row_ch

# fc: [1000, 2048] => [out_dim, in_dim]
D = fc  # [1000, 2048]

model = Clustering(D, K, CH, BA)
#criterion = nn.MSELoss(reduction='sum')
criterion = nn.MSELoss()
target = torch.zeros((1)).cuda().data.fill_(1000)

best_case = 77777
optimizer = torch.optim.SGD(model.parameters(), lr=100, momentum=0., weight_decay=0.0015)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 4000, 0.)

model.cuda()
model.train()

for i in range(4000):
    y, max_cmd, max_nnz_ch, col_ch, row_ch = model()
    loss = criterion(max_cmd, target)
    #loss = criterion(max_cmd, target) + torch.mean(col_ch)*10
    #loss = criterion(max_nnz_ch + col_ch + row_ch, torch.zeros((CH)).cuda().data.fill_(C/CH + nnz_per_cluster - 200)) 
    #loss = criterion(max_nnz_ch + col_ch + row_ch, torch.zeros((CH)).cuda().data.fill_(0)) 
    loss = torch.pow(max_nnz_ch + col_ch, 2).sum()
    print(i, ":\t{:.2f}\t{:.2f}".format(max_cmd.item(), torch.max(col_ch).item()))
    if (max_cmd.item() < best_case and i > 100):
        best_case = max_cmd.item()
        y_ = y
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

model.eval()
y_ = y_.cpu()
D = D.cpu()

# y_ [R, CH, BA] (max percentage to 1)
y_ = y_.reshape(R, CH*BA)
max_idx_ = torch.argmax(y_, dim=1)
y_.fill_(0.)
y_[torch.arange(R), max_idx_] = 1.
y_ = y_.reshape(R, CH, BA)
x = y_.view(R, 1, CH, BA) * D.view(R, C, 1, 1) # [R, C, CH, BA]

nnz_ch_ba = torch.sum(x, dim=[0, 1])  # [CH, BA]
nnz_ch = torch.sum(nnz_ch_ba, dim=1)  # [CH]
max_nnz_ch = torch.max(nnz_ch_ba, dim=1).values  # [CH]

col_ch_ba = 1 - torch.prod(1-x, 0)  # [C, CH, BA]
col_ch = 1 - torch.prod(1-col_ch_ba, 2)  # [C, CH]

num_col_ch_ba = torch.sum(col_ch_ba, dim=0)  # [CH, BA]
num_col_ch = torch.sum(col_ch, dim=0)  # [CH]

row_ch_ba = torch.sum(y_, dim=0)  # [CH, BA]
row_ch = torch.sum(y_, dim=[0, 2])  # [CH]

nnz_ch_ba = nnz_ch_ba.reshape(-1)
num_col_ch_ba = num_col_ch_ba.reshape(-1)
row_ch_ba = row_ch_ba.reshape(-1)

print("nnz_ch_ba / num_col_ch_ba / row_ch_ba")

print_("nnz_ch_ba", nnz_ch_ba)
print_("num_col_ch_ba", num_col_ch_ba)
print_("row_ch_ba", row_ch_ba)
print("")
print("max_nnz_ch / num_col_ch / row_ch")
print_("max_nnz_ch", max_nnz_ch)
#print_("nnz_ch", nnz_ch)
print_("col_ch", num_col_ch)
print_("row_ch", row_ch)
print("max cmd : ", torch.max(max_nnz_ch + num_col_ch + row_ch).item())

