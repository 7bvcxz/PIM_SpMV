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
nnz_per_cluster_l = int(nnz_per_cluster * 0.99)
nnz_per_cluster_h = int(nnz_per_cluster * 1.01)

cluster = torch.zeros((k, num_row)) - 1
centroid = torch.zeros((k, num_idx)) - 1
score = torch.zeros((num_row, num_row)) - 1
y = torch.zeros((k, num_row))
fc = model.fc.weight
print(fc.reshape(-1).shape)
print("< Configure Ended! >\n")

print("< Defining Functions... >")
def prune_layer(layer, sparsity):
    layer_ = layer.abs().reshape(-1).sort()
    threshold = layer_.values[int(num_idx*num_row*sparsity)-1].item()
    return (layer.abs() > threshold)*1.0

def simil(row0, row1):
    union = torch.count_nonzero(torch.logical_or(row0, row1)*1)
    intersection = torch.count_nonzero(torch.logical_and(row0, row1)*1)
    return intersection / union

def calculate_score(layer):
    for row0 in range(num_row):
        for row1 in range(num_row):
            if row0 < row1:
                tmp = simil(layer[row0], layer[row1])
                score[row0][row1] = tmp
                score[row1][row0] = tmp

def init_centroid():
    tmp = random.sample(range(num_row), 32)
    for k_idx in range(k):
        centroid[k_idx] = fc[tmp[k_idx]]
    return centroid
 
class GumbelSoftmax(torch.nn.Module):
    def __init__(self, hard=False):
        super(GumbelSoftmax, self).__init__()
        self.hard = hard
        self.gpu = False
        
    def cuda(self):
        self.gpu = True
    
    def cpu(self):
        self.gpu = False
        
    def sample_gumbel(self, shape, eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(shape)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        if self.gpu:
            return Variable(noise).cuda()
        else:
            return Variable(noise)

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumbel_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumbel_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dim = -1
        if self.training:
            gumbel_samples_tensor = self.sample_gumbel_like(logits.data)
            gumbel_trick_log_prob_samples = logits + Variable(gumbel_samples_tensor)
        else:
            gumbel_trick_log_prob_samples = logits
        soft_samples = F.softmax(gumbel_trick_log_prob_samples / temperature, dim)
        return soft_samples
    
    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            _, max_value_indexes = y.data.max(1, keepdim=True)
            y_hard = logits.data.clone().zero_().scatter_(1, max_value_indexes, 1)
            y = Variable(y_hard - y.data) + y
        return y
        
    def forward(self, logits, temp=1, force_hard=False):
        samplesize = logits.size()

        if self.training and not force_hard:
            return self.gumbel_softmax(logits, temperature=1, hard=False)
        else:
            return self.gumbel_softmax(logits, temperature=1, hard=True) 

class Clustering(nn.Module):
    def __init__(self, D, K):
        super(Clustering, self).__init__()
        self.D = D.cuda()  # [R, C]
        self.R = D.shape[0]
        self.C = D.shape[1]
        self.K = K
        self.Y = nn.Parameter(torch.zeros((R, K)), requires_grad=True)
        self.gs = GumbelSoftmax()

        torch.nn.init.kaiming_uniform_(self.Y)

    def forward(self):
        #print("self.Y : \n", self.Y)
        y = self.gs(self.Y, force_hard=True)   # [R, K]
        #y = y.view(R, 1, K) * self.D.view(R, C, 1)  # [R, C, K]
        y = y.view(R, 1, K) * self.D.view(R, C, 1)  # [R, C, K]
        reduced_y = 1. - torch.prod(1 - y, 0)   # [C, K]
        return torch.sum(reduced_y, dim=0), torch.sum(y, dim=[0, 1])  # [K], [K]
   
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

# fc: [1000, 2048] => [out_dim, in_dim]
D = fc
R, C = D.shape
K = 32

model = Clustering(D, K)
#criterion = nn.L1Loss()
criterion = nn.MSELoss()
target_col = torch.zeros((K)).cuda()
target_col.data.fill_(C/K)
target_calc = torch.zeros((K)).cuda()
#target_calc.data.fill_(R*C*0.25/K)
target_calc.data.fill_(16000)

minimum = 30000
lr = 30
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
model.cuda()

model.train()
for i in range(10000):
    col_per_cluster, calc_per_cluster = model()
    #print("col_per_cluster : \n", col_per_cluster)
    #print("calc_per_cluster : \n", calc_per_cluster)
    target_calc.data.fill_(min(col_per_cluster + calc_per_cluster)-100)
    loss = criterion(col_per_cluster + calc_per_cluster, target_calc)
    #loss = criterion(col_per_cluster, target_col) + criterion(calc_per_cluster, target_calc)
    #loss = criterion(col_per_cluster, target_col) * criterion(calc_per_cluster, target_calc)
    #loss = criterion(col_per_cluster + calc_per_cluster, target_col + target_calc)
    #loss = criterion(calc_per_cluster, target_calc)
    if (max(col_per_cluster + calc_per_cluster).item() < minimum):
        minimum = max(col_per_cluster + calc_per_cluster).item()
    #print("iter : ", i, "\tloss : ", criterion(col_per_cluster, target_col).item(), "\t", criterion(calc_per_cluster, target_col).item())
    print("iter : ", i, "\tloss : ", criterion(col_per_cluster, target_col).item() + criterion(calc_per_cluster, target_col).item())
    #print(optimizer.param_groups[0]['lr'], loss, torch.std(calc_per_cluster))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

model.eval()
col_per_cluster, calc_per_cluster = model()
print("col_per_cluster : \n", col_per_cluster)
print("calc_per_cluster : \n", calc_per_cluster)
print("elements_per_cluster : \n", col_per_cluster + calc_per_cluster)

print(minimum)
