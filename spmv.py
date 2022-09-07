import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cvxpy as cvp
import numpy as np
import random
import time

num_row = 0
num_col = 0
num_ch = 0
num_ba = 0
part_col = 0
part_row = 0
sparsity = 0
k = 0
nnz = 0
bound_ratio = 0
nnz_per_cluster = 0
nnz_per_cluster_h = 0
nnz_per_cluster_l = 0
device = 0

def init(layer_, sparsity_, num_ch_, num_ba_, part_col_, bound_ratio_, device_):
    global num_row
    global num_col
    global num_ch
    global num_ba
    global part_col
    global part_row
    global k
    global sparsity
    global bound_ratio
    global nnz
    global nnz_per_cluster
    global nnz_per_cluster_h
    global nnz_per_cluster_l
    global cuda

    num_row = layer_.shape[0]
    num_col = layer_.shape[1]
    num_ch = num_ch_
    num_ba = num_ba_
    part_col = part_col_
    part_row = int(num_ch / part_col)
    k = num_ch * num_ba
    sparsity = sparsity_
    bound_ratio = bound_ratio_
    #nnz = torch.count_nonzero(layer_, dim=1) * 1.0
    nnz_per_cluster = int(num_row * num_col * (1-sparsity) / k)
    nnz_per_cluster_h = int(nnz_per_cluster * (1+bound_ratio))
    nnz_per_cluster_l = int(nnz_per_cluster * (1-bound_ratio))
    device = device_

def print_(name, data):
    for i in data.cpu().detach().numpy():
        print(i, end=" ")
    print()

def prune_layer(layer, sparsity):
    layer_ = layer.abs().reshape(-1).sort()
    threshold = layer_.values[int(num_col*num_row*sparsity)-1].item()
    return (layer.abs() > threshold)*1.0

def print_specific(P, args):
    CH = int(args.num_ch)
    BA = int(args.num_ba)
    D = args.layer
    R = D.shape[0]
    C = D.shape[1]
    register_size = int(args.register_size)
    num_part = (C-1)//register_size + 1
 
    zeros = torch.zeros(R, num_part * register_size - C)
    D_ = torch.cat([D, zeros], dim=1).view(R, num_part, -1) # [R, num_part, reg_size]

    nnz_ch_ba = torch.zeros((num_part, CH, BA))
    
    col_ch = torch.zeros(CH, C)
    num_col_ch = torch.zeros(CH)
    num_row_ch = torch.zeros(CH)
   
    if args.print_option == 0:  # With Register_Size
        print("--Printing Performance (with reg_size)...--")
        # P [R, CH, BA] (max percentage to 1)
        P = P.reshape(R, CH*BA)
        max_idx_ = torch.argmax(P, dim=1)
        P.fill_(0.)
        P[torch.arange(R), max_idx_] = 1.
        P = P.reshape(R, CH, BA)
        x = P.view(R, 1, 1, CH, BA) * D_.view(R, num_part, register_size, 1, 1) # [R, np, rs, CH, BA]

        nnz_ch_ba = torch.sum(x, dim=[0, 2])  # [np, CH, BA]
        max_nnz_ch = torch.max(nnz_ch_ba, dim=2).values  # [np, CH]
        tot_max_nnz_ch = max_nnz_ch.sum(dim=0)  # [CH]

        num_col_ch = torch.zeros(CH).fill_(C)  # Set to all column for now
        num_row_ch = torch.sum(P, dim=[0, 2])

        print("nnz_ch / col_ch / row_ch")
        print_("", tot_max_nnz_ch)
        print_("", num_col_ch)
        print_("", num_row_ch)

        cost_ch = tot_max_nnz_ch + num_col_ch + num_row_ch
        print_("", cost_ch)
        print("max : ", torch.max(cost_ch).item())
        
        return torch.max(cost_ch).item()

    elif args.print_option == 1:  # Without Register_size
        print("--Printing Performance (without reg_size)...--")
        # P [R, CH, BA]
        # Blank for now
        print("hi 1")

    elif args.print_option == 2:  # With Register_size, Print [np, CH, BA]
        print("--Printing nnz of [np, CH, BA]...--")
        # P [R, CH, BA]
        x = P.view(R, 1, 1, CH, BA) * D_.view(R, num_part, register_size, 1, 1) # [R, np, rs, CH, BA]

        nnz_ch_ba = x.sum(dim=[0, 2])  # [np, CH, BA]
        print("CH >> np >> BA")
        for ch_i in range(CH):
            print("Channel", ch_i)
            for np_i in range(num_part):
                print("np", np_i, ":", end="\t")
                for ba_i in range(BA):
                    print("{:.1f}".format(nnz_ch_ba[np_i][ch_i][ba_i].item()), end=" ")
                print()

    elif args.print_option == 3:  # With Register_size, Print [np, CH] score
        print("--Printing [np, CH] L/B Score...--")
        # P [R, CH, BA]
        x = P.view(R, 1, 1, CH, BA) * D_.view(R, num_part, register_size, 1, 1) # [R, np, rs, CH, BA]
        nnz_ch_ba = x.sum(dim=[0, 2])  # [np, CH, BA]

        max_ch = torch.max(nnz_ch_ba, dim=2).values  # [np, CH]
        mean_nnz_ch = nnz_ch_ba.sum(dim=2) / BA  # [np, CH]

        lb_score_ch = max_ch / mean_nnz_ch  # [np, CH]

        for np_i in range(num_part):
            for ch_i in range(CH):
                print("{:.2f}".format(lb_score_ch[np_i][ch_i].item()), end=" ")
            print()
    else:
        print("Something gone wrong")
    return 0 


def hard_split(layer, part_col):
    layer_ = torch.clone(layer)
    layer_ = layer_.view(num_row, part_col, num_col//part_col)
    layer_ = torch.transpose(layer_, 0, 1)
    return layer_
 
def soft_split(layer, part_col):
    num_col_part = num_col // part_col
    layer_ = torch.clone(layer)
    layer_ = layer_.view(num_row, part_col, num_col_part)
    layer_ = torch.transpose(layer_, 0, 1)
    layer__ = torch.zeros((part_col, num_row, num_col))
    for part_i in range(part_col):
        left_col = part_i * num_col_part
        left_layer = torch.zeros((num_row, left_col))
        right_col = num_col - (part_i + 1) * num_col_part
        right_layer = torch.zeros((num_row, right_col))
        layer__[part_i] = torch.cat((left_layer, layer_[part_i], right_layer), 1)
    
    return layer__

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

def sequence_row_equal(layer, num_ch_):
    num_row_ = layer.shape[0]
    num_col_ = layer.shape[1]
    nnz_ch_ba = torch.zeros(num_ch_ * num_ba)
    nnz_ch = torch.zeros(num_ch_)
    col_ch = torch.zeros(num_ch_, num_col_)
    row_ch = torch.zeros(num_ch_)
    nnz_ = torch.count_nonzero(layer, dim=1) * 1.0
    
    ba_i = 0
    for row_i in range(num_row_):
        if row_i > int(num_row_ * (ba_i+1) / (num_ch_ * num_ba)):
            ba_i = ba_i + 1
        nnz_ch_ba[ba_i] = nnz_ch_ba[ba_i] + nnz_[row_i] 
        col_ch[ba_i//num_ba] = col_ch[ba_i//num_ba] + layer[row_i]
        row_ch[ba_i//num_ba] = row_ch[ba_i//num_ba] + 1

    for ch_i in range(num_ch_):
        offset = num_ba * ch_i
        nnz_ch[ch_i] = torch.max(nnz_ch_ba[offset:offset+num_ba]).item()

    col_ch = torch.count_nonzero(col_ch, dim=1)

    return nnz_ch, col_ch, row_ch

def sequence_row_threshold(layer, num_ch_):
    num_row_ = layer.shape[0]
    num_col_ = layer.shape[1]
    nnz_ch_ba = torch.zeros(part_row * num_ba)
    nnz_ch = torch.zeros(part_row)
    col_ch = torch.zeros(part_row, num_col_)
    row_ch = torch.zeros(part_row)
    nnz_ = torch.count_nonzero(layer, dim=1) * 1.0
    
    ba_i = 0
    for row_i in range(num_row_):
        if nnz_ch_ba[ba_i] + nnz_[row_i] > nnz_per_cluster_h:
            ba_i = ba_i + 1
        nnz_ch_ba[ba_i] = nnz_ch_ba[ba_i] + nnz_[row_i] 
        col_ch[ba_i//num_ba] = col_ch[ba_i//num_ba] + layer[row_i]
        row_ch[ba_i//num_ba] = row_ch[ba_i//num_ba] + 1

    for ch_i in range(part_row):
        offset = num_ba * ch_i
        nnz_ch[ch_i] = torch.max(nnz_ch_ba[offset:offset+num_ba]).item()

    col_ch = torch.count_nonzero(col_ch, dim=1)

    return nnz_ch, col_ch, row_ch

def sequence_nnz_threshold(layer, num_ch_):
    ba_i = 0
    nnz_ch_ba = torch.zeros(part_row * num_ba)
    nnz_ch = torch.zeros(part_row)
    col_ch = torch.zeros(part_row, int(num_col/part_col))
    row_ch = torch.zeros(part_row)
    nnz_per_cluster_ = torch.ceil(torch.count_nonzero(layer) / (part_row*num_ba)).item()
    nnz_ = torch.count_nonzero(layer, dim=1) * 1.0
    for row_i in range(num_row):
        print(nnz_ch_ba[ba_i] + nnz_[row_i], nnz_per_cluster_)
        if nnz_ch_ba[ba_i] + nnz_[row_i] > nnz_per_cluster_:
            front_nnz = nnz_per_cluster_ - nnz_ch_ba[ba_i]
            nnz_ch_ba[ba_i] = nnz_per_cluster_
            col_ch[ba_i//num_ba] = col_ch[ba_i//num_ba] + layer[row_i]
            row_ch[ba_i//num_ba] = row_ch[ba_i//num_ba] + 1
            ba_i = ba_i + 1
            nnz_ch_ba[ba_i] = nnz_ch_ba[ba_i] + (nnz_[row_i] - front_nnz)
            col_ch[ba_i//num_ba] = col_ch[ba_i//num_ba] + layer[row_i]
            row_ch[ba_i//num_ba] = row_ch[ba_i//num_ba] + 1
        else:
            nnz_ch_ba[ba_i] = nnz_ch_ba[ba_i] + nnz_[row_i]
            col_ch[ba_i//num_ba] = col_ch[ba_i//num_ba] + layer[row_i]
            row_ch[ba_i//num_ba] = row_ch[ba_i//num_ba] + 1
 
    for ch_i in range(part_row):
        offset = num_ba * ch_i
        nnz_ch[ch_i] = torch.max(nnz_ch_ba[offset:offset+num_ba]).item()

    col_ch = torch.count_nonzero(col_ch, dim=1)

    return nnz_ch, col_ch, row_ch

def sequence_row_threshold_register_size(layer, args):
    device = args.device
    CH = int(args.num_ch)
    BA = int(args.num_ba)
    register_size = int(args.register_size)
    R = layer.shape[0]
    C = layer.shape[1]
    nnz_M = torch.count_nonzero(layer, dim=1) * 1.0
    num_part = (C - 1) // register_size + 1

    nnz_ch_ba = torch.zeros(CH*BA)
    nnz_ch = torch.zeros(CH, num_part)
    col_ch = torch.zeros(CH, C)
    row_ch = torch.zeros(CH)
    P = torch.zeros((R, CH, BA))
    
    accum_ch_ba = torch.zeros(CH * BA, num_col)
    nnz_ch_ba_part = torch.zeros(CH * BA, num_part)
    
    ba_i = 0
    for row_i in range(R):
        if nnz_ch_ba[ba_i] + nnz_M[row_i] > nnz_per_cluster_h:
            ba_i = ba_i + 1
        nnz_ch_ba[ba_i] = nnz_ch_ba[ba_i] + nnz_M[row_i]
        col_ch[ba_i//BA] = col_ch[ba_i//BA] + layer[row_i]
        if nnz_M[row_i] != 0:
            row_ch[ba_i//BA] = row_ch[ba_i//BA] + 1
        accum_ch_ba[ba_i] = accum_ch_ba[ba_i] + layer[row_i]
        P[row_i][ba_i//BA][ba_i%BA] = 1

    for ba_i in range(CH*BA):
        e_idx = 0
        for part_i in range(num_part):
            s_idx = e_idx
            e_idx = e_idx + register_size
            if e_idx >= C:
                e_idx = C

            nnz_ch_ba_part[ba_i][part_i] = accum_ch_ba[ba_i][s_idx:e_idx].sum()  # [CH*BA, num_part]

    nnz_ch_ba_part = nnz_ch_ba_part.view(CH, BA, num_part)  # [CH, BA, num_part]
    max_nnz_ch = torch.max(nnz_ch_ba_part, dim=1).values.sum(dim=1)  # [CH]

    col_ch = torch.count_nonzero(col_ch, dim=1)

    return P, max_nnz_ch, col_ch, row_ch

def space_a(layer, num_ch_):
    ### >> for bank << ###
    nnz_ch_ba = torch.zeros(part_row * num_ba)
    col_ch_ba = torch.zeros(part_row * num_ba, int(num_col/part_col))
    row_ch_ba = torch.zeros(part_row * num_ba)
    nnz_ch = torch.zeros(part_row)
    col_ch = torch.zeros(part_row, int(num_col/part_col))
    row_ch = torch.zeros(part_row)
    nnz_per_ba = torch.ceil(torch.count_nonzero(layer) / (part_row*num_ba)).item()
    nnz_per_ch = torch.ceil(torch.count_nonzero(layer) / part_row).item()
    nnz_ = torch.count_nonzero(layer, dim=1) * 1.0
   
    score = torch.zeros(part_row*num_ba)
    for row_i in range(num_row):
        print("iter : ", row_i+1, end='\t')
        for ba_i in range(part_row*num_ba):
            if nnz_ch_ba[ba_i] + nnz_[row_i] > nnz_per_ba:
                score[ba_i] = -1 * (nnz_ch_ba[ba_i] + nnz_[row_i] - nnz_per_ba)
            else:
                overlap = torch.count_nonzero(torch.logical_and(col_ch_ba[ba_i], layer[row_i])*1).item()
                score[ba_i] = max(overlap / nnz_[row_i], 1 / (nnz_ch_ba[ba_i] + nnz_[row_i]))
        ba_i = torch.argmax(score)
    
        print("row ", row_i, "   \t→ cluster ", ba_i.item())
        nnz_ch_ba[ba_i] = nnz_ch_ba[ba_i] + nnz_[row_i]
        col_ch_ba[ba_i] = torch.logical_or(col_ch_ba[ba_i], layer[row_i])*1
        row_ch_ba[ba_i] = row_ch_ba[ba_i] + 1 
    
    score = torch.zeros(part_row)
    for ba_i in range(part_row*num_ba):
        print("iter : ", ba_i+1, end='\t')
        for ch_i in range(part_row):
            if nnz_ch[ch_i] + nnz_ch_ba[ba_i] > nnz_per_ch:
                score[ch_i] = -1 * (nnz_ch[ch_i] + nnz_ch_ba[ba_i] - nnz_per_ch)
            else:
                overlap = torch.count_nonzero(torch.logical_and(col_ch, col_ch_ba[ba_i])*1).item()
                score[ch_i] = max(overlap / nnz_ch_ba[ba_i], 1 / (nnz_ch[ch_i] + nnz_ch_ba[ba_i]))
        ch_i = torch.argmax(score)
    
        print("ba ", ba_i, "   \t→ ch ", ch_i.item())
        nnz_ch[ch_i] = nnz_ch[ch_i] + nnz_ch_ba[ba_i]
        col_ch[ch_i] = torch.logical_or(col_ch[ch_i], col_ch_ba[ba_i])*1
        row_ch[ch_i] = row_ch[ch_i] + row_ch_ba[ba_i]

    nnz_ch = nnz_ch // num_ba
    col_ch = torch.count_nonzero(col_ch, dim=1)

    return nnz_ch, col_ch, row_ch

def space_a_register_size(layer, args):
    device = args.device
    num_ch = int(args.num_ch)
    num_ba = int(args.num_ba)
    register_size = int(args.register_size)
 
    ### >> for bank << ###
    num_part = (num_col - 1) // register_size + 1

    nnz_ch_ba = torch.zeros(num_ch * num_ba)
    col_ch_ba = torch.zeros(num_ch * num_ba, num_col)
    row_ch_ba = torch.zeros(num_ch * num_ba)
    accum_ch_ba = torch.zeros(num_ch * num_ba, num_col)

    tot_nnz = torch.count_nonzero(layer)
    nnz_M = torch.count_nonzero(layer, dim=1) * 1.0
    nnz_per_ba = torch.ceil(tot_nnz / (num_ch*num_ba)).item()
    nnz_per_ch = torch.ceil(tot_nnz / num_ch).item()

    P = torch.zeros((num_row, num_ch, num_ba))
    P_ba = torch.zeros((num_row, num_ch*num_ba))
    ba_idx = torch.zeros(num_ch).int()
    
    score_ba = torch.zeros(num_ch*num_ba)
    for row_i in range(num_row):
        print("iter : ", row_i+1, end='\t')
        for ba_i in range(num_ch*num_ba):
            if (nnz_ch_ba[ba_i] + nnz_M[row_i]) > nnz_per_ba:
                score_ba[ba_i] = -1 * (nnz_ch_ba[ba_i] + nnz_M[row_i] - nnz_per_ba)
            else:
                overlap = torch.count_nonzero(torch.logical_and(col_ch_ba[ba_i], layer[row_i])*1).item()
                score_ba[ba_i] = max(overlap / nnz_M[row_i], 1 / (nnz_ch_ba[ba_i] + nnz_M[row_i]))
        ba_i = torch.argmax(score_ba)
    
        print("row ", row_i, "   \t→ cluster ", ba_i.item())
        nnz_ch_ba[ba_i] = nnz_ch_ba[ba_i] + nnz_M[row_i]
        col_ch_ba[ba_i] = torch.logical_or(col_ch_ba[ba_i], layer[row_i])*1
        if nnz_M[row_i] != 0:
            row_ch_ba[ba_i] = row_ch_ba[ba_i] + 1 
        accum_ch_ba[ba_i] = accum_ch_ba[ba_i] + layer[row_i]
        P_ba[row_i][ba_i] = 1

    nnz_ch_ba_part = torch.zeros(num_ch*num_ba, num_part)
    for ba_i in range(num_ch*num_ba):
        e_idx = 0
        for part_i in range(num_part):
            s_idx = e_idx
            e_idx = e_idx + register_size
            if e_idx >= num_col:
                e_idx = num_col

            nnz_ch_ba_part[ba_i][part_i] = accum_ch_ba[ba_i][s_idx:e_idx].sum()

    nnz_ch = torch.zeros(num_ch)
    col_ch = torch.zeros(num_ch, num_col)
    row_ch = torch.zeros(num_ch)
    max_nnz_ch = torch.zeros(num_ch, num_part)
 
    score_ch = torch.zeros(num_ch)
    for ba_i in range(num_ch*num_ba):
        print("iter : ", ba_i+1, end='\t')
        for ch_i in range(part_row):
            if ba_idx[ch_i] == 16:  # This Channel is Full!
                score_ch[ch_i] = -777777777
            elif nnz_ch[ch_i] + nnz_ch_ba[ba_i] > nnz_per_ch:
                score_ch[ch_i] = -1 * (nnz_ch[ch_i] + nnz_ch_ba[ba_i] - nnz_per_ch)
            else:
                overlap = torch.count_nonzero(torch.logical_and(col_ch, col_ch_ba[ba_i])*1).item()
                score_ch[ch_i] = max(overlap / nnz_ch_ba[ba_i], 1 / (nnz_ch[ch_i] + nnz_ch_ba[ba_i]))
        ch_i = torch.argmax(score_ch)
    
        print("ba ", ba_i, "   \t→ ch ", ch_i.item())
        nnz_ch[ch_i] = nnz_ch[ch_i] + nnz_ch_ba[ba_i]
        col_ch[ch_i] = torch.logical_or(col_ch[ch_i], col_ch_ba[ba_i])*1
        row_ch[ch_i] = row_ch[ch_i] + row_ch_ba[ba_i]
        for row_i in range(num_row):
            P[row_i][ch_i][ba_idx[ch_i]] = P_ba[row_i][ba_i]
        ba_idx[ch_i] = ba_idx[ch_i] + 1

        for part_i in range(num_part):
            max_nnz_ch[ch_i][part_i] = max(max_nnz_ch[ch_i][part_i], + nnz_ch_ba_part[ba_i][part_i])

    max_nnz_ch = max_nnz_ch.sum(dim=1)  # [CH]
    col_ch = torch.count_nonzero(col_ch, dim=1)

    return P, max_nnz_ch, col_ch, row_ch

def space_a_register_size_withrow(layer, args):
    device = args.device
    num_ch = int(args.num_ch)
    num_ba = int(args.num_ba)
    register_size = int(args.register_size)

    ### >> for bank << ###
    num_part = (num_col - 1) // register_size + 1

    nnz_ch_ba = torch.zeros(num_ch * num_ba)
    col_ch_ba = torch.zeros(num_ch * num_ba, num_col)
    row_ch_ba = torch.zeros(num_ch * num_ba)
    accum_ch_ba = torch.zeros(num_ch * num_ba, num_col)

    tot_nnz = torch.count_nonzero(layer)
    nnz_M = torch.count_nonzero(layer, dim=1) * 1.0
    thr_per_ba = torch.ceil((tot_nnz + num_row) / (num_ch*num_ba)).item()
    thr_per_ch = torch.ceil(tot_nnz / (num_ch*num_ba) + num_row / num_ch).item()

    P = torch.zeros((num_row, num_ch, num_ba))
    P_ba = torch.zeros((num_row, num_ch*num_ba))
    ba_idx = torch.zeros(num_ch).int()

    print("threshold = ", thr_per_ch)
    
    score_ba = torch.zeros(num_ch*num_ba)
    for row_i in range(num_row):
        print("iter : ", row_i+1, end='\t')
        for ba_i in range(num_ch*num_ba):
            if (nnz_ch_ba[ba_i] + row_ch_ba[ba_i] + nnz_M[row_i] + 1) > thr_per_ba:
                score_ba[ba_i] = -1 * (nnz_ch_ba[ba_i] + row_ch_ba[ba_i] + nnz_M[row_i] - thr_per_ba)
            else:
                overlap = torch.count_nonzero(torch.logical_and(col_ch_ba[ba_i], layer[row_i])*1).item()
                score_ba[ba_i] = max(overlap / nnz_M[row_i], 1 / (nnz_ch_ba[ba_i] + row_ch_ba[ba_i] + nnz_M[row_i]))
        ba_i = torch.argmax(score_ba)
    
        print("row ", row_i, "   \t→ cluster ", ba_i.item())
        nnz_ch_ba[ba_i] = nnz_ch_ba[ba_i] + nnz_M[row_i]
        col_ch_ba[ba_i] = torch.logical_or(col_ch_ba[ba_i], layer[row_i])*1
        if nnz_M[row_i] != 0:
            row_ch_ba[ba_i] = row_ch_ba[ba_i] + 1 
        accum_ch_ba[ba_i] = accum_ch_ba[ba_i] + layer[row_i]
        P_ba[row_i][ba_i] = 1

    nnz_ch_ba_part = torch.zeros(num_ch*num_ba, num_part)
    for ba_i in range(num_ch*num_ba):
        e_idx = 0
        for part_i in range(num_part):
            s_idx = e_idx
            e_idx = e_idx + register_size
            if e_idx >= num_col:
                e_idx = num_col

            nnz_ch_ba_part[ba_i][part_i] = accum_ch_ba[ba_i][s_idx:e_idx].sum()

    nnz_ch = torch.zeros(num_ch)
    col_ch = torch.zeros(num_ch, num_col)
    row_ch = torch.zeros(num_ch)
    max_nnz_ch = torch.zeros(num_ch, num_part)
 
    score_ch = torch.zeros(num_ch)
    for ba_i in range(num_ch*num_ba):
        print("iter : ", ba_i+1, end='\t')
        for ch_i in range(num_ch):
            if ba_idx[ch_i] == 16: # This Channel is Full!
                score_ch[ch_i] = -7777777777
            elif (nnz_ch[ch_i] + row_ch[ch_i] + nnz_ch_ba[ba_i] + row_ch_ba[ba_i]) > thr_per_ch:
                score_ch[ch_i] = -1 * (nnz_ch[ch_i] + row_ch[ch_i] + nnz_ch_ba[ba_i] + row_ch_ba[ba_i] - thr_per_ch)
            else:
                overlap = torch.count_nonzero(torch.logical_and(col_ch, col_ch_ba[ba_i])*1).item()
                score_ch[ch_i] = max(overlap / nnz_ch_ba[ba_i], 1 / (nnz_ch[ch_i] + row_ch[ch_i] + nnz_ch_ba[ba_i]))
        ch_i = torch.argmax(score_ch)
    
        print("ba ", ba_i, "   \t→ ch ", ch_i.item())
 
        nnz_ch[ch_i] = nnz_ch[ch_i] + nnz_ch_ba[ba_i]
        col_ch[ch_i] = torch.logical_or(col_ch[ch_i], col_ch_ba[ba_i])*1
        row_ch[ch_i] = row_ch[ch_i] + row_ch_ba[ba_i]
        for row_i in range(num_row):
            P[row_i][ch_i][ba_idx[ch_i]] = P_ba[row_i][ba_i]
        ba_idx[ch_i] = ba_idx[ch_i] + 1
       
        for part_i in range(num_part):
            max_nnz_ch[ch_i][part_i] = max(max_nnz_ch[ch_i][part_i], nnz_ch_ba_part[ba_i][part_i])

    max_nnz_ch = max_nnz_ch.sum(dim=1)  # [CH]
    col_ch = torch.count_nonzero(col_ch, dim=1)

    return P, max_nnz_ch, col_ch, row_ch

class new_loss_model(nn.Module):
    def __init__(self, D, CH, BA):
        super(new_loss_model, self).__init__()
        self.D = D.cuda()   # [R, C]
        self.R = D.shape[0]
        self.C = D.shape[1]
        self.CH = CH
        self.BA = BA
        self.W = nn.Parameter(torch.zeros((self.R, self.CH*self.BA)), requires_grad=True)
        
        self.sm = GumbelSoftmax()
        torch.nn.init.kaiming_uniform_(self.W)

    def forward(self, i):
        temp = 1000 - 999/3999*i
        P = self.sm(self.W, temp, force_hard=False).reshape(self.R, self.CH, self.BA)  # [R, CH, BA]
        x = P.view(self.R, 1, self.CH, self.BA) * self.D.view(self.R, self.C, 1, 1) # [R, C, CH, BA]

        nnz_ch_ba = torch.sum(x, dim=[0, 1])  # [CH, BA]
        max_nnz_ch = torch.max(nnz_ch_ba, dim=2).values
        mean_nnz_ch = torch.sum(nnz_ch_ba, dim=1) / self.BA
        col_ch = 1 - torch.prod(torch.prod(1-x, 0), 2)  # [C, CH]
        num_col_ch = torch.sum(col_ch, dim=0)
        num_row_ch = torch.sum(P, dim=[0, 2])
  
        col_density_ch = x.sum(dim=[0, 3]) / num_row_ch / num_col_ch  # [C, CH]
        col_density_ch = col_density_ch.sum(dim=0)  # [CH]

        tot_ch = max_nnz_ch + num_col_ch + num_row_ch

        num_row_sel = torch.max(P.reshape(self.R, -1), dim=1).values
        num_row_sel = torch.sum((num_row_sel > 0.99)*1.0).item()

        return P, tot_ch, max_nnz_ch, num_col_ch, num_row_ch, nnz_ch_ba, col_density_ch, num_row_sel

    def mix(self, P):
        x = P.view(self.R, 1, self.CH, self.BA) * self.D.view(self.R, self.C, 1, 1) # [R, C, CH, BA]
        col_density_ch = x.sum(dim=[0, 3]).transpose(0,1) # [CH, C]
        #col_density_ch = col_density_ch * prune_layer(col_density_ch, 0.2)
        #low_density_col = col_density_ch.argmin(dim=1)  # [CH]
        # self.W : [R, CH*BA]

def new_loss(layer, num_ch_):
    D = layer
    R = layer.shape[0]
    C = layer.shape[1]
    CH = num_ch_
    BA = num_ba
    model = new_loss_model(D, CH, BA)
    
    criterion = nn.MSELoss()
    current_best_cost = 1000000000
    prev_best_cost = 1000000000
    best_cost = 1000000000
    optimizer = torch.optim.SGD(model.parameters(), lr=100, momentum=0, weight_decay=0.015)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    
    target_tot = torch.zeros(CH).cuda().data.fill_(R*C*(1-sparsity)//(CH*BA) + C//CH + R//CH - 300)
    target_nnz = torch.zeros((CH, BA)).cuda().data.fill_(num_row*num_col*(1-sparsity)//k - 300)
    target_col = torch.zeros(CH).cuda().data.fill_(C//CH)
    target_row = torch.zeros(CH).cuda().data.fill_(R//CH)
    target_col_density_ch = torch.zeros(CH).cuda().data.fill_(0.1)
    target_P = torch.zeros((R, CH*BA)).cuda().data.fill_(0.)
    lr_init = 30
    lr_noise = 1

    print("target_tot : ", target_tot[0].item())
    print("target_nnz : ", target_nnz[0][0].item())
    print("target_col : ", target_col[0].item())
    print("target_row : ", target_row[0].item())

    model.cuda()
    model.train()

    for i in range(100000000):
        #if (i+1)%200 == 0 and i < 2000:
        #    optimizer = torch.optim.SGD(model.parameters(), lr=20 - i*20/3999, momentum=0, weight_decay=0.015)
        #    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        #elif i == 2000:
        #    optimizer = torch.optim.SGD(model.parameters(), lr=3e-03, momentum=0, weight_decay=0.015)
        #    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
        
        P, tot_ch, max_nnz_ch, num_col_ch, num_row_ch, nnz_ch_ba, col_density_ch, num_row_sel = model(i)
        loss_tot = criterion(tot_ch, target_tot)
        loss_nnz = criterion(nnz_ch_ba, target_nnz)
        loss_col = criterion(num_col_ch, target_col)
        P_ = P.reshape(R, CH*BA)
        loss_P = (P_ * (1-P_)).sum()
        loss_col_density = criterion(col_density_ch, target_col_density_ch)
        
        #loss = loss_tot + i*loss_P
        #if i < 3700:
        #    loss = 100*loss_tot + 100*loss_nnz + (10 - i*10/3999)*loss_col
        #else:
        #    loss = 10*loss_tot + loss_nnz + loss_P * 100000

        #loss = loss_tot + loss_nnz + loss_col + loss_P * 0.1 * (i%400+1)
        if i < 100000:
            loss = 3*loss_tot + loss_nnz
        else:
            loss = loss_tot

        if 1:
            # max(tot_ch) / max(max_nnz_ch) / max(num_col_ch) / max(num_row_ch) / num_row_sel
            print(i, ":\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(torch.max(tot_ch).item(),
                                                                                torch.max(max_nnz_ch).item(),
                                                                                torch.max(num_col_ch).item(),
                                                                                torch.max(num_row_ch).item(),
                                                                                num_row_sel,
                                                                                loss_P.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if torch.max(tot_ch).item() < current_best_cost:
            current_best_cost = torch.max(tot_ch).item()
            cur_model = model

        if (i+1)%400 == 0:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_init*lr_noise*(0.5+random.random()), momentum=0, weight_decay=0.015)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
            print("lr:", optimizer.param_groups[0]['lr'])
            if current_best_cost < prev_best_cost:
                print("+++++ +++++ +++++ ++++ +++++ +++++ +++++")
                print("+++++ +++++ +++++ NICE +++++ +++++ +++++", current_best_cost)
                print("+++++ +++++ +++++ ++++ +++++ +++++ +++++")
                lr_noise = 1
                lr_init = lr_init * 0.8
                model = cur_model
                prev_model = cur_model
                prev_best_cost = current_best_cost
                P_best = P
            else:
                print("------------------")
                print("----- RETURN -----", prev_best_cost)
                print("------------------")
                model = prev_model
                #lr_noise = lr_noise * 0.95
            current_best_cost = 1000000000

        #if (i+1)%100 == 0:
        #    model.mix(P)
 
 
    model.eval()
    P = P_best.cpu()
    D = D.cpu()

    # P : [R, CH, BA] (max percentage to 1)
    P = P.reshape(R, CH*BA)
    max_idx_ = torch.argmax(P, dim=1)
    P.fill_(0.)
    P[torch.arange(R), max_idx_] = 1.
    P = P.reshape(R, CH, BA)
    x = P.view(R, 1, CH, BA) * D.view(R, C, 1, 1) # [R, C, CH, BA]

    nnz_ch_ba = torch.sum(x, dim=[0, 1])  # [CH, BA]
    max_nnz_ch = torch.max(nnz_ch_ba, dim=1).values  # [CH]

    col_ch = 1 - torch.prod(torch.prod(1-x, 0), 2)  # [C, CH]
    num_col_ch = torch.sum(col_ch, dim=0)  # [CH]

    num_row_ch = torch.sum(P, dim=[0, 2])  # [CH]

    return max_nnz_ch, num_col_ch, num_row_ch

class grad_row_hard_model(nn.Module):
    def __init__(self, D, CH, BA):
        super(grad_row_hard_model, self).__init__()
        self.D = D.cuda()   # [R, C]
        self.R = D.shape[0]
        self.C = D.shape[1]
        self.CH = CH
        self.BA = BA
        self.Y = nn.Parameter(torch.zeros((self.R, self.CH*self.BA)), requires_grad=True)
        self.nnz = self.D.sum(dim=1)  # [R]
        
        self.sm = GumbelSoftmax()
        torch.nn.init.kaiming_uniform_(self.Y)

    def forward(self, i):
        temp = 1000 - 999/9999*i
        y = self.sm(self.Y, temp, force_hard=False).reshape(self.R, self.CH, self.BA)  # [R, CH, BA]
        x = y.view(self.R, 1, self.CH, self.BA) * self.D.view(self.R, self.C, 1, 1) # [R, C, CH, BA]

        nnz_ch_ba = torch.sum(x, dim=[0, 1])  # [CH, BA]
        max_nnz_ch = torch.max(nnz_ch_ba, dim=1).values
        mean_nnz_ch = torch.sum(nnz_ch_ba, dim=1) / self.BA

        col_ch = 1 - torch.prod(torch.prod(1-x, 0), 2)  # [C, CH]
        num_col_ch = torch.sum(col_ch, dim=0)
        row_ch = torch.sum(y, dim=[0, 2])
 
        col_freq_ch = torch.sum(x, dim=[0, 3]) / row_ch  # [C, CH]
        col_freq_ch = col_freq_ch * (1 - col_freq_ch)  # [CH, C]
        col_freq_ch = torch.sum(col_freq_ch)  # [1]

        centroid = torch.sum(x, dim=[0, 3]).detach() / row_ch    # [C, CH]
        centroid_diff = centroid.view(1, self.C, self.CH) - x.sum(dim=3)  # [R, C, CH]

        num_row_sel = torch.max(y.reshape(self.R, -1), dim=1).values
        num_row_sel = torch.sum((num_row_sel > 0.98)*1.0).item()

        real_ch = max_nnz_ch + num_col_ch + row_ch
        max_cmd = torch.max(real_ch)

        return y, nnz_ch_ba, max_cmd, real_ch, max_nnz_ch, num_col_ch, row_ch, col_freq_ch, centroid_diff, num_row_sel

def grad_row_hard(layer, num_ch_):
    D = layer
    R = layer.shape[0]
    C = layer.shape[1]
    CH = num_ch_
    BA = num_ba
    model = grad_row_hard_model(D, CH, BA)
    criterion = nn.MSELoss()
    target_tot = torch.zeros(CH).cuda().data.fill_(num_row*num_col*(1-sparsity)//k + num_col//CH + num_row//CH)
    target_nnz = torch.zeros((CH, BA)).cuda().data.fill_(num_row*num_col*(1-sparsity)//k - 300)
    target_col = torch.zeros(CH).cuda().data.fill_(num_col//CH)
    target_row = torch.zeros(CH).cuda().data.fill_(num_row//CH)
    target_col_freq = torch.zeros(CH).cuda().data.fill_(1)
    
    print("target_tot : ", target_tot[0].item())
    print("target_nnz : ", target_nnz[0][0].item())
    print("target_col : ", target_col[0].item())
    print("target_row : ", target_row[0].item())

    best_case = 777777777777
    # for ml application
    #optimizer = torch.optim.SGD(model.parameters(), lr=3, momentum=0, weight_decay=0.15)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    # for scientific dataset
    optimizer = torch.optim.SGD(model.parameters(), lr=200, momentum=0, weight_decay=0.015)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 4000, 0.)

    model.cuda()
    model.train()

    test = 6
    for i in range(10000):
        y, nnz_ch_ba, max_cmd, cmd_ch, max_nnz_ch, col_ch, row_ch, col_freq_ch, centroid_diff, num_row_sel = model(i)
        loss_nnz = criterion(nnz_ch_ba, target_nnz)
        loss_tot = criterion(cmd_ch, target_tot)
        loss_col = criterion(col_ch, target_col)
        #mean_dist = centroid_diff.pow(2).mean(dim=[0, 1]).sqrt().mean()

        if test==0:
            if i%20 < 17:
                loss = loss_nnz + loss_tot
            else:
                loss = criterion(col_ch, target_col)
        elif test==1:
            if i%20 < 17:
                loss = loss_nnz
            elif i%20 < 19:
                loss = loss_nnz + loss_tot
            else:
                loss = loss_nnz + 5*loss_col_freq
        elif test==2:
            loss = loss_nnz + loss_tot + loss_col_freq * i * 100
        elif test==3:
            loss = loss_nnz + loss_tot + mean_dist
        elif test==4:
            if i%1000 > 900:
                loss = loss_nnz + 5 * loss_tot + (y * (1-y)).sum() * 1000 * i + loss_col_freq
            else:
                loss = loss_nnz + 5 * loss_tot + loss_col_freq
        elif test==5:
            if i%1000 > 900:
                loss = loss_tot + loss_nnz + (y*(1-y)).sum() * i
            else:
                loss = loss_tot + loss_nnz
        elif test==6:
            loss = loss_tot + loss_nnz
        elif test==7:
            #loss = i/300*loss_tot + loss_col + col_freq_ch * col_freq_ch
            loss = loss_nnz + loss_col + col_freq_ch * col_freq_ch
            loss = loss_nnz + loss_col + col_freq_ch * col_freq_ch
        else:
            print("hi")

        if 1:
            print(i, ":\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(max_cmd.item(),
                                                                torch.mean(max_nnz_ch).item(),
                                                                torch.mean(col_ch).item(),
                                                                torch.mean(row_ch).item(),
                                                                num_row_sel))
        if (max_cmd.item() < best_case and i > 500 and num_row_sel > num_row*0.5):
            best_case = max_cmd.item()
            y_ = y

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    y = y_.cpu()
    D = D.cpu()

    print(R, CH, BA)
    print(y.shape)
    print(D.shape)

    # y_ [R, CH, BA] (max percentage to 1)
    y = y.reshape(R, CH*BA)
    max_idx_ = torch.argmax(y, dim=1)
    y.fill_(0.)
    y[torch.arange(R), max_idx_] = 1.
    y = y.reshape(R, CH, BA)
    x = y.view(R, 1, CH, BA) * D.view(R, C, 1, 1) # [R, C, CH, BA]

    nnz_ch_ba = torch.sum(x, dim=[0, 1])  # [CH, BA]
    nnz_ch = torch.sum(nnz_ch_ba, dim=1)  # [CH]
    max_nnz_ch = torch.max(nnz_ch_ba, dim=1).values  # [CH]

    col_ch_ba = 1 - torch.prod(1-x, 0)  # [C, CH, BA]
    col_ch = 1 - torch.prod(1-col_ch_ba, 2)  # [C, CH]

    num_col_ch_ba = torch.sum(col_ch_ba, dim=0)  # [CH, BA]
    num_col_ch = torch.sum(col_ch, dim=0)  # [CH]

    row_ch_ba = torch.sum(y, dim=0)  # [CH, BA]
    row_ch = torch.sum(y, dim=[0, 2])  # [CH]

    nnz_ch_ba = nnz_ch_ba.reshape(-1)
    num_col_ch_ba = num_col_ch_ba.reshape(-1)
    row_ch_ba = row_ch_ba.reshape(-1)

    return max_nnz_ch, num_col_ch, row_ch

class test_loss_model(nn.Module):
    def __init__(self, D, args):
        super(test_loss_model, self).__init__()
        self.D = D.to(device)   # [R, C]
        self.R = D.shape[0]
        self.C = D.shape[1]
        self.CH = int(args.num_ch)
        self.BA = int(args.num_ba)
        self.W = nn.Parameter(torch.zeros((self.R, self.CH*self.BA)), requires_grad=True)
        self.device = args.device
        self.num_iter = int(args.num_iter)
        self.gs_hard = args.gs_hard
        
        self.sm = GumbelSoftmax()
        torch.nn.init.kaiming_uniform_(self.W)

    def forward(self, i):
        temp = 1000 - 999 * i / self.num_iter
        P = self.sm(self.W, temp, force_hard=self.gs_hard).reshape(self.R, self.CH, self.BA)  # [R, CH, BA]
        x = P.view(self.R, 1, self.CH, self.BA) * self.D.view(self.R, self.C, 1, 1) # [R, C, CH, BA]

        nnz_ch_ba = torch.sum(x, dim=[0, 1])  # [CH, BA]
        max_nnz_ch = torch.max(nnz_ch_ba, dim=1).values
        
        col_ch = 1 - torch.prod(torch.prod(1-x, 0), 2)  # [C, CH]
        num_col_ch = torch.sum(col_ch, dim=0)
        
        num_row_ch = torch.sum(P, dim=[0, 2])
  
        col_density_ch = x.sum(dim=[0, 3]) / num_row_ch  # [C, CH]
        col_density_ch = (col_density_ch * (1-col_density_ch)).sum()

        cost_ch = max_nnz_ch + num_col_ch + num_row_ch

        num_row_sel = torch.max(P.reshape(self.R, -1), dim=1).values
        num_row_sel = torch.sum((num_row_sel > 0.97)*1.0).item()

        return P, cost_ch, max_nnz_ch, num_col_ch, num_row_ch, nnz_ch_ba, col_density_ch, num_row_sel

def test_loss(layer, args):
    D = layer
    R = layer.shape[0]
    C = layer.shape[1]
    CH = int(args.num_ch)
    BA = int(args.num_ba)
    device = args.device
    model = test_loss_model(D, args)
    num_iter = int(args.num_iter)
    lr_init = float(args.lr_init)
    w_decay = float(args.w_decay)
    
    criterion = nn.MSELoss()
    best_cost = 1000000000
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_init, momentum=0, weight_decay=w_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    
    #target_tot = torch.zeros(CH).to(device).data.fill_(R*C*(1-sparsity)//(CH*BA) + C//CH + R//CH - 300)
    target_tot = torch.zeros(CH).to(device).data.fill_(30000)
    target_nnz = torch.zeros((CH, BA)).to(device).data.fill_(num_row*num_col*(1-sparsity)//k - 300)
    target_col = torch.zeros(CH).to(device).data.fill_(C//CH)
    target_row = torch.zeros(CH).to(device).data.fill_(R//CH)

    print("target_tot : ", target_tot[0].item())
    print("target_nnz : ", target_nnz[0][0].item())
    print("target_col : ", target_col[0].item())
    print("target_row : ", target_row[0].item())

    model.to(device)
    model.train()

    for i in range(num_iter):
        P, cost_ch, max_nnz_ch, num_col_ch, num_row_ch, nnz_ch_ba, col_density_ch, num_row_sel = model(i)
        loss_tot = criterion(cost_ch, target_tot)
        loss_nnz = criterion(nnz_ch_ba, target_nnz)
        
        loss = loss_tot + loss_nnz + col_density_ch * 10000

        if 1:   # max(tot_ch) / max(max_nnz_ch) / max(num_col_ch) / max(num_row_ch) / num_row_sel
            print(i, ":\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(torch.max(cost_ch).item(),
                                                                                torch.max(max_nnz_ch).item(),
                                                                                torch.mean(num_col_ch).item(),
                                                                                torch.mean(num_row_ch).item(),
                                                                                num_row_sel))
        if (torch.max(cost_ch) < best_cost and i > 1000):
            best_cost = torch.max(cost_ch)
            P_best = P
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


    model.eval()
    P = P_best.cpu()
    D = D.cpu()

    # P : [R, CH, BA] (max percentage to 1)
    P = P.reshape(R, CH*BA)
    max_idx_ = torch.argmax(P, dim=1)
    P.fill_(0.)
    P[torch.arange(R), max_idx_] = 1.
    P = P.reshape(R, CH, BA)
    x = P.view(R, 1, CH, BA) * D.view(R, C, 1, 1) # [R, C, CH, BA]

    nnz_ch_ba = torch.sum(x, dim=[0, 1])  # [CH, BA]
    max_nnz_ch = torch.max(nnz_ch_ba, dim=1).values  # [CH]

    col_ch = 1 - torch.prod(torch.prod(1-x, 0), 2)  # [C, CH]
    num_col_ch = torch.sum(col_ch, dim=0)  # [CH]

    num_row_ch = torch.sum(P, dim=[0, 2])  # [CH]

    return P, max_nnz_ch, num_col_ch, num_row_ch

class grad_row_soft_model(nn.Module):
    def __init__(self, D, CH, BA):
        super(grad_row_soft_model, self).__init__()
        self.D = D.cuda()   # [R, C]
        self.part_col = D.shape[0]
        self.R = D.shape[1]
        self.C = D.shape[2]  # C//part_col
        self.CH = CH
        self.BA = BA
        self.Y = nn.Parameter(torch.zeros((self.part_col*self.R, CH*BA)), requires_grad=True)
        
        self.sm = GumbelSoftmax()
        torch.nn.init.kaiming_uniform_(self.Y)

    def forward(self, i):
        temp = 1000 - 999/3999*i
        y = self.sm(self.Y, temp, force_hard=False).reshape(self.part_col, self.R, self.CH, self.BA)  # [P_COL, R, CH, BA]
        # [P_COL, R, C//P_COL, CH, BA]
        x = y.view(self.part_col, self.R, 1, self.CH, self.BA) * self.D.view(self.part_col, self.R, self.C, 1, 1)

        nnz_ch_ba = torch.sum(x, dim=[0, 1, 2])  # [CH, BA]
        max_nnz_ch = torch.max(nnz_ch_ba, dim=1).values  # [CH]
        mean_nnz_ch = torch.sum(nnz_ch_ba, dim=1) / self.BA

        col_ch = 1 - torch.prod(torch.prod(1-x, 1), 3) # [P_COL, C//P_COL, CH]
        num_col_ch = torch.sum(col_ch, dim=[0, 1])
        
        row_ch = 1 - torch.prod(torch.prod(1-y, 0), 2)  # [R, CH]
        num_row_ch = torch.sum(row_ch, dim=0)    # [CH]
        
        mean_cost_ch = mean_nnz_ch + num_col_ch + num_row_ch
        real_cost_ch = max_nnz_ch + num_col_ch + num_row_ch
        
        max_cost = torch.max(real_cost_ch)

        return y, nnz_ch_ba, max_cost, mean_cost_ch, real_cost_ch, max_nnz_ch, num_col_ch, num_row_ch

def grad_row_soft(layer, num_ch_):
    D = layer
    R = layer.shape[1]
    C = layer.shape[2]
    CH = num_ch
    BA = num_ba
    model = grad_row_soft_model(D, CH, BA)
    criterion = nn.MSELoss()
    target_tot = torch.zeros(CH).cuda().data.fill_(num_row*num_col*(1-sparsity)//k + num_col//CH + num_row//CH)
    target_nnz = torch.zeros((CH, BA)).cuda().data.fill_(num_row*num_col*(1-sparsity)//k)
    target_col = torch.zeros(CH).cuda().data.fill_(num_col//CH)
    target_row = torch.zeros(CH).cuda().data.fill_(num_row//CH)
    
    print("target_tot : ", target_tot[0].item())
    print("target_nnz : ", target_nnz[0][0].item())
    print("target_col : ", target_col[0].item())
    print("target_row : ", target_row[0].item())

    best_case = 777777777777
    optimizer = torch.optim.SGD(model.parameters(), lr=100, momentum=0., weight_decay=0.0015)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 4000, 0.)

    model.cuda()
    model.train()

    for i in range(4000):
        y, nnz_ch_ba, max_cost, mean_cost_ch, real_cost_ch, max_nnz_ch, col_ch, row_ch = model(i)
        #loss = criterion(nnz_ch_ba, target_nnz) + criterion(mean_cost_ch, target_tot)
        #loss_tot = criterion(mean_cost_ch, target_tot)
        #loss_nnz = criterion(nnz_ch_ba, target_nnz)
        #loss_col = criterion(col_ch, target_col)
        #loss_row = criterion(row_ch, target_row)
        if i%40 < 20:  # row, col
            loss = criterion(row_ch + col_ch, target_row + target_col)
        elif i%40 < 35:  # nnz
            loss = criterion(nnz_ch_ba, target_nnz)
        else:  #tot
            loss = criterion(mean_cost_ch, target_tot)
        #loss = loss_nnz + loss_col + loss_row
        #loss = i*loss_nnz + (5000-i)*loss_col + (5000-i)*loss_row
        #loss = loss_nnz + loss_tot
        
        if 1:
            print(i, ":\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(max_cost.item(),
                                                                torch.max(max_nnz_ch).item(),
                                                                torch.max(col_ch).item(),
                                                                torch.max(row_ch).item()))
        else:
            print(i, ":\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(max_cost.item(),
                                                                loss_nnz.item(),
                                                                loss_col.item(),
                                                                loss_row.item()))
        if (max_cost.item() < best_case and i > 100):
            best_case = max_cost.item()
            y_ = y

        y_idx = torch.argmax(y.clone().detach().reshape(R, -1), dim=1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    y = y_.cpu()
    D = D.cpu()

    print(R, CH, BA)
    print(y.shape)
    print(D.shape)

    # y_ [P_COL, R, CH, BA] (max percentage to 1)
    y = y.reshape(part_col*R, CH*BA)
    max_idx_ = torch.argmax(y, dim=1)
    y.fill_(0.)
    y[torch.arange(part_col*R), max_idx_] = 1.
    y = y.reshape(part_col, R, CH, BA)

    # [P_COL, R, C//P_COL, CH, BA]
    x = y.view(part_col, R, 1, CH, BA) * D.view(part_col, R, C, 1, 1)

    nnz_ch_ba = torch.sum(x, dim=[0, 1, 2])  # [CH, BA]
    max_nnz_ch = torch.max(nnz_ch_ba, dim=1).values  # [CH]

    num_col_ch = torch.zeros(CH)
    for part_i in range(part_col):
        z = x[part_i] # [R, C, CH, BA]
        num_col_ch = num_col_ch + torch.sum(1 - torch.prod(torch.prod(1-z, 0), 2), dim=0) # [CH]
        
    row_ch = 1 - torch.prod(torch.prod(1-y, 0), 2)  # [R, CH]
    num_row_ch = torch.sum(row_ch, dim=0)    # [CH]
    
    return max_nnz_ch, num_col_ch, num_row_ch

class grad_nnz_model(nn.Module):
    def __init__(self, D, CH, BA):
        super(grad_nnz_model, self).__init__()
        self.D = D.cuda()   # [R, C]
        self.R = D.shape[0]
        self.C = D.shape[1]
        self.CH = CH
        self.BA = BA
        self.W = nn.Parameter(torch.zeros((self.R*self.C, self.CH)), requires_grad=True)
        
        self.sm = GumbelSoftmax()
        torch.nn.init.kaiming_uniform_(self.W)

    def forward(self, i):
        temp = 1000 - 899/3999*i
        P = self.sm(self.W, temp, force_hard=False)*self.D.view(self.R*self.C, 1)  # [R*C, CH]
        P = P.reshape(self.R, self.C, self.CH)  # [R, C, CH]

        nnz_ch = P.sum(dim=[0,1])  # [CH]
        col_ch = 1 - torch.prod(1-P, 0)  # [C, CH]
        row_ch = 1 - torch.prod(1-P, 1)  # [R, CH]

        mean_nnz_ch = nnz_ch / self.BA  # [CH]
        num_col_ch = col_ch.sum(0)  # [CH]
        num_row_ch = row_ch.sum(0)
  
        col_density_ch = P.sum(0) / num_row_ch / num_col_ch  # [C, CH]
        col_density_ch = col_density_ch.sum(0)  # [CH]

        tot_ch = mean_nnz_ch + num_col_ch + num_row_ch

        num_nnz_sel = torch.max(P.reshape(self.R*self.C, -1), dim=1).values  # [R*C]
        num_nnz_sel = torch.sum((num_nnz_sel > 0.99)*1.0).item()

        return P, tot_ch, mean_nnz_ch, num_col_ch, num_row_ch, col_density_ch, num_nnz_sel

def grad_nnz(layer, num_ch_):
    D = layer
    R = layer.shape[0]
    C = layer.shape[1]
    CH = num_ch_
    BA = num_ba
    model = grad_nnz_model(D, CH, BA)
    criterion = nn.MSELoss()
    
    best_case = 10000000000
    optimizer = torch.optim.SGD(model.parameters(), lr=100, momentum=0, weight_decay=0.015)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    target_tot = torch.zeros(CH).cuda().data.fill_(R*C*(1-sparsity)//(CH*BA) + C//CH + R//CH - 300)
    target_col = torch.zeros(CH).cuda().data.fill_(C//CH)
    target_row = torch.zeros(CH).cuda().data.fill_(R//CH)
    target_col_density_ch = torch.zeros(CH).cuda().data.fill_(0.1)
    target_P = torch.zeros((R, CH*BA)).cuda().data.fill_(0.)
    print("target_tot : ", target_tot[0].item())
    print("target_col : ", target_col[0].item())
    print("target_row : ", target_row[0].item())

    model.cuda()
    model.train()

    for i in range(4000):
        P, tot_ch, mean_nnz_ch, num_col_ch, num_row_ch, col_density_ch, num_nnz_sel = model(i)

        loss_tot = criterion(tot_ch, target_tot)
        loss_col = criterion(num_col_ch, target_col)
        loss_row = criterion(num_row_ch, target_row)
        
        P_ = P.reshape(R*C, CH)
        loss_P = (P_ * (1-P_)).sum()
        loss_col_density = criterion(col_density_ch, target_col_density_ch)
        
        loss = loss_tot

        if 1:
            # max(tot_ch) / max(max_nnz_ch) / max(num_col_ch) / max(num_row_ch) / num_row_sel
            print(i, ":\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(torch.max(tot_ch).item(),
                                                                                torch.max(mean_nnz_ch).item(),
                                                                                torch.max(num_col_ch).item(),
                                                                                torch.max(num_row_ch).item(),
                                                                                num_nnz_sel,
                                                                                loss_P.item()))
        if torch.max(tot_ch).item() < best_case and i > 100:
            best_case = torch.max(tot_ch).item()
            P_best = P

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    P = P_best.cpu()
    D = D.cpu()

    P = P.reshape(R*C, CH)
    max_idx_ = torch.argmax(P, dim=1)
    P.fill_(0.)
    P[torch.arange(R*C), max_idx_] = 1
    P = (P * D.view(R*C, 1)).reshape(R, C, CH)

    nnz_ch = P.sum(dim=[0,1])  # [CH]
    col_ch = 1 - torch.prod(1-P, 0)  # [C, CH]
    row_ch = 1 - torch.prod(1-P, 1)  # [R, CH]

    mean_nnz_ch = nnz_ch / BA  # [CH]
    num_col_ch = col_ch.sum(0)  # [CH]
    num_row_ch = row_ch.sum(0)

    torch.save(P, 'nnz'+str(sparsity)+'.pt')

    return P, mean_nnz_ch, num_col_ch, num_row_ch

# M [C, R], P [R, CH] → [C, CH]
def sparse_dense_mul(M, P, nnz_rows, device):
    C = nnz_rows.shape[0]
    R = P.shape[0]
    CH = P.shape[1]
    i_row = M._indices()[1]  # [nnz row indices]

    num_col_ch = torch.zeros(CH).to(device)
    e_idx = 0
    for c_i in range(C):
        s_idx = e_idx
        e_idx = int(nnz_rows[c_i].item())

        tar_prob = i_row[s_idx:e_idx]  # [R']
        col_prob = 1 - torch.prod(P[tar_prob, :], dim=0)  # [CH]
        num_col_ch = num_col_ch + col_prob  # [CH]

    return num_col_ch

class grad_row_lowmem_model(nn.Module):
    def __init__(self, D, CH, BA, device):
        super(grad_row_lowmem_model, self).__init__()
        self.nnz_rows = torch.cumsum(D.T.sum(1), dim=0).to(device) # [C]
        print("nnz_rows: ", self.nnz_rows)
        self.D = D.T.to_sparse_coo().to(device)   # [C, R]
        self.R = D.shape[0]
        self.C = D.shape[1]
        self.CH = CH
        self.BA = BA
        self.device = device
        self.Y = nn.Parameter(torch.zeros((self.R, self.CH*self.BA)), requires_grad=True)
        self.nnz = D.sum(dim=1).to(device)  # [R]
        
        self.sm = GumbelSoftmax()
        torch.nn.init.kaiming_uniform_(self.Y)

    def forward(self, i):
        temp = 1000 - 999/2000*i
        y = self.sm(self.Y, temp, force_hard=False).reshape(self.R, self.CH, self.BA)  # [R, CH, BA]
        y_ch = y.sum(dim=2) # [R, CH]
        y_ch_ = 1 - y_ch  # [R, CH]

        x_ = y * self.nnz.view(-1, 1, 1)  # [R, CH, BA]
        nnz_ch_ba = torch.sum(x_, dim=0)  # [CH, BA]
        max_nnz_ch = torch.max(nnz_ch_ba, dim=1).values
        mean_nnz_ch = torch.sum(nnz_ch_ba, dim=1) / self.BA

        #num_col_ch = sparse_dense_mul(self.D, y_ch_, self.nnz_rows, self.device)
        num_col_ch = torch.zeros(self.CH).fill_(3000).to(self.device)

        row_ch = torch.sum(y, dim=[0, 2])
 
        real_ch = max_nnz_ch + num_col_ch + row_ch
        max_cmd = torch.max(real_ch)

        return y, nnz_ch_ba, max_cmd, real_ch, max_nnz_ch, num_col_ch, row_ch

def grad_row_lowmem(layer, num_ch_, device):
    D = layer
    R = layer.shape[0]
    C = layer.shape[1]
    CH = num_ch_
    BA = num_ba
    model = grad_row_lowmem_model(D, CH, BA, device)
    criterion = nn.MSELoss()
    #target_tot = torch.zeros(CH).to(device).data.fill_(num_row*num_col*(1-sparsity)//k + num_col//CH + num_row//CH)
    target_tot = torch.zeros(CH).to(device).data.fill_(40000)
    target_nnz = torch.zeros((CH, BA)).to(device).data.fill_(num_row*num_col*(1-sparsity)//k - 300)
    target_col = torch.zeros(CH).to(device).data.fill_(num_col//CH)
    target_row = torch.zeros(CH).to(device).data.fill_(num_row//CH)
    target_col_freq = torch.zeros(CH).to(device).data.fill_(1)
    
    print("target_tot : ", target_tot[0].item())
    print("target_nnz : ", target_nnz[0][0].item())
    print("target_col : ", target_col[0].item())
    print("target_row : ", target_row[0].item())

    best_case = 777777777777
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0, weight_decay=0.3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 4000, 0.)

    model.to(device)
    model.train()

    for i in range(2000):
        y, nnz_ch_ba, max_cmd, cmd_ch, max_nnz_ch, col_ch, row_ch = model(i)
        loss_nnz = criterion(nnz_ch_ba, target_nnz)
        loss_tot = criterion(cmd_ch, target_tot)
        loss_col = criterion(col_ch, target_col)

        loss = loss_col + loss_nnz + loss_tot
        #loss = loss_nnz + loss_tot + loss_col * 100
        print(i, ":\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(max_cmd.item(),
                                                            torch.max(max_nnz_ch).item(),
                                                            torch.mean(col_ch).item(),
                                                            torch.mean(row_ch).item()))
        if (max_cmd.item() < best_case and i > 300):
            best_case = max_cmd.item()
            y_ = y

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    y = y_.cpu()
    D = D.cpu()

    print(R, CH, BA)
    print(y.shape)
    print(D.shape)

    # y_ [R, CH, BA] (max percentage to 1)
    y = y.reshape(R, CH*BA)
    max_idx_ = torch.argmax(y, dim=1)
    y.fill_(0.)
    y[torch.arange(R), max_idx_] = 1.
    y = y.reshape(R, CH, BA)

    max_nnz_ch = torch.zeros(CH)
    num_col_ch = torch.zeros(CH)
    row_ch = torch.zeros(CH)
    
    return max_nnz_ch, num_col_ch, row_ch

class grad_row_register_size_model(nn.Module):
    def __init__(self, D, CH, BA, register_size, device):
        super(grad_row_register_size_model, self).__init__()
        self.D = D.to(device)   # [R, C]
        self.R = D.shape[0]
        self.C = D.shape[1]
        self.CH = CH
        self.BA = BA
        self.nnz = self.D.sum(dim=1)  # [R]
        self.reg_size = register_size
        self.device = device
        self.num_part = (self.C-1) // self.reg_size + 1
        
        zeros = torch.zeros(self.R, self.num_part * self.reg_size - self.C).to(device)
        self.D_ = torch.cat([self.D, zeros], dim=1).view(self.R, self.num_part, -1) # [R, num_part, reg_size]

        self.W = nn.Parameter(torch.zeros((self.R, self.CH*self.BA)), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.W)
        self.sm = GumbelSoftmax()

        self.mask = ((self.nnz != 0)*1.0).reshape(self.R, 1, 1)

    def forward(self, i):
        temp = 1000 - 999/9999*i
        P = self.sm(self.W, temp, force_hard=False).reshape(self.R, self.CH, self.BA)  # [R, CH, BA]
        P = P * self.mask  # Get rid of nnz=0 rows
        x = P.view(self.R, 1, 1, self.CH, self.BA) * self.D_.view(self.R, self.num_part, self.reg_size, 1, 1) # [R, np, rs, CH, BA]

        nnz_ch_ba = torch.sum(x, dim=[0, 2])  # [np, CH, BA]
        max_nnz_ch = torch.max(nnz_ch_ba, dim=2).values  # [np, CH]
        tot_max_nnz_ch = max_nnz_ch.sum(dim=0)  # [CH]

        num_col_ch = torch.zeros(self.CH).fill_(self.C).to(device)  # Set to all column for now
        num_row_ch = torch.sum(P, dim=[0, 2])

        cost_ch = tot_max_nnz_ch + num_col_ch + num_row_ch
        max_cost = torch.max(cost_ch)

        return P, max_cost, cost_ch, max_nnz_ch, nnz_ch_ba, num_col_ch, num_row_ch, tot_max_nnz_ch

def grad_row_register_size(layer, args):
    device = args.device
    D = layer.to(device)
    R = layer.shape[0]
    C = layer.shape[1]
    CH = int(args.num_ch)
    BA = int(args.num_ba)
    register_size = int(args.register_size)
    lr_init = float(args.lr_init)
    weight_decay = float(args.w_decay)
  
    ideal_nnz = R * C * (1-sparsity) // (CH*BA)
    ideal_tot = ideal_nnz + C + R/CH
    target_tot_ = ideal_tot * 0.8

    num_part = ((C-1)//register_size) + 1
    zeros = torch.zeros(R, num_part * register_size - C).to(device)
    D_ = torch.cat([D, zeros], dim=1).view(R, num_part, -1)  # [R, num_part, register_size]
    model = grad_row_register_size_model(D, CH, BA, register_size, device)
    criterion = nn.MSELoss()
    target_tot = torch.zeros(CH).to(device).data.fill_(target_tot_)
 
    print("target_tot : ", target_tot[0].item())

    best_case = 777777777777
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_init, momentum=0, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)

    model.to(device)
    model.train()

    for i in range(10000):
        P, max_cost, cost_ch, max_nnz_ch, nnz_ch_ba, num_col_ch, num_row_ch, tot_max_nnz_ch = model(i)
        # nnz_ch_ba [np, CH, BA]
        loss_tot = criterion(cost_ch, target_tot)
        loss_new = torch.std(nnz_ch_ba, dim=2).sum()

        loss = (0.5+i/5000)*loss_tot + 20*loss_new
        
        print(i, ":\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\tloss: {:.2f}\t{:.2f}".format(max_cost.item(),
                                                            torch.max(tot_max_nnz_ch).item(),
                                                            torch.max(num_col_ch).item(),
                                                            torch.max(num_row_ch).item(),
                                                            loss_tot.item(),
                                                            loss_new.item()))
        if (max_cost.item() < best_case and i > 1000):
            best_case = max_cost.item()
            P_ = P

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    P = P_.cpu()
    D_ = D_.cpu()
    
    # P [R, CH, BA] (max percentage to 1)
    P = P.reshape(R, CH*BA)
    max_idx_ = torch.argmax(P, dim=1)
    P.fill_(0.)
    P[torch.arange(R), max_idx_] = 1.
    P = P.reshape(R, CH, BA)
    x = P.view(R, 1, 1, CH, BA) * D_.view(R, num_part, register_size, 1, 1) # [R, np, rs, CH, BA]

    nnz_ch_ba = torch.sum(x, dim=[0, 2])  # [np, CH, BA]
    max_nnz_ch = torch.max(nnz_ch_ba, dim=2).values  # [np, CH]
    tot_max_nnz_ch = max_nnz_ch.sum(dim=0)  # [CH]

    num_col_ch = torch.zeros(CH).fill_(C)  # Set to all column for now
    num_row_ch = torch.sum(P, dim=[0, 2])

    return P, tot_max_nnz_ch, num_col_ch, num_row_ch

class grad_row_no_register_size_model(nn.Module):
    def __init__(self, D, CH, BA, register_size, device):
        super(grad_row_no_register_size_model, self).__init__()
        self.D = D.to(device)   # [R, C]
        self.R = D.shape[0]
        self.C = D.shape[1]
        self.CH = CH
        self.BA = BA
        self.nnz = self.D.sum(dim=1)  # [R]
        self.reg_size = register_size
        self.device = device
        self.num_part = (self.C-1) // self.reg_size + 1

        self.W = nn.Parameter(torch.zeros((self.R, self.CH*self.BA)), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.W)
        self.sm = GumbelSoftmax()
        
        self.mask = ((self.nnz != 0)*1.0).reshape(self.R, 1, 1)

    def forward(self, i):
        temp = 1000 - 999/9999*i
        P = self.sm(self.W, temp, force_hard=False).reshape(self.R, self.CH, self.BA)  # [R, CH, BA]
        P = P * self.mask  # Get rid of nnz=0 rows
        x = P.view(self.R, 1, self.CH, self.BA) * self.D.view(self.R, self.C, 1, 1) # [R, C, CH, BA]

        nnz_ch_ba = torch.sum(x, dim=[0, 1])  # [CH, BA]
        max_nnz_ch = torch.max(nnz_ch_ba, dim=1).values  # [CH]
        tot_max_nnz_ch = max_nnz_ch  # [CH]

        num_col_ch = torch.zeros(self.CH).fill_(self.C).to(device)  # Set to all column for now
        num_row_ch = torch.sum(P, dim=[0, 2])

        cost_ch = tot_max_nnz_ch + num_col_ch + num_row_ch
        max_cost = torch.max(cost_ch)

        return P, max_cost, cost_ch, max_nnz_ch, nnz_ch_ba, num_col_ch, num_row_ch, tot_max_nnz_ch

def grad_row_no_register_size(layer, args):
    device = args.device
    D = layer.to(device)
    R = layer.shape[0]
    C = layer.shape[1]
    CH = int(args.num_ch)
    BA = int(args.num_ba)
    register_size = int(args.register_size)
    lr_init = float(args.lr_init)
    weight_decay = float(args.w_decay)
    
    ideal_nnz = R * C * (1-sparsity) // (CH*BA)
    ideal_tot = ideal_nnz + C + R/CH
    target_nnz_ = ideal_nnz * 0.8
    target_tot_ = ideal_tot * 0.8

    num_part = ((C-1)//register_size) + 1
    zeros = torch.zeros(R, num_part * register_size - C).to(device)
    D_ = torch.cat([D, zeros], dim=1).view(R, num_part, -1)  # [R, num_part, register_size]
    model = grad_row_no_register_size_model(D, CH, BA, register_size, device)
    criterion = nn.MSELoss()
    target_tot = torch.zeros(CH).to(device).data.fill_(target_tot_)
    target_nnz = torch.zeros((CH, BA)).to(device).data.fill_(target_nnz_)
 
    print("target_tot : ", target_tot[0].item())
    print("target_nnz : ", target_nnz[0][0].item())

    best_case = 777777777777
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_init, momentum=0, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)

    model.to(device)
    model.train()

    for i in range(10000):
        P, max_cost, cost_ch, max_nnz_ch, nnz_ch_ba, num_col_ch, num_row_ch, tot_max_nnz_ch = model(i)
        # nnz_ch_ba [np, CH, BA]
        loss_nnz = criterion(nnz_ch_ba, target_nnz)
        loss_tot = criterion(cost_ch, target_tot)

        loss = (0.5+i/5000)*loss_tot + loss_nnz
        
        print(i, ":\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\tloss: {:.2f}\t{:.2f}".format(max_cost.item(),
                                                            torch.max(tot_max_nnz_ch).item(),
                                                            torch.max(num_col_ch).item(),
                                                            torch.max(num_row_ch).item(),
                                                            loss_tot.item(),
                                                            loss_nnz.item()))
        if (max_cost.item() < best_case and i > 1000):
            best_case = max_cost.item()
            P_ = P

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    P = P_.cpu()
    D_ = D_.cpu()
    
    # P [R, CH, BA] (max percentage to 1)
    P = P.reshape(R, CH*BA)
    max_idx_ = torch.argmax(P, dim=1)
    P.fill_(0.)
    P[torch.arange(R), max_idx_] = 1.
    P = P.reshape(R, CH, BA)
    x = P.view(R, 1, 1, CH, BA) * D_.view(R, num_part, register_size, 1, 1) # [R, np, rs, CH, BA]

    nnz_ch_ba = torch.sum(x, dim=[0, 2])  # [np, CH, BA]
    max_nnz_ch = torch.max(nnz_ch_ba, dim=2).values  # [np, CH]
    tot_max_nnz_ch = max_nnz_ch.sum(dim=0)  # [CH]

    num_col_ch = torch.zeros(CH).fill_(C)  # Set to all column for now
    num_row_ch = torch.sum(P, dim=[0, 2])

    return P, tot_max_nnz_ch, num_col_ch, num_row_ch

def assign_row(D_, P, sum_row_ch, sum_col_ch, ch_i, r_i):
    aft_sum_col_ch = sum_col_ch[ch_i] + D_[r_i]  # [P_COL, COL_P]
    aft_sum_row_ch = torch.clone(sum_row_ch[ch_i])  # [R]
    aft_sum_row_ch[r_i] = aft_sum_row_ch[r_i] - 1
    aft_nnz_ch = aft_sum_col_ch.sum() / num_ba
    aft_num_col_ch = torch.count_nonzero(aft_sum_col_ch)
    aft_num_row_ch = torch.count_nonzero(aft_sum_row_ch)
    return aft_nnz_ch, aft_num_col_ch, aft_num_row_ch

def pick_row_cuda(D_, P, sum_row_ch, sum_col_ch):  
    # D_ : [R, P_COL, COL_P]
    # P : [R*P_COL]
    # sum_row_ch : [CH, R]
    # sum_col_ch : [CH, P_COL, COL_P]
    # aft_nnz_ch : [R*P_COL, CH]
    # aft_num_col_ch : [R*P_COL, CH]
    # aft_num_row_ch : [R*P_COL, CH]
    aft_sum_row_ch = sum_row_ch.repeat(num_row*part_col, 1)  # [R*P_COL, CH, R]
    aft_sum_col_ch = sum_col_ch.repeat(num_row*part_col, 1, 1)  # [R*P_COL, CH, P_COL, COL_P]

    nnz_ch = sum_col_ch.sum(dim=[1,2])  # [CH]
    num_row_ch = sum_row_ch.sum(dim=1)  # [CH]

    aft_nnz_ch = nnz_ch.repeat(num_row*part_col, 1)  # [R*P_COL, CH]
    nnz = D_.sum(dim=2).reshape(num_row*part_col)  # [R*P_COL]
    P_CH = P[torch.arange(num_row*part_col)] // num_ba  # [R*P_COL]
    aft_nnz_ch[torch.arange(num_row*part_col), P_CH] -= nnz[torch.arange(num_row*part_col)]
    #aft_num_col_ch[torch.arange(num_row*part_col), P_CH] -= 

    return aft_nnz_ch, aft_num_col_ch, aft_num_row_ch


def pick_row(D_, P, sum_row_ch, sum_col_ch, rp_i):
    r_i = rp_i // part_col
    p_i = rp_i % part_col
    ch_i = P[rp_i] // num_ba
    
    aft_sum_col_ch = sum_col_ch[ch_i] - D_[r_i]  # [P_COL, COL_P]
    aft_sum_row_ch = torch.clone(sum_row_ch[ch_i])  # [R]
    aft_sum_row_ch[r_i] = aft_sum_row_ch[r_i] + 1
    aft_nnz_ch = aft_sum_col_ch.sum() / num_ba
    aft_num_col_ch = torch.count_nonzero(aft_sum_col_ch)
    aft_num_row_ch = torch.count_nonzero(aft_sum_row_ch)
    return aft_nnz_ch, aft_num_col_ch, aft_num_row_ch
 
def pop_push_1row(layer, num_ch_):
    D = layer  # [P_COL, R, C//P_COL]
    P_COL = layer.shape[0]
    R = layer.shape[1]
    COL_P = layer.shape[2]  # = C//P_COL
    CH = num_ch
    BA = num_ba

    # Setting
    D_ = D.clone().transpose(0,1)  # [R, P_COL, COL_P]
    nnz_ch_ba = torch.zeros((CH, BA))
    nnz_ch = torch.zeros(CH)
    num_col_ch = torch.zeros(CH)
    num_row_ch = torch.zeros(CH)
    col_ch = torch.zeros(CH, P_COL, COL_P)
    row_ch = torch.zeros(CH, R)

    # Random Initializing
    P = torch.randint(CH*BA, (R*P_COL,))  # [R*P_COL]
    y = torch.zeros((R*P_COL, CH*BA))  # [R*P_COL, CH*BA]
    y[torch.arange(R*P_COL), P] = 1.
    y = y.reshape(R, P_COL, CH, BA)  # [R, P_COL, CH, BA]
    x = y.reshape(R, P_COL, 1, CH, BA) * D.reshape(R, P_COL, COL_P, 1, 1)  # [R, P_COL, COL_P, CH, BA]
    
    sum_col_ch = x.sum(dim=[0,4]).transpose(0,2).transpose(1,2) # [CH, P_COL, COL_P]
    sum_row_ch = y.sum(dim=[1,3]).transpose(0,1)   # [CH, R]

    score_rp = torch.zeros((R*P_COL))
    score_ch = torch.zeros(CH)

    return nnz_ch, col_ch, row_ch

    # gonna change y & sum_col_ch → calculate nnz_ch, num_col_ch, num_row_ch    
    for i in range(300000000):
        nnz_ch = sum_col_ch.sum(dim=[1,2]) / BA
        num_col_ch = torch.count_nonzero(sum_col_ch, dim=[1,2])
        num_row_ch = torch.count_nonzero(sum_row_ch, dim=1)
        tot_ch = nnz_ch + num_col_ch + num_row_ch
        print(i, ":\t{:.2f}\t{:.2f}".format(tot_ch.sum(), max(tot_ch).item()), end="\t")

        # calculate score_rp by cpu
        ch_i = torch.argmax(tot_ch)
        """
        for rp_i in range(R*P_COL):
            if P[rp_i].item() // BA != ch_i:
                score_rp[rp_i] = -1
                continue

            ch_i = P[rp_i] // num_ba
            nnz_ch_, num_col_ch_, num_row_ch_ = pick_row(D_, P, sum_row_ch, sum_col_ch, rp_i)
            tot_ch_ = torch.clone(tot_ch)
            tot_ch_[ch_i] = nnz_ch_ + num_col_ch_ + num_row_ch_
            score_rp[rp_i] = max(tot_ch_).item()
        """
        while(1):
            rp_i = torch.randint(R*P_COL, (1,)).item()
            if P[rp_i].item() // BA == ch_i:
                break

        #rp_i = torch.randint(R*P_COL, (1,)).item()

        #rp_i = torch.argmax(score_rp).item()
        r_i = rp_i // P_COL
        p_i = rp_i % P_COL
        ch_i = P[rp_i].item() // BA
        ba_i = P[rp_i].item() % BA

        sum_col_ch[ch_i] = sum_col_ch[ch_i] - D_[r_i]
        sum_row_ch[ch_i][r_i] = sum_row_ch[ch_i][r_i] - 1
        
        # score of R x P_COL → CH assign
        for ch_j in range(CH):
            nnz_ch_, num_col_ch_, num_row_ch_ = assign_row(D_, P, sum_row_ch, sum_col_ch, ch_j, r_i)
            tot_ch_ = torch.clone(tot_ch)
            tot_ch_[ch_j] = nnz_ch_ + num_col_ch_ + num_row_ch_
            score_ch[ch_j] = max(tot_ch_).item()

        score_ch[ch_i] = 9999
        # to change : P, sum_col_ch, sum_row_ch
        #target_ch_ba = torch.randint(CH*BA, (1,)).item()
        target_ch_ba = torch.argmin(score_ch).item()*BA
        ch_tar = target_ch_ba // BA
        ba_tar = target_ch_ba % BA

        print("row", rp_i, " : ", ch_i, "   \t→\t", ch_tar)        
        P[rp_i] = target_ch_ba
        sum_col_ch[ch_tar] = sum_col_ch[ch_tar] + D_[r_i]
        sum_row_ch[ch_tar][r_i] = sum_row_ch[ch_tar][r_i] + 1
               
    return nnz_ch, num_col_ch, num_row_ch

def pop_push_rows(layer, num_ch_):
    D = layer  # [R, C]
    R = layer.shape[0]
    C = layer.shape[1]
    CH = num_ch_
    
    # Setting
    nnz_D = D.sum(dim=1)  # [R] 
    nnz_ch = torch.zeros(CH)
    num_col_ch = torch.zeros(CH)
    num_row_ch = torch.zeros(CH)
    col_ch = torch.zeros(CH, P_COL, COL_P)
    row_ch = torch.zeros(CH, R)
    
    # Random Initializing
    P = torch.randint(CH, (R,))  # [R]
    y = torch.zeros((R, CH))  # [R, CH]
    y[torch.arange(R), P] = 1.
    x = y.reshape(R, 1, CH) * D.reshape(R, C, 1)  # [R, C, CH]
    
    sum_col_ch = x.sum(dim=[0,4]).transpose(0,2).transpose(1,2) # [CH, C]
    sum_row_ch = y.sum(dim=[1,3]).transpose(0,1)   # [CH, R]

    for i in range(300000):
        nnz = 0
        for ch_i in range(CH):
            nnz, r_i = pop_1row(ch_i, nnz, nnz_D)
        

    return nnz_ch, num_col_ch, num_row_ch

def pop_1nnz(ch_i, nnz_idx_ch, nnz_ch, col_ch, row_ch, ROW, COL):
    nnz_to_pop = random.randint(0, nnz_ch[ch_i]-1)
    nnz_idx = int(nnz_idx_ch[ch_i][nnz_to_pop])
    nnz_ch[ch_i] = nnz_ch[ch_i] - 1
    col_ch[ch_i][COL[nnz_idx]] = col_ch[ch_i][COL[nnz_idx]] - 1
    row_ch[ch_i][ROW[nnz_idx]] = row_ch[ch_i][ROW[nnz_idx]] - 1
    nnz_idx_ch[ch_i] = torch.cat([torch.cat([nnz_idx_ch[ch_i][:nnz_to_pop], nnz_idx_ch[ch_i][nnz_to_pop+1:]]), torch.zeros(1)])
    
    return nnz_idx, nnz_idx_ch, nnz_ch, col_ch, row_ch

def push_1nnz(ch_i, ch_j, nnz_idx_ch, nnz_ch, col_ch, row_ch, ROW, COL, idx):
    ch_i = int(ch_i)
    ch_j = int(ch_j)
    nnz_idx = int(idx[ch_i])
    nnz_idx_ch[ch_j][int(nnz_ch[ch_j])] = nnz_idx
    nnz_ch[ch_j] = nnz_ch[ch_j] + 1
    col_ch[ch_j][COL[nnz_idx]] = col_ch[ch_j][COL[nnz_idx]] + 1
    row_ch[ch_j][ROW[nnz_idx]] = row_ch[ch_j][ROW[nnz_idx]] + 1
    return nnz_idx_ch, nnz_ch, col_ch, row_ch

def get_score(ch_i, ch_j, nnz_ch_, col_ch_, row_ch_, nnz_i, col_i, row_i):
    nnz_ch_j = torch.clone(nnz_ch_[ch_j])
    col_ch_j = torch.clone(col_ch_[ch_j])
    row_ch_j = torch.clone(row_ch_[ch_j])

    nnz_ch_j = nnz_ch_j + 1
    col_ch_j[col_i] = col_ch_j[col_i] + 1
    row_ch_j[row_i] = row_ch_j[row_i] + 1
    tot_ch = nnz_ch_j / 4 + torch.count_nonzero(col_ch_j) + torch.count_nonzero(row_ch_j)

    return tot_ch

def erase_max(score):
    CH = score.shape[0]
    score_ = torch.clone(score)  # [CH, CH]
    score_ = score_.view(-1)
    score_[torch.argmax(score_)] = -1
    return score_.view(CH, CH)

def non_erase_min(score, mask):
    CH = score.shape[0]
    mask_ = mask.clone().view(-1)
    score_ = score * (mask-0.5)
    min_idx = score_.argmin()
    mask_[min_idx] = 1
    return mask_.reshape(mask.shape)

def mask_init(score, mask):
    CH = score.shape[0]
    score_ = score.T
    mask_ = mask.T
 
    for ch_j in range(CH):
        mask_[ch_j] = non_erase_min(score_[ch_j], mask_[ch_j])
    mask = mask_.T

    return mask

def available_ch_ch(ch_i, ch_j, mask, ch_idx, remain_ch_j):
    CH = mask.shape[0]
   
    if ch_idx[ch_j] != -1:
        # ch_j is already used!
        return False

    ch_idx[ch_j] = ch_i
    # put success!

    if ch_i+1 == CH:
        return True
        
    for ch_j_ in remain_ch_j[ch_i+1]:
        if available_ch_ch(ch_i+1, ch_j_, mask, ch_idx, remain_ch_j):
            return True
    ch_idx[ch_j] = -1
    return False

def available(mask, ch_idx):
    CH = mask.shape[0]
    
    remain_ch_j = []
    for ch_i in range(CH):
        remain_ch_j.append([])

    for ch_i in range(CH):
        for ch_j in range(CH):
            if mask[ch_i][ch_j] == 1:
                remain_ch_j[ch_i].append(ch_j)

    ch_i = 0
    for ch_j in remain_ch_j[0]:
        if available_ch_ch(ch_i, ch_j, mask, ch_idx, remain_ch_j):
            return True
    return False

def pop_push_nnzs(layer, num_ch_):
    D = layer  # [R, C]
    R = layer.shape[0]
    C = layer.shape[1]
    NNZs = int(R*C*(1-sparsity))
    CH = num_ch_
    ROW = D.view(-1).nonzero().view(-1) // C  # [NNZs]
    COL = D.view(-1).nonzero().view(-1) % C   # [NNZs]

    print("ROW:", ROW)
    print("COL:", COL)
    
    # Setting
    nnz_ch = torch.zeros(CH)
    nnz_idx_ch = torch.zeros(CH, NNZs)
    num_col_ch = torch.zeros(CH)
    num_row_ch = torch.zeros(CH)
    col_ch = torch.zeros(CH, C)
    row_ch = torch.zeros(CH, R)
    
    # Random Initializing
    if 0:
        P = torch.randint(CH, (NNZs,))  # [NNZs]
    else:
        # Start from grad_nnz's optimal point
        P = torch.load('nnz'+str(sparsity)+'.pt')  # [R, C, CH]
        P = P.nonzero() # [NNZs, 3]
        idx = torch.Tensor([2]).to(torch.int32)
        P = P.index_select(1, idx).reshape(-1)  # [NNZs]

    for nnz_i in range(NNZs):
        ch_i = P[nnz_i]
        nnz_idx_ch[ch_i][int(nnz_ch[ch_i])] = nnz_i
        nnz_ch[ch_i] = nnz_ch[ch_i] + 1
        col_ch[ch_i][COL[nnz_i]] = col_ch[ch_i][COL[nnz_i]] + 1
        row_ch[ch_i][ROW[nnz_i]] = row_ch[ch_i][ROW[nnz_i]] + 1
    num_col_ch = col_ch.count_nonzero(dim=1)
    num_row_ch = row_ch.count_nonzero(dim=1)
    tot_ch = nnz_ch / 4 + num_col_ch + num_row_ch

    print(num_col_ch)
    print(num_row_ch)

    score = torch.zeros((CH,CH))
    idx_ = torch.zeros(CH+1)
    for i in range(1000000000):
        # Pop CH+1 nnz
        ch_idx_2nnz = random.randint(0, CH-1)
        for ch_i in range(CH):
            if ch_i == ch_idx_2nnz:
                nnz_i1, nnz_idx_ch, nnz_ch, col_ch, row_ch = pop_1nnz(ch_i, nnz_idx_ch, nnz_ch, col_ch, row_ch, ROW, COL)
                nnz_i2, nnz_idx_ch, nnz_ch, col_ch, row_ch = pop_1nnz(ch_i, nnz_idx_ch, nnz_ch, col_ch, row_ch, ROW, COL)
                idx_[ch_i] = nnz_i1
                idx_[CH] = nnz_i2
            else:
                nnz_i, nnz_idx_ch, nnz_ch, col_ch, row_ch = pop_1nnz(ch_i, nnz_idx_ch, nnz_ch, col_ch, row_ch, ROW, COL)
                idx_[ch_i] = nnz_i
        
        # Calculate Score
        for ch_i in range(CH):
            for ch_j in range(CH):
                if ch_i == ch_idx_2nnz:
                    score[ch_i][ch_j] = get_score(ch_i, ch_j, nnz_ch, col_ch, row_ch, 1, COL[int(idx_[ch_i])], ROW[int(idx_[ch_i])])
                    score[ch_i][ch_j] = get_score(CH, ch_j, nnz_ch, col_ch, row_ch, 1, COL[int(idx_[CH])], ROW[int(idx_[CH])])
                else:
                    score[ch_i][ch_j] = get_score(ch_i, ch_j, nnz_ch, col_ch, row_ch, 1, COL[int(idx_[ch_i])], ROW[int(idx_[ch_i])])

        print(score)
        # Erase max until it is available
        itr = 1
        mask = (score >= tot_ch.max().item())*1.
        mask_test1 = mask.sum(0).prod()
        mask_test2 = mask.sum(0).prod()
        find_flag = 1
        if mask_test1 == 0 or mask_test2 == 0:
            find_flag = 0
        
        while(find_flag):
            #print("iter : ", itr)
            #print(mask)
            ch_idx = torch.zeros(CH).fill_(-1)
            if available(mask, ch_idx) == False:
                itr = itr + 1
                mask = non_erase_min(score, mask)
            else:
                break
        
        score = score * mask
        print("check:", score.max().item(), "\t", tot_ch.max().item()) 

        if score.max() > tot_ch.max():
            # don't assign, just run again
            ch_idx = torch.arange(CH)
            print(i, "\tbad.. reset\tch_j[ch_i] : ", end="")
        else:
            print(i, "\terased:", itr, "\tch_j[ch_i] : ", end="")
        print_("ch_idx", ch_idx.int())

        # Push CH+1 nnz
        
        for ch_j in range(CH):
            if ch_idx[ch_j] == ch_idx_2nnz:
                nnz_idx_ch, nnz_ch, col_ch, row_ch = push_1nnz(int(ch_idx[ch_j]), int(ch_j), nnz_idx_ch,
                                                               nnz_ch, col_ch, row_ch, ROW, COL, idx_)
                nnz_idx_ch, nnz_ch, col_ch, row_ch = push_1nnz(CH, int(ch_j), nnz_idx_ch, nnz_ch, col_ch,
                                                               row_ch, ROW, COL, idx_)
            else: 
                nnz_idx_ch, nnz_ch, col_ch, row_ch = push_1nnz(ch_idx[ch_j], ch_j, nnz_idx_ch, nnz_ch, col_ch, row_ch, ROW, COL, idx_)
            
        num_col_ch = torch.count_nonzero(col_ch, dim=1)
        num_row_ch = torch.count_nonzero(row_ch, dim=1)
        tot_ch = nnz_ch // 4 + num_col_ch + num_row_ch
        print(i, "\t", sum(tot_ch).item(), "\t", max(tot_ch).item(), "\t", max(nnz_ch//4).item(), "\t",
              max(num_col_ch).item(), "\t", max(num_row_ch).item())
        print_("tot_ch", tot_ch)
        print_("nnz_ch", nnz_ch)
        print_("num_col_ch", num_col_ch)
        print_("num_row_ch", num_row_ch)

    return nnz_ch, num_col_ch, num_row_ch


def k_means(layer, num_ch_):
    D = layer  # [R, C]
    R = layer.shape[0]
    C = layer.shape[1]
    CH = num_ch

    G = CH
    RG = torch.zeros((R, CH))
    R_nnz = D.sum(dim=1)
    for r in range(R):
        RG[r, r // ((R+CH-1)//CH)] = 1.0

    for i in range(100):
        RR_intersection = torch.matmul(D, D.T)
        #RR_union = torch.count_nonzero(D.view(R, C, 1) + D.T.view(1, C, R), dim=1)
        RR_nnz1 = R_nnz.view(1,R).repeat((R, 1))
        RR_nnz2 = R_nnz.view(1,R).T.repeat((1,R))
        RR_min = torch.minimum(RR_nnz1, RR_nnz2)

        #RR_distance = 1 - RR_intersection / RR_union
        RR_distance = 1 - RR_intersection / RR_min
        RG_distance = torch.sum(RR_distance.view(R, R, 1) * RG.view(R, 1, G), dim=1) / torch.sum(RG, dim=0, keepdim=True)  # [R, G]

        #new_group_idx = torch.argmin(RG_distance, dim=1)
        #RG.fill_(0.)
        #RG[range(R), new_group_idx] = 1.0

        RG.fill_(0.)
        group_min, group_idx = torch.min(RG_distance, dim=1) # [R]
        sorted_row_idx = torch.argsort(group_min)
        nnz_ch = torch.zeros(CH)
        for r in sorted_row_idx:
            flag = True
            idx = 0
            while nnz_ch[group_idx[r]] + R_nnz[r] >= nnz_per_cluster_h:
                if flag:
                    sorted_group_idx = torch.argsort(RG_distance[r])
                    flag = False
                idx += 1
                group_idx[r] = sorted_group_idx[idx]
            RG[r, group_idx[r]] = 1.0
            nnz_ch[group_idx[r]] += R_nnz[r]

        RCG = D.view(R, C, 1) * RG.view(R, 1, G)
        nnz_ch = RCG.sum(dim=[0, 1])
        col_ch = torch.count_nonzero(RCG.sum(dim=0), dim=0)
        row_ch = RG.sum(dim=0)

        print("***", i, "\t", "*"*50)
        print(nnz_ch)
        print(col_ch)
        print(row_ch)
        print((nnz_ch/4).int()+col_ch+row_ch)

    nnz_ch = nnz_ch/4

    return nnz_ch, col_ch, row_ch
    

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
    centroid = torch.zeros((k, num_col)) - 1
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
    col_k = torch.zeros((k, num_col))
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
        layer[torch.argmin(tmp).item()] = torch.zeros(num_col) - 1
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
    layer[row_i] = torch.zeros(num_col) - 1
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
    centroid,_,_,_= w_b_k_means(layer, sparsity, nnz, k, nnz_k, col_k, row_k, bound_ratio)

    layer_ = torch.clone(layer)

    nnz_k = torch.zeros(k)
    col_k = torch.zeros((k, num_col))
    row_k = torch.zeros(k)

    print("centroid : \n", centroid)
    KRows_idx = SelectKRows(layer_, centroid)
    print(KRows_idx)
    for k_idx in range(k):
        nnz_k[k_idx] = torch.count_nonzero(layer[int(KRows_idx[k_idx])]).item()
        col_k[k_idx] = layer[int(KRows_idx[k_idx])]
        row_k[k_idx] = 1
        dns_k[k_idx] = layer[int(KRows_idx[k_idx])]
        layer_[int(KRows_idx[k_idx])] = torch.zeros(num_col) - 1
    print("< Selecting k rows Ended! >\n")

def main():
    InitScore(layer_, nnz_k, col_k, row_k, dns_k)
    for i in range(num_row - k):
        print("iter : ", i+1, end='\t')
        layer_, nnz_k, col_k, row_k, dns_k = Assign1Row(layer_, nnz_k, col_k, row_k, dns_k)
    InitScore(layer_, nnz_k, col_k, row_k, dns_k)

if __name__ == "__main__":
    main()
