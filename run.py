import torch
import torchvision.models as models
import cvxpy as cvp
import numpy as np
import random
import time
import argparse
import spmv
import os

########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--ver', required=False, default='soft_split', help='')
parser.add_argument('--algorithm', required=False, default='grad_row', help='')
parser.add_argument('--model', required=False, default='resnet', help='')
parser.add_argument('--num_ch', required=False, default=16, type=int, help='')
parser.add_argument('--num_ba', required=False, default=4, type=int, help='')
parser.add_argument('--part_col', required=False, default=16, type=int, help='')
parser.add_argument('--col_first', required=False, default=0, help='')
parser.add_argument('--gs', required=False, default=1, help='')
parser.add_argument('--gs_hard', required=False, default=False, type=bool, help='')
parser.add_argument('--sparsity', required=False, default=0.95, type=float, help='')
parser.add_argument('--register_size', required=False, default=128, type=int, help='')
parser.add_argument('--bound_ratio', required=False, default=0.02, type=float, help='')
parser.add_argument('--device', required=False, default='cuda', help='')
parser.add_argument('--num_iter', required=False, default=4000, type=int, help='')
parser.add_argument('--lr_init', required=False, default=100, type=float, help='')
parser.add_argument('--w_decay', required=False, default=0.015, type=float, help='')
parser.add_argument('--print_option', required=False, default=3, help='')
args = parser.parse_args()
# Algorithm Option
# 0 :  sequence_row_equal
# 1 :  sequence_row_threshold
# 2 :  sequence_nnz_threshold
# 3 :  space_a
# 4 :  grad_row
# 6 :  new_loss
# 6 :  k_means
# 7 :  pop_push_1row
# 8 :  pop_push_rows
# 9 :  pop_push_nnzs
# 10 : grad_nnz
# 11 : grad_row_lowmem
# 12 : grad_row_register_size
# 13 : space_a_register_size
# 14 : space_a_register_size_withrow
########################################################################
torch.set_default_dtype(torch.float32)

print("model loading...")
if args.model == 'resnet':
    model = models.resnet50(pretrained=True)
    layer = model.fc.weight
elif args.model == 'ds2_0':
    layer = torch.load('deepspeech2/ds2_0.pt')
elif args.model == 'ds2_1':
    layer = torch.load('deepspeech2/ds2_1.pt')
elif args.model == 'ds2_2':
    layer = torch.load('deepspeech2/ds2_2.pt')
elif args.model == 'ds2_3':
    layer = torch.load('deepspeech2/ds2_3.pt')
elif args.model == 'ds2_4':
    layer = torch.load('deepspeech2/ds2_4.pt')
elif args.model == 'ds2_0r':
    layer = torch.load('deepspeech2/ds2_0r.pt')
elif args.model == 'ds2_1r':
    layer = torch.load('deepspeech2/ds2_1r.pt')
elif args.model == 'ds2_2r':
    layer = torch.load('deepspeech2/ds2_2r.pt')
elif args.model == 'ds2_3r':
    layer = torch.load('deepspeech2/ds2_3r.pt')
elif args.model == 'ds2_4r':
    layer = torch.load('deepspeech2/ds2_4r.pt')

elif args.model == 'gnmt_dec_0':
    layer = torch.load('gnmt/gnmt_dec_0.pt')
elif args.model == 'gnmt_dec_1':
    layer = torch.load('gnmt/gnmt_dec_1.pt')
elif args.model == 'gnmt_dec_2':
    layer = torch.load('gnmt/gnmt_dec_2.pt')
elif args.model == 'gnmt_enc_0':
    layer = torch.load('gnmt/gnmt_enc_0.pt')
elif args.model == 'gnmt_enc_0r':
    layer = torch.load('gnmt/gnmt_enc_0r.pt')
elif args.model == 'gnmt_enc_1':
    layer = torch.load('gnmt/gnmt_enc_1.pt')
elif args.model == 'gnmt_enc_2':
    layer = torch.load('gnmt/gnmt_enc_2.pt')
elif args.model == 'gnmt_enc_3':
    layer = torch.load('gnmt/gnmt_enc_3.pt')

elif args.model == 'bcsstk32':
    layer = torch.load('sci_model/csstk32.pt')
elif args.model == 'cant':
    layer = torch.load('sci_model/cant.pt')
elif args.model == 'consph':
    layer = torch.load('sci_model/consph.pt')
elif args.model == 'pdb1HYS':
    layer = torch.load('sci_model/pdb1HYS.pt')
elif args.model == 'webbase':
    layer = torch.load('sci_model/webbase-1M.pt')
elif args.model == 'test':
    layer = torch.load('test.pt')
else:
    print("no model found!")
print("model loaded!")

lr = [80, 40, 20, 10, 5, 3, 1, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0007]
wd = [0.015, 0.05, 0.1, 0.2, 0.35, 0.4, 0.6, 0.8, 1, 3, 6, 10, 15]

algorithm = args.algorithm
num_row = layer.shape[0]
num_col = layer.shape[1]
num_ch = int(args.num_ch)
num_ba = int(args.num_ba)
gs = int(args.gs)
sparsity = float(args.sparsity)
args.sparsity = float(args.sparsity)
register_size = int(args.register_size)
bound_ratio = float(args.bound_ratio)
k = num_ch * num_ba
part_col = int(args.part_col)
num_ch_row = int(num_ch / part_col)
device = args.device
lr_init = float(args.lr_init)
weight_decay = float(args.w_decay)
file_name = "./ckpt/" + args.model + "_alg" + args.algorithm + "_sp" + str(args.sparsity) + "_ch" + str(args.num_ch) + "_ba" + str(args.num_ba) + "_reg" + str(args.register_size) + ".pt"

if sparsity != 0:  # ML Application
    #spmv.init(layer, sparsity, num_ch, num_ba, part_col, bound_ratio, device)
    layer = spmv.prune_layer(layer, sparsity)
else:  # Scientific Dataset
    sparsity = 1 - torch.count_nonzero(layer) / (num_row * num_col)
    print("nnzs : ", torch.count_nonzero(layer))
    print("sparsity: ", sparsity)
    spmv.init(layer, sparsity, num_ch, num_ba, part_col, bound_ratio, device)
args.layer = layer
########################################################################

nnz_ch = torch.zeros((num_ch))
col_ch = torch.zeros((num_ch))
row_ch = torch.zeros((num_ch))
P = torch.zeros((num_row, num_ch, num_ba))

print("algorithm : ", algorithm)

if args.ver == "hard_split":
    layer = spmv.hard_split(layer, part_col)  # [part_col, R, C//part_col]
    for part_i in range(part_col):
        if algorithm == "sequence_row_equal" or algorithm == "0":
            nnz_ch_, col_ch_, row_ch_ = spmv.sequence_row_equal(layer[part_i], num_ch_row)
        elif algorithm == "sequence_row_threshold" or algorithm == "1":
            nnz_ch_, col_ch_, row_ch_ = spmv.sequence_row_threshold(layer[part_i], num_ch_row)
        elif algorithm == "sequence_nnz_threshold" or algorithm == "2":
            nnz_ch_, col_ch_, row_ch_ = spmv.sequence_nnz_threshold(layer[part_i], num_ch_row)
        elif algorithm == "space_a" or algorithm == "3":
            nnz_ch_, col_ch_, row_ch_ = spmv.space_a(layer[part_i], num_ch_row)
        elif algorithm == "grad_row" or algorithm == "4":
            nnz_ch_, col_ch_, row_ch_ = spmv.grad_row_hard(layer[part_i], num_ch_row)
        elif algorithm == "new_loss" or algorithm == "5":
            nnz_ch_, col_ch_, row_ch_ = spmv.new_loss(layer[part_i], num_ch_row)

        offset = part_i*num_ch_row
        nnz_ch[offset:offset+num_ch_row] = nnz_ch_
        col_ch[offset:offset+num_ch_row] = col_ch_
        row_ch[offset:offset+num_ch_row] = row_ch_

elif args.ver == "soft_split":
    layer = spmv.hard_split(layer, part_col)  # [part_col, R, C//part_col]
    if args.col_first == "1":
        layer = torch.transpose(layer, 0, 1)
    if algorithm == "sequence_row_equal" or algorithm == "0":
        nnz_ch, col_ch, row_ch = spmv.sequence_row_equal(layer, num_ch)
    elif algorithm == "sequence_row_threshold" or algorithm == "1":
        nnz_ch, col_ch, row_ch = spmv.sequence_row_threshold(layer, num_ch)
    elif algorithm == "sequence_nnz_threshold" or algorithm == "2":
        nnz_ch, col_ch, row_ch = spmv.sequence_nnz_threshold(layer, num_ch)
    elif algorithm == "space_a" or algorithm == "3":
        nnz_ch, col_ch, row_ch = spmv.space_a(layer, num_ch, register_size, device)
    elif algorithm == "grad_row" or algorithm == "4":
        nnz_ch, col_ch, row_ch = spmv.grad_row_soft(layer, num_ch)
    elif algorithm == "pop_push_1row" or algorithm == "7":
        nnz_ch, col_ch, row_ch = spmv.pop_push_1row(layer, num_ch)
elif args.ver == "no_split":
    if algorithm == "k_means" or algorithm == "6":
        nnz_ch, col_ch, row_ch = spmv.k_means(layer, num_ch)
    elif algorithm == "pop_push_rows" or algorithm == "8":
        nnz_ch, col_ch, row_ch = spmv.pop_push_rows(layer, num_ch)
    elif algorithm == "pop_push_nnzs" or algorithm == "9":
        nnz_ch, col_ch, row_ch = spmv.pop_push_nnzs(layer, num_ch)
    elif algorithm == "grad_nnz" or algorithm == "10":
        _, nnz_ch, col_ch, row_ch = spmv.grad_nnz(layer, num_ch)
    elif algorithm == "grad_row_lowmem" or algorithm == "11":
        nnz_ch, col_ch, row_ch = spmv.grad_row_lowmem(layer, num_ch, device)
    elif algorithm == "grad_row_register_size" or algorithm == "12":
        P, nnz_ch, col_ch, row_ch = spmv.grad_row_register_size(layer, args)
    elif algorithm == "space_a_register_size" or algorithm == "13":
        P, nnz_ch, col_ch, row_ch = spmv.space_a_register_size(layer, args)
    elif algorithm == "space_a_register_size_withrow" or algorithm == "14":
        P, nnz_ch, col_ch, row_ch = spmv.space_a_register_size_withrow(layer, args)
    elif algorithm == "sequence_row_threshold_register_size" or algorithm == "15":
        P, nnz_ch, col_ch, row_ch = spmv.sequence_row_threshold_register_size(layer, args)
    elif algorithm == "test_loss" or algorithm == "16":
        P, nnz_ch, col_ch, row_ch = spmv.test_loss(layer, args)
    elif algorithm == "grad_row_no_register_size" or algorithm == "17":
        P, nnz_ch, col_ch, row_ch = spmv.grad_row_no_register_size(layer, args)
elif args.ver == "by_cmd":
    if algorithm == "grad_row_register_size" or algorithm == "18":
        P, nnz_ch, col_ch, row_ch = spmv.grad_row_register_size_cmd(layer, args)
    elif algorithm == "space_a_cmd" or algorithm == "19":
        P, nnz_ch, col_ch, row_ch = spmv.space_a_cmd(layer, args)
    elif algorithm == "space_a_register_size_withrow" or algorithm == "20":
        P, nnz_ch, col_ch, row_ch = spmv.space_a_withrow_cmd(layer, args)
    elif algorithm == "sequence_row_threshold_cmd" or algorithm == "21":
        P, nnz_ch, col_ch, row_ch = spmv.sequence_row_threshold_cmd(layer, args)
    elif algorithm == "grad_row_no_register_size" or algorithm == "22":
        P, nnz_ch, col_ch, row_ch = spmv.grad_row_no_register_size_cmd(layer, args)
elif args.ver == "test_split":
    if algorithm == "grad_row_register_size" or algorithm == "12":
        best_case = 1000000
        for i in range(len(lr)):
            print("start hyperparameter [",i,"]")
            args.lr_init = lr[i]
            args.w_decay = wd[i]
            P_, nnz_ch_, col_ch_, row_ch_ = spmv.grad_row_register_size(layer, args)
            if torch.max(nnz_ch_ + col_ch_ + row_ch_).item() < best_case:
                best_case = torch.max(nnz_ch_ + col_ch_ + row_ch_).item()
                nnz_ch = nnz_ch_
                col_ch = col_ch_
                row_ch = row_ch_
                P = P_
    elif algorithm == "grad_row_no_register_size" or algorithm == "17":
        best_case = 1000000
        for i in range(len(lr)):
            print("start hyperparameter [",i,"]")
            args.lr_init = lr[i]
            args.w_decay = wd[i]
            P_, nnz_ch_, col_ch_, row_ch_ = spmv.grad_row_no_register_size(layer, args)
            if torch.max(nnz_ch_ + col_ch_ + row_ch_).item() < best_case:
                best_case = torch.max(nnz_ch_ + col_ch_ + row_ch_).item()
                nnz_ch = nnz_ch_
                col_ch = col_ch_
                row_ch = row_ch_
                P = P_
elif args.ver == "load_checkpoint":
    P = torch.load(file_name)
elif args.ver == "test":
    print("!!!test!!!")
    nnz = layer.sum(1).sort().values
    np_nnz = nnz.numpy()
    np.save('./graph/nnz_'+args.model+'_'+args.sparsity+'.npy', np_nnz)
elif args.ver == "test_result":
    #models = ['ds2_0r', 'ds2_1', 'ds2_2r', 'ds2_3r', 'ds2_4']  # → Ended!
    #models = ['gnmt_enc_0', 'gnmt_enc_0r', 'gnmt_enc_1', 'gnmt_enc_2', 'gnmt_enc_3']  # Ended!
    models = ['gnmt_dec_0', 'gnmt_dec_1', 'gnmt_dec_2']  # Ended!
    #models = ['gnmt_enc_0', 'gnmt_enc_0r', 'gnmt_enc_1']  # Ended!
    #models = ['gnmt_dec_0', 'gnmt_dec_1', 'gnmt_dec_2', 'gnmt_enc_0r', 'gnmt_enc_3']  # Ended!
    #models = ['ds2_0', 'ds2_1r', 'ds2_2', 'ds2_3', 'ds2_4r']  # → Not yet
    #models = ['gnmt_enc_2'] # → Ended!
    algorithms = [15, 13, 14, 17, 12]
    sparsitys = [0.9, 0.8, 0.7, 0.6]
    register_sizes = [64, 128]
   
    args.print_option = 4
    for args.model in models:
        if args.model == 'ds2_0':
            layer = torch.load('deepspeech2/ds2_0.pt')
        elif args.model == 'ds2_1':
            layer = torch.load('deepspeech2/ds2_1.pt')
        elif args.model == 'ds2_2':
            layer = torch.load('deepspeech2/ds2_2.pt')
        elif args.model == 'ds2_3':
            layer = torch.load('deepspeech2/ds2_3.pt')
        elif args.model == 'ds2_4':
            layer = torch.load('deepspeech2/ds2_4.pt')
        elif args.model == 'ds2_0r':
            layer = torch.load('deepspeech2/ds2_0r.pt')
        elif args.model == 'ds2_1r':
            layer = torch.load('deepspeech2/ds2_1r.pt')
        elif args.model == 'ds2_2r':
            layer = torch.load('deepspeech2/ds2_2r.pt')
        elif args.model == 'ds2_3r':
            layer = torch.load('deepspeech2/ds2_3r.pt')
        elif args.model == 'ds2_4r':
            layer = torch.load('deepspeech2/ds2_4r.pt')

        elif args.model == 'gnmt_dec_0':
            layer = torch.load('gnmt/gnmt_dec_0.pt')
        elif args.model == 'gnmt_dec_1':
            layer = torch.load('gnmt/gnmt_dec_1.pt')
        elif args.model == 'gnmt_dec_2':
            layer = torch.load('gnmt/gnmt_dec_2.pt')
        elif args.model == 'gnmt_enc_0':
            layer = torch.load('gnmt/gnmt_enc_0.pt')
        elif args.model == 'gnmt_enc_0r':
            layer = torch.load('gnmt/gnmt_enc_0r.pt')
        elif args.model == 'gnmt_enc_1':
            layer = torch.load('gnmt/gnmt_enc_1.pt')
        elif args.model == 'gnmt_enc_2':
            layer = torch.load('gnmt/gnmt_enc_2.pt')
        elif args.model == 'gnmt_enc_3':
            layer = torch.load('gnmt/gnmt_enc_3.pt')

        for args.register_size in register_sizes:
            for args.sparsity in sparsitys:
                args.layer = spmv.prune_layer(layer, args.sparsity)
                print(args.model, args.register_size, args.sparsity, end="\t")
                spmv.init(args.layer, args.sparsity, num_ch, num_ba, part_col, bound_ratio, device)
                for args.algorithm in algorithms:
                    file_name = "./ckpt/" + args.model + "_alg" + str(args.algorithm) + "_sp" + str(args.sparsity) + "_ch" + str(args.num_ch) + "_ba" + str(args.num_ba) + "_reg" + str(args.register_size) + ".pt"
                    P = torch.load(file_name).detach()
                    cost = spmv.print_specific(P, args)
                    print(cost, end="\t")
                print()

P = P.detach()
spmv.print_specific(P, args)

args.print_option = 0 + 10
print("new optimized cost!")
new_cost = spmv.print_specific(P, args)

if os.path.exists(file_name):
    P_ = torch.load(file_name).detach()
    print("old optimized cost...")
    old_cost = spmv.print_specific(P_, args)
    if new_cost < old_cost:
        print("save new optimized cost!!!")
        torch.save(P, file_name)
    else:
        print("old cost is better than this")

else:
    print("save new optimized cost!!!")
    torch.save(P, file_name)
