import os
import argparse
from utils import *
from tqdm import tqdm
from torch import optim
from model import SEGC
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--sigma', type=float, default=0.01, help='Sigma of gaussian distribution')
parser.add_argument('--dataset', type=str, default='citeseer', help='type of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='type of dataset.')
parser.add_argument('--device', type=str, default='cuda:0', help='device')

args = parser.parse_args()


for args.dataset in ["cora", "citeseer", "amap", "bat", "eat", "uat"]:
    print("Using {} dataset".format(args.dataset))
    file = open("result_baseline.csv", "a+")
    print(args.dataset, file=file)
    file.close()

    if args.dataset == 'cora':
        args.cluster_num = 7
        args.gnnlayers = 3
        args.lr = 1e-3
        args.dims = [500]
    elif args.dataset == 'citeseer':
        args.cluster_num = 6
        args.gnnlayers = 2
        args.lr = 5e-5
        args.dims = [500]
    elif args.dataset == 'amap':
        args.cluster_num = 8
        args.gnnlayers = 5
        args.lr = 1e-5
        args.dims = [500]
    elif args.dataset == 'bat':
        args.cluster_num = 4
        args.gnnlayers = 3
        args.lr = 1e-3
        args.dims = [500]
    elif args.dataset == 'eat':
        args.cluster_num = 4
        args.gnnlayers = 5
        args.lr = 1e-3
        args.dims = [500]
    elif args.dataset == 'uat':
        args.cluster_num = 4
        args.gnnlayers = 3
        args.lr = 1e-3
        args.dims = [500]
    elif args.dataset == 'corafull':
        args.cluster_num = 70
        args.gnnlayers = 2
        args.lr = 1e-3
        args.dims = [500]
    X, y, A = load_graph_data(args.dataset, show_details=False)
    features = X
    true_labels = y
    adj = sp.csr_matrix(A)