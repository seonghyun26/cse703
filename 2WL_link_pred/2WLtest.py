from math import e
from scipy.sparse import data
from sklearn import utils
import random
import numpy as np
from model import LocalWLNet, WLNet, FWLNet, LocalFWLNet, Net_cora
from datasets import load_dataset, dataset
from impl import train
import torch
from torch.optim import Adam
from ogb.linkproppred import Evaluator
import yaml

import os

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate_hits(pos_pred, neg_pred, K):
    results = {}
    evaluator = Evaluator(name='ogbl-collab')
    evaluator.K = K
    hits = evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })[f'hits@{K}']

    results[f'Hits@{K}'] = hits

    return results

def testparam(datetime, seed, device="cpu", dsname="Celegans"):  # mod_params=(32, 2, 1, 0.0), lr=3e-4
    device = torch.device(device)
    bg = load_dataset(dsname, args.pattern, seed, args.instanceSize)
    
    # print(bg)
    # exit(0)
    
    bg.to(device)
    bg.preprocess()
    bg.setPosDegreeFeature()
    max_degree = torch.max(bg.x[2])

    trn_ds = dataset(*bg.split(0))
    val_ds = dataset(*bg.split(1))
    tst_ds = dataset(*bg.split(2))
    # How is tst_ds made?
    if trn_ds.na != None:
        print("use node feature")
        trn_ds.na = trn_ds.na.to(device)
        val_ds.na = val_ds.na.to(device)
        tst_ds.na = tst_ds.na.to(device)
        use_node_attr = True
    else:
        use_node_attr = False


    def valparam(datetime, seed, **kwargs):
        lr = kwargs.pop('lr')
        epoch = kwargs.pop('epoch')
        if args.pattern == '2wl':
            mod = WLNet(max_degree, use_node_attr, trn_ds.na, **kwargs).to(device)
        elif args.pattern == '2wl_l':
            mod = LocalWLNet(max_degree, use_node_attr, trn_ds.na, **kwargs).to(device)
        elif args.pattern == '2fwl':
            mod = FWLNet(max_degree, use_node_attr, trn_ds.na, **kwargs).to(device)
        elif args.pattern == '2fwl_l':
            mod = LocalFWLNet(max_degree, use_node_attr, trn_ds.na, **kwargs).to(device)
        opt = Adam(mod.parameters(), lr=lr)
        return train.train_routine(args.dataset, mod, opt, trn_ds, val_ds, tst_ds, epoch, datetime, seed, args.instanceSize, verbose=True)

    with open(f"config/{args.pattern}/{args.dataset}.yaml") as f:
        params = yaml.safe_load(f)

    if seed == 0:
        valparam(datetime, seed, **(params))
    else:
        mod = torch.load(os.getcwd() + "/checkpoint/"+datetime+"/2WL_dict_"+ args.instanceSize + "_"+str(seed-1)+".pt")
        mod.load_state_dict(torch.load(os.getcwd() + "/checkpoint/"+datetime+"/2WL_state_dict_"+ args.instanceSize + "_" + str(seed-1)+".pt"))
        lr = params.pop('lr')
        epoch = params.pop('epoch')
        opt = Adam(mod.parameters(), lr=lr)
        train.train_routine(args.dataset, mod, opt, trn_ds, val_ds, tst_ds, epoch, datetime, seed, args.instanceSize, verbose=True)
        


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pattern', type=str, default="2wl_l")
    parser.add_argument('--dataset', type=str, default="Cora")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--path', type=str, default="opt/")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--check', action="store_true")
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--subgraph', type=str, default="original")
    parser.add_argument('--name', type=str, default='jssp')
    parser.add_argument('--instanceSize', type=str, default='30_20')
    # path, cycle, incoming, outgoing
    
    
    args = parser.parse_args()
    # print(args.dataset)
    print(args.instanceSize)
    
    if args.device < 0:
        args.device = "cpu"
    else:
        args.device = "cuda:" + str(args.device)
    started = datetime.now()
    dt = datetime.today().strftime("%Y%m%d_%H%M%S")
    os.mkdir(os.getcwd() + "/checkpoint/" + dt)
    for i in range(10):
        set_seed(i + args.seed)
        print(">>>> " + str(i+1) + "/10 test <<<<")
        testparam(dt, i, args.device, args.dataset)
        