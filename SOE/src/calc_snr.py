import os
import sys
import glob
import time
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from easydict import EasyDict as edict

from utils.training import set_seed, plot_history_v2

def calc_snr(cfg, args):
    # set up device
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert torch.cuda.is_available()

    # dataset
    print("Loading dataset ...")
    if cfg.dataset.name == "realworld":
        from dataset.realworld import RealWorldDataset, collate_fn, pre_process_data
        dataset = RealWorldDataset(**cfg.dataset.params)
    elif cfg.dataset.name == "robomimic":
        from dataset.robomimic_v2 import RoboMimicDataset, collate_fn, pre_process_data
        dataset = RoboMimicDataset(**cfg.dataset.params)
    elif cfg.dataset.name == "robomimic_old":
        from dataset.robomimic import RoboMimicDataset, collate_fn, pre_process_data
        dataset = RoboMimicDataset(**cfg.dataset.params)
    else:
        raise NotImplementedError
    print("dataset size:", len(dataset))
    
    # sampler
    num_samples = len(dataset)
    # random sampler
    sampler = torch.utils.data.RandomSampler(
        dataset, 
        replacement = False,  # make sure all data are used before some are reused
        num_samples = num_samples
    )

    # dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = cfg.batch_size,
        num_workers = cfg.num_workers,
        collate_fn = collate_fn,
        sampler = sampler
    )

    # load policy
    print("Loading policy ...")
    if cfg.policy.name == "DP":
        from policy.dp import DP
        policy = DP(**cfg.policy.params).to(device)
    elif cfg.policy.name == "BC":
        from policy.bc import get_bc
        policy = get_bc(cfg.policy.params.algo)(**cfg.policy.params).to(device)
    elif cfg.policy.name == "ResBC":
        from policy.residual_bc import ResBC
        policy = ResBC(**cfg.policy.params).to(device)
    elif cfg.policy.name == "AE":
        from policy.ae import AE
        policy = AE(**cfg.policy.params).to(device)
    elif cfg.policy.name == "VAE":
        from policy.vae import VAE
        policy = VAE(**cfg.policy.params).to(device)
    elif cfg.policy.name == "MIDP":
        from policy.mi_dp_v2 import MIDP
        policy = MIDP(**cfg.policy.params).to(device)
    elif cfg.policy.name == "LDP":
        from policy.ldp import LDP
        policy = LDP(**cfg.policy.params).to(device)
    elif cfg.policy.name == "CVAE":
        from policy.cvae import CVAE
        policy = CVAE(**cfg.policy.params).to(device)
    elif cfg.policy.name == "DPExt":
        from policy.dp_ext import DPExt
        policy = DPExt(**cfg.policy.params).to(device)
    else:
        raise NotImplementedError
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

    # load checkpoint
    assert args.ckpt is not None
    policy.load_state_dict(torch.load(args.ckpt, map_location = device), strict = False)
    
    policy.eval()
    pbar = tqdm(dataloader)
    mu_list = []
    var_list = []
    with torch.no_grad():
        for data in pbar:
            batch = pre_process_data(data, device, cfg)
            mu, logvar = policy.get_latent_action(**batch)
            mu_list.append(mu.detach().cpu().numpy())
            var_list.append(torch.exp(logvar).detach().cpu().numpy())
    mu = np.concatenate(mu_list, axis = 0)
    var = np.concatenate(var_list, axis = 0)
    print("mu shape:", mu.shape)
    print("var shape:", var.shape)
    var_of_mu = np.var(mu, axis = 0)
    mu_of_var = np.mean(var, axis = 0)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print("var of mu:", var_of_mu)
    print("mu of var:", mu_of_var)
    snr = var_of_mu / mu_of_var
    print("SNR:", snr)
    return snr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action = 'store', type = str, help = 'config file', required = True)
    parser.add_argument('--ckpt', action = 'store', type = str, help = 'checkpoint file', default = None)
    args = parser.parse_args()
    if args.ckpt is None:
        args.ckpt = glob.glob(os.path.join(os.path.dirname(args.config), "ckpt", "policy_last.ckpt"))[0]
        print("Using checkpoint:", args.ckpt)
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    cfg = edict(cfg)
    print(cfg)
    snr = calc_snr(cfg, args)
    snr_threshold = 0.05
    cnt = np.sum(snr > snr_threshold)
    print("informative feature dimension:", cnt)
