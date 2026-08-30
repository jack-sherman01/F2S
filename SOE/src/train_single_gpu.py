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
from diffusers.optimization import get_cosine_schedule_with_warmup

from utils.training import set_seed, plot_history_v2

def train(cfg):
    # set up device
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert torch.cuda.is_available()

    log_root = os.path.join(cfg.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    original_stdout = sys.stdout
    sys.stdout = open(os.path.join(log_root, 'log.txt'), 'w')

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
    if not hasattr(cfg, "num_iters_per_epoch") or cfg.num_iters_per_epoch is None:
        num_samples = len(dataset)
    else:
        num_samples = cfg.num_iters_per_epoch * cfg.batch_size
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
    elif cfg.policy.name == "VDP":
        from policy.vdp import VDP
        policy = VDP(**cfg.policy.params).to(device)
    else:
        raise NotImplementedError
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

    # load checkpoint
    if cfg.resume_ckpt is not None:
        resume_ckpt = cfg.resume_ckpt.format(seed = cfg.seed)
        resume_ckpt_list = glob.glob(resume_ckpt)
        if len(resume_ckpt_list) == 0:
            raise FileNotFoundError("Checkpoint {} not found.".format(resume_ckpt))
        elif len(resume_ckpt_list) > 1:
            print("Multiple checkpoints found, using the last one.")
            resume_ckpt = sorted(resume_ckpt_list)[-1]
        else:
            resume_ckpt = resume_ckpt_list[0]
        if hasattr(cfg, "resume_list") and cfg.resume_list is not None:
            # only resume modules parameters in resume_list
            resume_state_dict = torch.load(resume_ckpt, map_location = device)
            new_state_dict = {}
            for key in resume_state_dict:
                if any([resume in key for resume in cfg.resume_list]):
                    new_state_dict[key] = resume_state_dict[key]
                    print("Resuming {} ...".format(key))
            policy.load_state_dict(new_state_dict, strict = False)
        else:
            policy.load_state_dict(torch.load(resume_ckpt, map_location = device), strict = False)
        cfg.resume_ckpt = resume_ckpt
        print("Checkpoint {} loaded.".format(resume_ckpt))

    # ckpt path
    ckpt_dir = os.path.join(log_root, "ckpt")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    # optimizer and lr scheduler
    print("Loading optimizer and scheduler ...")
    if cfg.optimizer.name == "AdamW":
        if hasattr(cfg, "freeze_action_head") and cfg.freeze_action_head:
            for name, param in policy.named_parameters():
                if "action_decoder" in name:
                    param.requires_grad = False
                print(name, param.requires_grad)
        if hasattr(cfg, "freeze_image_encoder") and cfg.freeze_image_encoder:
            for name, param in policy.named_parameters():
                if "img_encoder" in name:
                    param.requires_grad = False
                print(name, param.requires_grad)
        if hasattr(cfg, "freeze_list") and cfg.freeze_list is not None:
            for name, param in policy.named_parameters():
                if any([freeze in name for freeze in cfg.freeze_list]):
                    param.requires_grad = False
                print(name, param.requires_grad)
        optimizer = torch.optim.AdamW(policy.parameters(), **cfg.optimizer.params)
    else:
        raise NotImplementedError

    if cfg.lr_scheduler.name == "cosine_with_warmup":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer = optimizer,
            num_warmup_steps = 2000,
            num_training_steps = len(dataloader) * cfg.num_epochs
        )
    else:
        raise NotImplementedError
    lr_scheduler.last_epoch = len(dataloader) * (cfg.resume_epoch + 1) - 1

    # save config
    with open(os.path.join(log_root, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent = 4)

    # training
    train_history = {}

    policy.train()
    for epoch in range(cfg.resume_epoch + 1, cfg.num_epochs):
        print("Epoch {}".format(epoch))
        optimizer.zero_grad()
        num_steps = len(dataloader)
        pbar = tqdm(dataloader)
        avg_loss = {"loss": 0.0}
        
        for data in pbar:
            batch = pre_process_data(data, device, cfg)
            # forward
            loss = policy(**batch)
            if isinstance(loss, dict):
                for key in loss:
                    if key not in avg_loss:
                        avg_loss[key] = 0.0
                    avg_loss[key] += loss[key].item()
                final_loss = loss["loss"]
            else:
                avg_loss["loss"] += loss.item()
                final_loss = loss
            # backward
            if hasattr(policy, "backward"):
                policy.backward(loss)
            else:
                final_loss.backward()
            if hasattr(cfg, "clip_grad_norm") and cfg.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        # avg_loss = avg_loss / num_steps
        for key in avg_loss:
            avg_loss[key] /= num_steps
        
        for key in avg_loss:
            if key not in train_history:
                train_history[key] = []
            train_history[key].append(avg_loss[key])

        print("Train loss: {:.6f}".format(avg_loss["loss"]), end = " ")
        for key in train_history:
            print("{}: {:.6f}".format(key, train_history[key][-1]), end = " ")
        print("")
        if (epoch + 1) % cfg.save_epochs == 0 \
            or (epoch + 1) in list(range(cfg.num_epochs - 100, cfg.num_epochs, 10)): # save last 100 epochs every 10 epochs
            torch.save(
                policy.state_dict(),
                os.path.join(ckpt_dir, "policy_epoch_{}_seed_{}.ckpt".format(epoch + 1, cfg.seed))
            )
            for key in train_history:
                plot_history_v2(train_history[key], epoch, ckpt_dir, key + "_seed_{}.png".format(cfg.seed))

    torch.save(
        policy.state_dict(),
        os.path.join(ckpt_dir, "policy_last.ckpt")
    )

    sys.stdout.close()
    sys.stdout = original_stdout
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action = 'store', type = str, help = 'config file', required = True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    cfg = edict(cfg)
    print(cfg)
    train(cfg)
