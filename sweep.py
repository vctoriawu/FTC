# -*- coding: utf-8 -*-
from network import Network
from get_config import get_config
from dataloader.as_dataloader import get_as_dataloader
from get_model import get_model
import os
from utils import validation_constructive
import yaml
import torch
import random
import numpy as np
import argparse

import wandb

def update_nested_dict(original, update):
    for key, value in update.items():
        if isinstance(value, dict):
            original[key] = update_nested_dict(original.get(key, {}), value)
        else:
            original[key] = value[0] if isinstance(value, list) and len(value) == 1 else value
    return original

def dict_print(a_dict):
    for k, v in a_dict.items():
        print(f"{k}: {v}")

def set_seed(seed):
    """
    Set up random seed number
    """
    # # Setup random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train():

    with open("sweep.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    if config['parameters']['use_wandb']:
        wandb.init(
            project="as_tab", 
            entity="rcl_stroke", 
            config = config, 
            name = '3_class_CLIP')
    
    # Update config based on wandb sweep selected configs
    config_wandb = wandb.config
    config = update_nested_dict(config, config_wandb)

    # printing the configuration again
    print(f"################################ NEW CONFIGS ######################")
    dict_print(config)

    # ############# handling the logistics of (seed), and (logging) ###############
    set_seed(config["seed"])
    model = get_model(config)
    net = Network(model, config)
    dataloader_tr = get_as_dataloader(config, split='train', mode='train')
    dataloader_ssl = get_as_dataloader(config, split='train_all', mode='ssl')
    dataloader_va = get_as_dataloader(config, split='val', mode='val')
    dataloader_test = get_as_dataloader(config, split='test', mode='val')
    dataloader_te = get_as_dataloader(config, split='test', mode='test')
    dataloader_validation = get_as_dataloader(config, split='val', mode='test')
    
    if config['mode']=="train":
        net.train(dataloader_tr, dataloader_va,dataloader_test)
        net.test_comprehensive(dataloader_te, mode="test")
    if config['mode']=="ssl":
        net.train(dataloader_ssl, dataloader_va)
        #net.test_comprehensive(dataloader_te, mode="test")
    if config['mode']=="test":
        net.test_comprehensive(dataloader_validation, mode="test", record_embeddings=False)
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    with open("sweep.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    #train()
    wandb.agent(config['SWEEP_ID'], train, count=config['SWEEP_COUNT'])