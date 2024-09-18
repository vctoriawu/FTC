# -*- coding: utf-8 -*-
from network import Network
from get_config import get_config
from dataloader.as_dataloader import get_as_dataloader
from get_model import get_model
import os
import sys
from random import randint
from utils import validation_constructive, set_seed

import wandb

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    config = get_config()

    if config["seed"]:
        set_seed(config["seed"])
    
    if config['use_wandb']:
        run = wandb.init(project="as_tab", entity="rcl_stroke", config = config, name = 'kl_reg-vid_no-entropy_fixed')
    
    model = get_model(config)
    net = Network(model, config)
    dataloader_tr = get_as_dataloader(config, split='train', mode='train')
    dataloader_ssl = get_as_dataloader(config, split='train_all', mode='ssl')
    dataloader_va = get_as_dataloader(config, split='val', mode='val')
    dataloader_test = get_as_dataloader(config, split='test', mode='val')
    dataloader_te = get_as_dataloader(config, split='test', mode='test')
    dataloader_validation = get_as_dataloader(config, split='val', mode='test')

    #dataloader_tr_info = get_as_dataloader(config, split='train', mode='train_analysis')
    
    if config['mode']=="train":
        net.train(dataloader_tr, dataloader_va,dataloader_test)
        net.test_comprehensive(dataloader_te, split='test', mode="test")
    if config['mode']=="ssl":
        net.train(dataloader_ssl, dataloader_va)
        net.test_comprehensive(dataloader_te, mode="test")
    if config['mode']=="test":
        net.test_comprehensive(dataloader_te, split='test', mode="test", record_embeddings=False)
        #net.test_comprehensive(dataloader_tr_info, split='train', mode="test", record_embeddings=False)
    if config['use_wandb']:
        wandb.finish()
