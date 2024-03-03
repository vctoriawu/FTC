###Use TMED dataset entirely for testing###

# -*- coding: utf-8 -*-
from network_p import Network
from get_config import get_config
from dataloader.tmed_patientloader import get_as_dataloader
from get_model import get_model
import os
from utils import validation_constructive

import wandb

if __name__ == "__main__":
    
    config = get_config()
    
    if config['use_wandb']:
        run = wandb.init(project="TMED", entity="asproject",config = config, name = 'FTC_image_tmed_test')
    
    model = get_model(config)
    net = Network(model, config)
    dataloader_te = get_as_dataloader(config, split='all', mode='test')
    net.test_comprehensive(dataloader_te, mode="test", record_embeddings=True)
    
    if config['use_wandb']:
        wandb.finish()