"""Evaluate DFR on spurious correlations datasets."""

import torch
import torch.nn as nn
import torchvision
#from torch.utils.tensorboard import SummaryWriter
import pathlib
import numpy as np
import pandas as pd
import os
import tqdm
import argparse
#import sys
#from collections import defaultdict
import json
from functools import partial
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils import Logger, evaluate, get_y_p

from dfr_network import Network
from dfr_configs import get_config
from dataloader.as_dataloader import get_as_dataloader
from dataloader.as_dataloader_dfr import get_dfr_dataloader
from get_model import get_model
from random import randint
from utils import validation_constructive, set_seed
from FTC.util.misc import NestedTensor

import wandb

## won't need this
C_OPTIONS = [1., 0.7, 0.3, 0.1, 0.07, 0.03, 0.01] 
CLASS_WEIGHT_OPTIONS = [1., 2., 3., 10., 100., 300., 1000.] 
REG = "l2" 
CLASS_WEIGHT_OPTIONS = [{0: 1, 1: w, 2: w, 3: 1} for w in CLASS_WEIGHT_OPTIONS] 
# CLASS_WEIGHT_OPTIONS = [{0: 1, 1: w} for w in CLASS_WEIGHT_OPTIONS] + [
#         {0: w, 1: 1} for w in CLASS_WEIGHT_OPTIONS] 
#TODO - play around with class weights - maybe put more weight on the middle classes?

def dfr_on_validation_tune(
        all_info, preprocess=True,
        balance_val=False, add_train=True, num_retrains=1):

    all_embeddings, all_att_weights, all_y, all_g, _, _, _ = all_info

    worst_accs = {}
    for i in range(num_retrains):
        x_val = all_embeddings["val"] ##np.array[n_train,F,E]
        att_val = all_att_weights["val"] ##np.array[n_train,F,1]
        y_val = all_y["val"]
        g_val = all_g["val"]
        n_groups = np.max(g_val) + 1

        n_val = len(x_val) // 5 #TODO - train on a larger subset of val
        idx = np.arange(len(x_val))
        np.random.shuffle(idx)

        x_valtrain = x_val[idx[n_val:]]
        att_valtrain = att_val[idx[n_val:]]
        y_valtrain = y_val[idx[n_val:]]
        g_valtrain = g_val[idx[n_val:]]

        n_groups = np.max(g_valtrain) + 1
        g_idx = [np.where(g_valtrain == g)[0] for g in range(n_groups)]
        min_g = np.min([len(g) for g in g_idx])
        for g in g_idx:
            np.random.shuffle(g)
        if balance_val:
            x_valtrain = np.concatenate([x_valtrain[g[:min_g]] for g in g_idx])
            att_valtrain = np.concatenate([att_valtrain[g[:min_g]] for g in g_idx])
            y_valtrain = np.concatenate([y_valtrain[g[:min_g]] for g in g_idx])
            g_valtrain = np.concatenate([g_valtrain[g[:min_g]] for g in g_idx])

        x_val = x_val[idx[:n_val]]
        att_val = att_val[idx[:n_val]]
        y_val = y_val[idx[:n_val]]
        g_val = g_val[idx[:n_val]]

        n_train = len(x_valtrain) if add_train else 0

        x_train = np.concatenate([all_embeddings["train"][:n_train], x_valtrain])
        att_train = np.concatenate([all_att_weights["train"][:n_train], att_valtrain])
        y_train = np.concatenate([all_y["train"][:n_train], y_valtrain])
        g_train = np.concatenate([all_g["train"][:n_train], g_valtrain])
        print(np.bincount(g_train))

        #TODO - add normal bicuspid cases from training, otherwise, there are only 3 cases.

        #[n,F,E] -> [n, E] 
        x_train = (x_train * att_train).sum(1) 
        x_val = (x_val * att_val).sum(1) 
        
        if preprocess: # Try with and without
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train) 
            x_val = scaler.transform(x_val)


        if balance_val and not add_train:
            cls_w_options = [{0: 1., 1: 1., 2: 1., 3: 1.}] 
        else:
            cls_w_options = CLASS_WEIGHT_OPTIONS
        for c in C_OPTIONS:
            for class_weight in cls_w_options:
                # if args.classifier=="logistic_reg":
                logreg = LogisticRegression(penalty=REG, C=c, solver="lbfgs",
                                            class_weight=class_weight) 
                logreg.fit(x_train, y_train) #logreg expects shape [n_samples, n_features]
                preds_val = logreg.predict(x_val) #[n, E] -> [n,]             
                group_accs = np.array(
                    [(preds_val == y_val)[g_val == g].mean()
                     for g in range(n_groups)])
                worst_acc = np.min(group_accs)
                if i == 0:
                    worst_accs[c, class_weight[0], class_weight[1], class_weight[2], class_weight[3]] = worst_acc
                else:
                    worst_accs[c, class_weight[0], class_weight[1], class_weight[2], class_weight[3]] += worst_acc
                # print(c, class_weight[0], class_weight[1], worst_acc, worst_accs[c, class_weight[0], class_weight[1]])
    ks, vs = list(worst_accs.keys()), list(worst_accs.values())
    best_hypers = ks[np.argmax(vs)]
    return best_hypers


def dfr_on_validation_eval( 
        c, w1, w2, w3, w4, all_info, save_path, num_retrains=20,
        preprocess=True, balance_val=False, add_train=True):
    coefs, intercepts = [], []

    all_embeddings, all_att_weights, all_y, all_g, all_p, all_views, all_echo_ids  = all_info

    #[n, F, E] --> [n, E]
    preprocess_train = (all_embeddings["train"] * all_att_weights["train"]).sum(1) 

    if preprocess: # Try with and without
        scaler = StandardScaler()
        scaler.fit(preprocess_train)

    for i in range(num_retrains):
        x_val = all_embeddings["val"]
        att_val = all_att_weights["val"]
        y_val = all_y["val"]
        g_val = all_g["val"]
        n_groups = np.max(g_val) + 1
        g_idx = [np.where(g_val == g)[0] for g in range(n_groups)]

        # add normal bicuspid cases from training
        train_idx = np.where(all_g["train"] == 1)[0] 
        n_train_g1 = len(train_idx)

        min_g = np.min([len(g) for g in g_idx]) 
        for g in g_idx:
            np.random.shuffle(g)
        if balance_val:
            x_val_ls, att_val_ls, y_val_ls, g_val_ls = [], [], [], []
            for i, g in enumerate(g_idx):
                if i==1:
                    x_valg1 = np.concatenate((x_val[g[:min_g]], all_embeddings["train"][train_idx]))
                    x_val_ls.append(x_valg1)
                    att_valg1 = np.concatenate((att_val[g[:min_g]], all_att_weights["train"][train_idx]))
                    att_val_ls.append(att_valg1)
                    y_valg1 = np.concatenate((y_val[g[:min_g]], all_y["train"][train_idx]))
                    y_val_ls.append(y_valg1)
                    g_valg1 = np.concatenate((g_val[g[:min_g]], all_g["train"][train_idx]))
                    g_val_ls.append(g_valg1)
                else:
                    x_val_ls.append(x_val[g[:min_g+n_train_g1]])
                    att_val_ls.append(att_val[g[:min_g+n_train_g1]])
                    y_val_ls.append(y_val[g[:min_g+n_train_g1]])
                    g_val_ls.append(g_val[g[:min_g+n_train_g1]])
            x_val = np.concatenate(x_val_ls)
            att_val = np.concatenate(att_val_ls)
            y_val = np.concatenate(y_val_ls)
            g_val = np.concatenate(g_val_ls)
            # x_val = np.concatenate([x_val[g[:min_g]] for g in g_idx])
            # att_val = np.concatenate([att_val[g[:min_g]] for g in g_idx])
            # y_val = np.concatenate([y_val[g[:min_g]] for g in g_idx])
            # g_val = np.concatenate([g_val[g[:min_g]] for g in g_idx])

        n_train = len(x_val) if add_train else 0
        train_idx = np.arange(len(all_embeddings["train"]))
        np.random.shuffle(train_idx)
        train_idx = train_idx[:n_train]

        x_train = np.concatenate(
            [all_embeddings["train"][train_idx], x_val])
        att_train = np.concatenate(
            [all_att_weights["train"][train_idx], att_val]) 
        y_train = np.concatenate([all_y["train"][train_idx], y_val])
        g_train = np.concatenate([all_g["train"][train_idx], g_val])
        print(np.bincount(g_train))

        #[n, F, E] -> [n, E]
        x_train = (x_train * att_train).sum(1) 

        if preprocess:
            x_train = scaler.transform(x_train)

        logreg = LogisticRegression(penalty=REG, C=c, solver="lbfgs",
                                    class_weight={0: w1, 1: w2, 2: w3, 3: w4}) 
        logreg.fit(x_train, y_train) #logreg expects shape [n_samples, n_features]
        coefs.append(logreg.coef_)
        intercepts.append(logreg.intercept_)

    x_test = all_embeddings["test"]
    att_test = all_att_weights["test"]
    y_test = all_y["test"]
    g_test = all_g["test"]
    print(np.bincount(g_test))

    #[n, F, E] -> [n, E]
    x_test = (x_test * att_test).sum(1) 

    if preprocess:
        x_test = scaler.transform(x_test)
    logreg = LogisticRegression(penalty=REG, C=c, solver="lbfgs",
                                class_weight={0: w1, 1: w2, 2: w3, 3: w4}) 
    n_classes = np.max(y_train) + 1
    # the fit is only needed to set up logreg
    logreg.fit(x_train[:n_classes], np.arange(n_classes))
    logreg.coef_ = np.mean(coefs, axis=0)
    logreg.intercept_ = np.mean(intercepts, axis=0)

    preds_test = logreg.predict(x_test) 
    preds_test_logits = logreg.predict_proba(x_test)
    preds_train = logreg.predict(x_train)
    n_groups = np.max(g_train) + 1
    test_accs = [(preds_test == y_test)[g_test == g].mean()
                 for g in range(n_groups)]
    test_mean_acc = (preds_test == y_test).mean()
    train_accs = [(preds_train == y_train)[g_train == g].mean()
                  for g in range(n_groups)]
    #save to csv to calculate study level metrics
    d = {"GT_AS": y_test,
        "echo_id": all_echo_ids["test"],
        "view": all_views["test"],
        "groups_arr": g_test,
        "places_arr": all_p["test"],
        "normal_pred": preds_test_logits[:, 0],
        "mild_pred": preds_test_logits[:, 1],
        "mod_pred": preds_test_logits[:, 2],
        "severe_pred": preds_test_logits[:, 3]}
    df = pd.DataFrame(data=d)
    test_results_file = os.path.join(save_path, "test.csv") 
    df.to_csv(test_results_file)

    return test_accs, test_mean_acc, train_accs

# def dfr_train_subset_tune(
#         all_embeddings, all_y, all_g, preprocess=True,
#         learn_class_weights=False):

#     x_val = all_embeddings["val"]
#     y_val = all_y["val"]
#     g_val = all_g["val"]

#     x_train = all_embeddings["train"]
#     y_train = all_y["train"]
#     g_train = all_g["train"]

#     if preprocess:
#         scaler = StandardScaler()
#         scaler.fit(x_train)

#     n_groups = np.max(g_train) + 1
#     g_idx = [np.where(g_train == g)[0] for g in range(n_groups)]
#     for g in g_idx:
#         np.random.shuffle(g)
#     min_g = np.min([len(g) for g in g_idx])
#     x_train = np.concatenate([x_train[g[:min_g]] for g in g_idx])
#     y_train = np.concatenate([y_train[g[:min_g]] for g in g_idx])
#     g_train = np.concatenate([g_train[g[:min_g]] for g in g_idx])
#     print(np.bincount(g_train))
#     if preprocess:
#         x_train = scaler.transform(x_train)
#         x_val = scaler.transform(x_val)

#     worst_accs = {}
#     if learn_class_weights:
#         cls_w_options = CLASS_WEIGHT_OPTIONS
#     else:
#         cls_w_options = [{0: 1., 1: 1.}]
#     for c in C_OPTIONS:
#         for class_weight in cls_w_options:
#             logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
#                                         class_weight=class_weight, max_iter=20)
#             logreg.fit(x_train, y_train)
#             preds_val = logreg.predict(x_val)
#             group_accs = np.array(
#                 [(preds_val == y_val)[g_val == g].mean() for g in range(n_groups)])
#             worst_acc = np.min(group_accs)
#             worst_accs[c, class_weight[0], class_weight[1]] = worst_acc
#             print(c, class_weight, worst_acc, group_accs)

#     ks, vs = list(worst_accs.keys()), list(worst_accs.values())
#     best_hypers = ks[np.argmax(vs)]
#     return best_hypers


# def dfr_train_subset_eval(
#         c, w1, w2, all_embeddings, all_y, all_g, num_retrains=10,
#         preprocess=True):
#     coefs, intercepts = [], []
#     x_train = all_embeddings["train"]
#     scaler = StandardScaler()
#     scaler.fit(x_train)

#     for i in range(num_retrains):
#         x_train = all_embeddings["train"]
#         y_train = all_y["train"]
#         g_train = all_g["train"]
#         n_groups = np.max(g_train) + 1

#         g_idx = [np.where(g_train == g)[0] for g in range(n_groups)]
#         min_g = np.min([len(g) for g in g_idx])
#         for g in g_idx:
#             np.random.shuffle(g)
#         x_train = np.concatenate([x_train[g[:min_g]] for g in g_idx])
#         y_train = np.concatenate([y_train[g[:min_g]] for g in g_idx])
#         g_train = np.concatenate([g_train[g[:min_g]] for g in g_idx])
#         print(np.bincount(g_train))

#         if preprocess:
#             x_train = scaler.transform(x_train)

#         logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
#                                         class_weight={0: w1, 1: w2})
#         logreg.fit(x_train, y_train)

#         coefs.append(logreg.coef_)
#         intercepts.append(logreg.intercept_)

#     x_test = all_embeddings["test"]
#     y_test = all_y["test"]
#     g_test = all_g["test"]

#     if preprocess:
#         x_test = scaler.transform(x_test)

#     logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")
#     n_classes = np.max(y_train) + 1
#     # the fit is only needed to set up logreg
#     logreg.fit(x_train[:n_classes], np.arange(n_classes))
#     logreg.coef_ = np.mean(coefs, axis=0)
#     logreg.intercept_ = np.mean(intercepts, axis=0)

#     preds_test = logreg.predict(x_test)
#     preds_train = logreg.predict(x_train)
#     n_groups = np.max(g_train) + 1
#     test_accs = [(preds_test == y_test)[g_test == g].mean()
#                  for g in range(n_groups)]
#     test_mean_acc = (preds_test == y_test).mean()
#     train_accs = [(preds_train == y_train)[g_train == g].mean()
#                   for g in range(n_groups)]
#     return test_accs, test_mean_acc, train_accs

# Extract embeddings
def get_embed(m, x):
    # Video dimension (B x F x C x H x W)
    x = x.permute(0,2,1,3,4)
    nB, nF, nC, nH, nW = x.shape
    # Merge batch and frames dimension
    x = x.contiguous().view(nB*nF,nC,nH,nW)
    # (BxF) x C x H x W => (BxF) x Emb
    embeddings = m.AE(x).squeeze()
    embeddings_reshaped = embeddings.view(nB, nF, m.em )
    embeddings_reshaped = embeddings_reshaped.permute(0,2,1)
        
    # Creatining porsitional embedding and nested tensor
    mask = torch.ones((nB, nF), dtype=torch.bool).cuda()
    #embeddings_reshaped = embeddings_reshaped.cuda()
    samples = NestedTensor(embeddings_reshaped, mask)
    pos = m.pos_embed(samples).cuda() #(bs, c, t)
    
    #  B x F x Emb
    outputs = m.transformer([embeddings_reshaped],[pos],[mask]) #(bs, emb, F)

    #only train/validate the video branch (no tab data)
    embedding_out = m.map_embed(outputs) #[B, F, E]
    
    # attention weights B x F x 1
    att_weight = m.attentionweights(embedding_out)
    att_weight = nn.functional.softmax(att_weight, dim=1) #[B, F, 1]

    output = (embedding_out, att_weight)

    return output

# def get_preds_dfr(x_val, finetuned_mlp):
#     """
#     args:
#         x_val: np.array of shape [batch, 2?] where second dim consists of tuple (embedding, att_weight)
#         finetuned_mlp: MLP classifier finetuned on a subset of the validation dataset.

#     Passes embeddings and attention weights from instances in the validation set 
#     into the finetuned MLP to get predictions 
#     """
#     pass

def restore_model(model, pt_file):
    """Restoring trained model."""
    print(f"restoring {pt_file}")

    # Read checkpoint file.
    load_res = torch.load(pt_file)
    # Loading model.
    model.load_state_dict(load_res["model"], strict=False)

    return model

def reinitialize_weights(module): #TODO
    if isinstance(module, nn.Linear):  # Or other layers you want to reinitialize
        module.reset_parameters()  # Or another initialization method

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    configs = get_config()

    if configs["seed"]:
        set_seed(configs["seed"])
    if configs['use_wandb']:
        run = wandb.init(project="as_tab", entity="rcl_stroke", config = configs, name = 'dfr_mlp')

    configs["logs_dir"] = configs["dfr_result_path"] + configs["experiment_name"]
    logs_path = pathlib.Path(configs["logs_dir"])
    logs_path.mkdir(parents=True, exist_ok=True)

    ## Load data
    dataloader_tr = get_as_dataloader(configs, split='train', dfr=True, mode='train') 
    dataloader_va = get_as_dataloader(configs, split='val', dfr=True, mode='val')
    dataloader_te = get_as_dataloader(configs, split='test', dfr=True, mode='test') 
    dataloader_trainval = get_dfr_dataloader(configs, split='train', dfr=True, mode='train_val') 
    dataloader_test = get_dfr_dataloader(configs, split='train', dfr=True, mode='test') 

    # Load MultiASNet 
    model = get_model(configs)
    bestmodel_path = pathlib.Path(configs["pretrained_model_dir"])
    model = restore_model(model, bestmodel_path)
    model.cuda()
    model.eval()

    # Evaluate model
    print("Base Model")
    base_model_results = {}
    get_yp_func = partial(get_y_p, n_places=configs["n_places"]) #set up func that gets labels y and p=0
    # base_model_results["train"] = evaluate(model, dataloader_tr, get_yp_func) 
    # base_model_results["val"] = evaluate(model, dataloader_va, get_yp_func)
    # base_model_results["test"] = evaluate(model, dataloader_test, get_yp_func)
    # print(base_model_results)
    # print() #TODO - comment out for now

    model.eval() 

    if configs["classifier"]=='mlp': 
        for param in model.parameters():
            param.requires_grad = False
        for param in model.aorticstenosispred.parameters():
            param.requires_grad = True
        ## Check that the right layers are trainable/frozen 
        # for name, param in model.named_parameters():
        #     print(f'Layer: {name} | requires_grad: {param.requires_grad}')
        # print("Before re-initialization:")
        # print("Weights:", model.aorticstenosispred[0].weight)
        # print("Biases:", model.aorticstenosispred[0].bias)
        model.aorticstenosispred.apply(reinitialize_weights)
        # print("\nAfter re-initialization:")
        # print("Weights:", model.aorticstenosispred[0].weight)
        # print("Biases:", model.aorticstenosispred[0].bias)

        # set up train and test
        dataloaders_dict = {"train": dataloader_trainval, "test": dataloader_test}
        net = Network(model, configs)

        if configs["do_train"]:
            net.train(dataloaders_dict["train"], dataloaders_dict["test"])
        if configs["do_comprehensive_test"]:
            net.test_comprehensive(dataloaders_dict["test"], mode='test')

        #test_accs, test_mean_acc, train_accs = net.test_comprehensive(dataloaders_dict["test"], mode='test') #TODO - comment out for now

    else: #logistic_reg
        all_embeddings = {}
        all_att_weights = {}
        all_y, all_p, all_g = {}, {}, {}
        all_views, all_echo_ids = {}, {}
        for name, loader in [("train", dataloader_tr), ("test", dataloader_te), ("val", dataloader_va)]:
            all_embeddings[name], all_att_weights[name] = [], []
            all_y[name], all_p[name], all_g[name] = [], [], []
            all_views[name], all_echo_ids[name] = [], []
            for data, dfr_info in tqdm.tqdm(loader):
                if name=="test":
                    x, _, y, _, di, _ = data
                else:
                    x, _, y, _ = data
                p, g = dfr_info
                with torch.no_grad():
                    (embeddings, att_weights) = get_embed(model, x.cuda())
                    all_embeddings[name].append(embeddings.detach().cpu().numpy()) #[[B,F,E], [B,F,E], ...]
                    all_att_weights[name].append(att_weights.detach().cpu().numpy()) 
                    all_y[name].append(y.detach().cpu().numpy())
                    all_g[name].append(g.detach().cpu().numpy())
                    all_p[name].append(p.detach().cpu().numpy())
                    if name=="test":
                        all_views[name].append(di['view'])
                        all_echo_ids[name].append(di['Echo ID#'].detach().cpu().numpy())
            all_embeddings[name] = np.vstack(all_embeddings[name]) #[n_train,F,E]
            all_att_weights[name] = np.vstack(all_att_weights[name]) #[n_train,F,E]
            all_y[name] = np.concatenate(all_y[name])
            all_g[name] = np.concatenate(all_g[name])
            all_p[name] = np.concatenate(all_p[name])
            if name=="test":
                all_views[name] = np.concatenate(all_views[name])
                all_echo_ids[name] = np.concatenate(all_echo_ids[name])

        # DFR on validation
        print("DFR on validation")
        dfr_val_results = {}
        all_info = (all_embeddings, all_att_weights, all_y, all_g, all_p, all_views, all_echo_ids)
        c, w1, w2, w3, w4 = dfr_on_validation_tune(
            all_info,
            balance_val=configs["balance_dfr_val"], add_train=not configs["notrain_dfr_val"])
        dfr_val_results["best_hypers"] = (c, w1, w2, w3, w4)
        print("Hypers:", (c, w1, w2, w3, w4))
        test_accs, test_mean_acc, train_accs = dfr_on_validation_eval( 
                c, w1, w2, w3, w4, all_info, logs_path,
            balance_val=configs["balance_dfr_val"], add_train=not configs["notrain_dfr_val"])

    dfr_val_results["test_accs"] = test_accs
    dfr_val_results["train_accs"] = train_accs
    dfr_val_results["test_worst_acc"] = np.min(test_accs)
    dfr_val_results["test_mean_acc"] = test_mean_acc
    print(dfr_val_results)
    print() #TODO - comment out for now

    # # DFR on train subsampled
    # print("DFR on train subsampled")
    # dfr_train_results = {}
    # c, w1, w2 = dfr_train_subset_tune(
    #     all_embeddings, all_y, all_g,
    #     learn_class_weights=args.tune_class_weights_dfr_train)
    # dfr_train_results["best_hypers"] = (c, w1, w2)
    # print("Hypers:", (c, w1, w2))
    # test_accs, test_mean_acc, train_accs = dfr_train_subset_eval(
    #         c, w1, w2, all_embeddings, all_y, all_g)
    # dfr_train_results["test_accs"] = test_accs
    # dfr_train_results["train_accs"] = train_accs
    # dfr_train_results["test_worst_acc"] = np.min(test_accs)
    # dfr_train_results["test_mean_acc"] = test_mean_acc
    # print(dfr_train_results)
    # print()

    all_results = {}
    all_results["base_model_results"] = base_model_results
    all_results["dfr_val_results"] = dfr_val_results
    ##all_results["dfr_train_results"] = dfr_train_results
    print(all_results)

    pickle_file = configs["logs_dir"] + "/dfr_multiasnet.pkl"
    with open(pickle_file, 'wb') as f: 
        pickle.dump(all_results, f) #TODO - comment out  for now