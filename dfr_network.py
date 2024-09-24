import os
import torch
import wandb
import pathlib
import torch.nn.functional as F
import numpy as np
import pandas as pd

from tqdm import tqdm
#from torchsummary import summary

from losses import laplace_cdf_loss, laplace_cdf
from losses import evidential_loss, evidential_prob_vacuity, loss_coteaching
from losses import SupConLoss, CLIPLoss, CeLossAbstain
from visualization.vis import plot_tsne_visualization
import dataloader.utils as utils
from utils import validation_constructive
from pathlib import Path
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

class TransformerFeatureMap:
    def __init__(self, model, layer_name='avgpool'):
        self.model = model
        for name, module in self.model.named_modules():
            if layer_name in name:
                module.register_forward_hook(self.get_feature)
        self.feature = []

    def get_feature(self, module, input, output):
        self.feature.append(output.cpu())

    def __call__(self, input_tensor):
        self.feature = []
        with torch.no_grad():
            output = self.model(input_tensor.cuda())

        return self.feature
    
class Network(object):
    """Wrapper for training and testing pipelines."""

    def __init__(self, model, config):
        """Initialize configuration."""
        self.config = config
        self.model = None
        self.model = model
        self.encoder = TransformerFeatureMap(self.model.cuda())
        
        if self.config['use_cuda']:  
            print(torch.cuda.device_count())
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
        self.num_classes_AS = config['num_classes']

        if config['l2_reg_method']=='reg_on_vid_only': 
            params_with_l2 = []
            params_without_l2 = []
            for name, param in self.model.named_parameters():
                # Apply L2 regularization only to video encoder weights
                if 'AE' or 'transformer' or 'aorticstenosispred' in name:
                    params_with_l2.append(param)
                else:  # Exclude other parameters 
                    params_without_l2.append(param)

        if config['l2_reg_method']=='reg_on_vid_only': 
            self.optimizer = torch.optim.Adam([
            {'params': params_with_l2, 'weight_decay': 1e-4},
            {'params': params_without_l2, 'weight_decay': 0.0}
            ], lr=config['lr'])
        elif config['l2_reg_method']== 'reg_on_all':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'], weight_decay=1e-4) 
        else: #no reg
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        
        self.loss_type = config['loss_type']
        self.contrastive_method = config['cotrastive_method']
        self.temperature = config['temp']
        # Recording the training losses and validation performance.
        self.train_losses = []
        self.valid_oas = []
        self.idx_steps = []

        # init auxiliary stuff such as log_func
        self._init_aux()

    def _init_aux(self):
        """Intialize aux functions, features."""
        # Define func for logging.
        self.log_func = print

        # Define directory wghere we save states such as trained model.
        self.log_dir = pathlib.Path(self.config['logs_dir']).mkdir(parents=True, exist_ok=True)

        # File that we save trained model into. 
        self.checkpts_file = os.path.join(self.config["logs_dir"], "checkpoint.pth")
        # We save model that achieves the best performance: early stopping strategy.
        self.best_model_dir = pathlib.Path(self.config['best_model_dir']).mkdir(parents=True, exist_ok=True)
        self.bestmodel_acc = os.path.join(self.config['best_model_dir'], 'best_model.pth') 
        # For test, we also save the results dataframe
        self.test_results_file = os.path.join(self.config['logs_dir'], "test.csv") 
    
    def _save(self, pt_file):
        """Saving trained model."""
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            pt_file,
        )

    def _restore(self, pt_file):
        """Restoring trained model."""
        print(f"restoring {pt_file}")

        # Read checkpoint file.
        load_res = torch.load(pt_file)
        # Loading model.
        self.model.load_state_dict(load_res["model"], strict=False)        
    
    def _get_loss(self, logits, target, nclasses):
        """ Compute loss function based on configs """
        if self.loss_type == 'cross_entropy':
            # BxC logits into Bx1 ground truth
            loss = F.cross_entropy(logits, target)
        else:
            raise NotImplementedError
        return loss
        
    #def train(self, loader_tr, loader_va,loader_te):
    def train(self, loader_tr, loader_te):
        """Training pipeline."""
        # Switch model into train mode.
        self.model.train()
        best_va_acc = 0.0 # Record the best validation metrics.
        best_va_loss = 10.0
        forget_rate = 0.0625
            
        for epoch in range(self.config['num_epochs']):
            losses = []
            correct_predictions = 0
            total_samples = 0
            print('Epoch: ' + str(epoch)) 
            
            with tqdm(total=len(loader_tr)) as pbar:
                for _, (data, dfr_info) in enumerate(loader_tr): 
                    cine, tab_data, target_AS, _ = data
                    places_arr, group_arr = dfr_info

                    # Cross Entropy Training
                    # Transfer data from CPU to GPU.
                    if self.config['use_cuda']: 
                        cine = cine.cuda()
                        tab_data = tab_data.cuda()
                        target_AS = target_AS.cuda()
                        places_arr = places_arr.cuda()
                        group_arr = group_arr.cuda()

                    pred_AS,_,_, _, _, _, _, _, _ = self.model(cine, tab_data, split='val') # Bx3xTxHxW
                    
                    if self.config["coteaching"] == True:
                        if self.config['abstention'] == True:
                            loss_vid, loss_tab = loss_coteaching(pred_AS, ca_preds, target_AS, forget_rate, self.abstention_loss)
                        else:    
                            loss_vid, loss_tab = loss_coteaching(pred_AS, ca_preds, target_AS, forget_rate)
                    else:
                        loss_vid = self._get_loss(pred_AS, target_AS, self.num_classes_AS)
                        #loss_tab = self._get_loss(ca_preds, target_AS, self.num_classes_AS)
                    loss = loss_vid #+ loss_tab
                    losses += [loss] 
                    # losses_vid += [loss_vid_weight*loss_vid]
                    # losses_tab += [loss_tab_weight*loss_tab]
                    
                    # Calculate the training accuracy
                    _, predicted = torch.max(pred_AS, 1)
                    correct_predictions += (predicted == target_AS).sum().item()
                    # Get correct preds per group #TODO
                    total_samples += target_AS.size(0)

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad() 
                    pbar.set_postfix_str("loss={:.4f}".format(loss.item()))
                    pbar.update()

            loss_avg = torch.mean(torch.stack(losses)).item() 
            # loss_avg_vid = torch.mean(torch.stack(losses_vid)).item()
            # loss_avg_tab = torch.mean(torch.stack(losses_tab)).item()

            ## Validation Data Evaluation 
            # acc_AS, val_total_loss, _, _, _, _, _, _, _ = self.test(loader_val, mode="val") 
            if self.config['use_wandb']:
                wandb.log({##training losses
                            "tr_loss":loss_avg, 
                            # "tr_loss-vid": loss_avg_vid,
                            # "tr_loss-tab": loss_avg_tab,
                            "tr_acc": (correct_predictions / total_samples)})
            
            ## Test Data Evaluation    
            acc_AS_te, te_total_loss = self.test(loader_te, mode="val") #TODO - log all losses
            if self.config['use_wandb']:
                wandb.log({ "te_loss":te_total_loss, "test_acc":acc_AS_te})

            # Save model every epoch.
            self._save(self.checkpts_file)

            # # Early stopping strategy.#TODO - what val set am I going to use?
            # if acc_AS > best_va_acc:
            #     # Save model with the best accuracy on validation set.
            #     best_va_acc = acc_AS
            #     self._save(self.bestmodel_acc)
            
            print(
                "Epoch: %3d, loss: %.5f, train acc: %.5f"
                % (epoch, loss_avg, (correct_predictions / total_samples))
            ) 

            # print(
            #     "Epoch: %3d, loss: %.5f, train acc: %.5f, val loss: %.5f, val acc: %.5f, top AS acc: %.5f"
            #     % (epoch, loss_avg, (correct_predictions / total_samples), val_total_loss, acc_AS, best_va_acc)
            # ) 

            # Recording training losses and validation performance.
            self.train_losses += [loss_avg]
            # self.valid_oas += [acc_AS]
            self.idx_steps += [epoch]

    @torch.no_grad()
    def test(self, loader_te, mode="test"):
        """Estimating the performance of model on the given dataset."""
        # Choose which model to evaluate.
        if mode=="test":
            self._restore(self.checkpts_file)
        # Switch the model into eval mode.
        self.model.eval()
        #conf_B = np.zeros((2,2))
        forget_rate = 0.0625
        losses = []
        # losses_vid = []
        # losses_tab = []
        correct_predictions = 0
        total_samples = 0

        for (data, _) in tqdm(loader_te): 
            cine, tab_data, target_AS, _, _, _ = data
            
            # Cross Entropy Training
            # Transfer data from CPU to GPU.
            if self.config['use_cuda']:
                cine = cine.cuda()
                tab_data = tab_data.cuda()
                target_AS = target_AS.cuda()
                
            pred_AS,_,_, _, _, _, _, _, _ = self.model(cine, tab_data, split='val') # Bx3xTxHxW #Train split allows us to get multimodal outputs to calculate losses

            if self.config["coteaching"] == True:
                if self.config['abstention'] == True:
                    loss_vid, loss_tab = loss_coteaching(pred_AS, ca_preds, target_AS, forget_rate, self.abstention_loss)
                else:    
                    loss_vid, loss_tab = loss_coteaching(pred_AS, ca_preds, target_AS, forget_rate)
            else:
                loss_vid = self._get_loss(pred_AS, target_AS, self.num_classes_AS)
                #loss_tab = self._get_loss(ca_preds, target_AS, self.num_classes_AS)

            # Calculate the training accuracy
            _, predicted = torch.max(pred_AS, 1)
            correct_predictions += (predicted == target_AS).sum().item()
            # Get correct preds per group #TODO
            total_samples += target_AS.size(0)

            loss = loss_vid
            losses += [loss]
            # losses_vid += [loss_vid_weight*loss_vid]
            # losses_tab += [loss_tab_weight*loss_tab]
        
        acc_AS = correct_predictions/total_samples
        total_loss_avg = torch.mean(torch.stack(losses)).item()
        # loss_avg_vid = torch.mean(torch.stack(losses_vid)).item()
        # loss_avg_tab = torch.mean(torch.stack(losses_tab)).item()

        # Switch the model into training mode
        self.model.train()
        return acc_AS, total_loss_avg, #loss_avg_vid, loss_avg_tab
    
    @torch.no_grad()
    def test_comprehensive(self, loader, mode="test"):
        """Logs the network outputs in dataloader
        computes per-patient preds and outputs result to a DataFrame"""
        print('NOTE: test_comprehensive mode uses batch_size=1 to correctly display metadata')
        # Choose which model to evaluate.
        if mode=="test":
            self._restore(self.checkpts_file)
        # Switch the model into eval mode.
        self.model.eval()
        echo, view, target_AS_arr, g_arr, p_arr = [], [], [], [], []
        normal_pred, mild_pred, mod_pred, severe_pred = [], [], [], []
        
        for (data, dfr_info) in tqdm(loader):
            cine, tab_info, labels_AS, _, data_info, _ = data
            places_arr, group_arr = dfr_info

            # collect the label info
            target_AS_arr.append(int(labels_AS[0]))

            if self.config['use_cuda']:
                cine = cine.cuda()
                tab_info = tab_info.cuda()
                labels_AS = labels_AS.cuda()
                
            # collect metadata from data_info
            echo.append(int(data_info['Echo ID#'][0]))
            view.append(data_info['view'][0])
            g_arr.append(group_arr.item())
            p_arr.append(places_arr.item())

            pred_AS,_,_, _, _, _, _, _, _ = self.model(cine, tab_info, split='val') 

            # collect the model prediction info
            normal_pred.append(pred_AS[:, 0].item()) 
            mild_pred.append(pred_AS[:, 1].item())
            mod_pred.append(pred_AS[:, 2].item())
            severe_pred.append(pred_AS[:, 3].item())
                
        d = {"GT_AS": target_AS_arr,
            "echo_id": echo,
            "view": view,
            "groups_arr": g_arr,
            "places_arr": p_arr,
            "normal_pred": normal_pred,
            "mild_pred": mild_pred,
            "mod_pred": mod_pred,
            "severe_pred": severe_pred}
        df = pd.DataFrame(data=d) 

        df.to_csv(self.test_results_file)

        ## For each group, get the test accs and then mean acc TODO
        # test_accs = [(preds_test == y_test)[g_test == g].mean()
        #              for g in range(n_groups)]
        # test_mean_acc = (preds_test == y_test).mean()
        # train_accs = [(preds_train == y_train)[g_train == g].mean()
        #               for g in range(n_groups)]
        
        #return test_accs, test_mean_acc, train_accs #TODO - comment out for now