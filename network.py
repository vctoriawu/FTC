import os
import torch
import wandb
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
    
def NPairsample():
    labels = torch.arange(32)
    # Reshape labels to compute adjacency matrix.
    labels_reshaped = torch.reshape(labels, (labels.shape[0], 1))
    labels_remapped_p2 = 1*(torch.eq(labels_reshaped, (labels_reshaped + 2).T))
    labels_remapped_p1 = 1*(torch.eq(labels_reshaped, (labels_reshaped + 1).T))
    labels_remapped_0 = 1*(torch.eq(labels_reshaped, (labels_reshaped).T))
    labels_remapped_b1 = 1*(torch.eq(labels_reshaped, (labels_reshaped - 1).T))
    labels_remapped_b2 = 1*(torch.eq(labels_reshaped, (labels_reshaped - 2).T))
    neg = 1 - (labels_remapped_p2 + labels_remapped_p1 + labels_remapped_0 + labels_remapped_b1 + labels_remapped_b2) 
    pos = labels_remapped_p2 + labels_remapped_p1 + labels_remapped_b1 + labels_remapped_b2
    pos = pos.float()/ torch.sum(pos , dim=1, keepdims=True)
    return pos.cuda(),neg.cuda()
    
    

class Network(object):
    """Wrapper for training and testing pipelines."""

    def __init__(self, model, config):
        """Initialize configuration."""
        self.config = config
        self.model = None
        self.model = model
        self.encoder = TransformerFeatureMap(self.model.cuda())
        # pos and neg matrix for npair_loss
        self.pos , self.neg = NPairsample()
        if config['cotrastive_method']=='Linear':
            checkpoint = torch.load(Path('/AS_Neda/FTC/logs/tad_1e-4_added_losses_try2/checkpoint.pth'))
            self.model.load_state_dict(checkpoint["model"], strict=False)
            print("Checkpoint_loaded")
        if self.config['use_cuda']:  
            print(torch.cuda.device_count())
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
        self.num_classes_AS = config['num_classes']

# TODO - finish setting up regularization only on the video encoder
# Separate parameters into groups
        params_with_l2 = []
        params_without_l2 = []
# Apply L2 regularization only to video encoder weights
        for name, param in self.model.named_parameters():
            # Apply L2 regularization
            if 'AE' or 'transformer' or 'aorticstenosispred' in name:
                params_with_l2.append(param)
            # Exclude other parameters 
            else:  
                params_without_l2.append(param)

        if config['cotrastive_method']=='Linear':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config['lr']) 
            
        else:
            if config['l2_reg_method']=='reg_on_vid_only': 
                self.optimizer = torch.optim.Adam([
                {'params': params_with_l2, 'weight_decay': 1e-4},
                {'params': params_without_l2, 'weight_decay': 0.0}
                ], lr=config['lr'])
            elif config['l2_reg_method']== 'reg_on_all':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'], weight_decay=1e-4) 
            else: #no reg
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])

        if config['lr_scheduler'] == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        mode='min', 
                                                                        factor=0.1, 
                                                                        patience=5, verbose=False)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        T_max=config['num_epochs'],
                                                                        eta_min = 0.000001)
        
        self.loss_type = config['loss_type']
        self.contrastive_method = config['cotrastive_method']
        self.temperature = config['temp']
        #self.bicuspid_weight = config['bicuspid_weight']
        # Recording the training losses and validation performance.
        self.train_losses = []
        self.valid_oas = []
        self.idx_steps = []

        # init auxiliary stuff such as log_func
        self._init_aux()

        # loss for the embedding space
        self.embed_loss_cos = CLIPLoss(temperature=0.1, lambda_0=0.5)

        if self.config["abstention"]:
            self.abstention_loss = CeLossAbstain(loss_weight=1, ab_weight=0.3, reduction="mean", ab_logitpath="joined")

        # self._restore('../checkpoint.pth')

    def _init_aux(self):
        """Intialize aux functions, features."""
        # Define func for logging.
        self.log_func = print

        # Define directory wghere we save states such as trained model.
        if self.config['use_wandb']:
            if self.config['mode'] == 'test':
                # read from the pre-specified test folder
                if self.config['model_load_dir'] is None:
                    raise AttributeError('For test-only mode, please specify the model state_dict folder')
                self.log_dir = os.path.join(self.config['log_dir'], self.config['model_load_dir'])
            else:
                # create a new directory
                self.log_dir = os.path.join(self.config['log_dir'], wandb.run.name)
        else:
            self.log_dir = self.config['log_dir']
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # File that we save trained model into.
        self.checkpts_file = os.path.join(self.log_dir, "checkpoint.pth")
        # We save model that achieves the best performance: early stopping strategy.
        self.best_model_dir = self.config['best_model_dir']
        if not os.path.exists(self.best_model_dir):
            os.makedirs(self.best_model_dir)      
        self.bestmodel_acc = os.path.join(self.best_model_dir, 'best_model_acc.pth')
        self.bestmodel_loss = os.path.join(self.best_model_dir, 'best_model_loss.pth')
        self.bestmodel_file_contrastive = os.path.join(self.log_dir, "best_model_cont.pth")
        # For test, we also save the results dataframe
        self.test_results_file = os.path.join(self.log_dir, "best_model.pth")

        # if self.config['use_tab']:
        #     self.checkpts_file = os.path.join(self.log_dir, "multimodal_checkpoint.pth")

        # if self.config['use_tab']:
        #     self.bestmodel_file = os.path.join(self.best_model_dir, 'best_multimodal_model.pth')
        # else:
        #     self.bestmodel_file = os.path.join(self.best_model_dir, 'best_model.pth')

        # if self.config['use_tab']:
        #     self.bestmodel_file_contrastive = os.path.join(self.log_dir, "best_multimodalmodel_cont.pth")
        # else:
        #     self.bestmodel_file_contrastive = os.path.join(self.log_dir, "best_model_cont.pth")

        # if self.config['use_tab']:
        #     self.test_results_file = os.path.join(self.log_dir, "best_multimodal_model.pth")
        # else:
        #     self.test_results_file = os.path.join(self.log_dir, "best_model.pth")

    
    def _save(self, pt_file):
        """Saving trained model."""

        # Save the trained model.
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
        # Loading optimizer.
        #self.optimizer.load_state_dict(load_res["optimizer"])
        
        
    
    def _get_loss(self, logits, target, nclasses):
        """ Compute loss function based on configs """
        if self.loss_type == 'cross_entropy':
            # BxC logits into Bx1 ground truth
            loss = F.cross_entropy(logits, target)
        elif self.loss_type == 'evidential':
            # alpha = F.softplus(logits)+1
            # target_oh = F.one_hot(target, nclasses)
            # mse, kl = evidential_mse(alpha, target_oh, alpha.device)
            # loss = torch.mean(mse + 0.1*kl)
            loss = evidential_loss(logits, target, nclasses)
        elif self.loss_type == 'laplace_cdf':
            # logits_categorical = laplace_cdf(F.sigmoid(logits), nclasses, logits.device)
            # target_oh = 0.9*F.one_hot(target, nclasses) + 0.1/nclasses
            # loss = F.binary_cross_entropy(logits_categorical, target_oh)
            loss = laplace_cdf_loss(logits, target, nclasses)
        elif self.loss_type == 'SupCon' or self.loss_type == 'SimCLR':
            criterion = SupConLoss(temperature=self.temperature)
            if torch.cuda.is_available():
                criterion = criterion.cuda()
            if self.contrastive_method == 'SupCon':
                loss = criterion(logits, target)
            elif self.contrastive_method == 'SimCLR':
                loss = criterion(logits)
        else:
            raise NotImplementedError
        return loss
    
    # obtain summary statistics of
    # argmax, max_percentage, entropy, evid.uncertainty for each function
    # expects logits BxC for classification, Bx2 for cdf
    def _get_prediction_stats(self, logits, nclasses):
        # convert logits to probabilities
        if self.loss_type == 'cross_entropy':
            prob = F.softmax(logits, dim=1)
            vacuity = -1
        elif self.loss_type == 'evidential':
            prob, vacuity = evidential_prob_vacuity(logits, nclasses)
            #vacuity = vacuity.squeeze()
        elif self.loss_type == 'laplace_cdf':
            prob = laplace_cdf(F.sigmoid(logits), nclasses, logits.device)
            vacuity = -1
        else:
            raise NotImplementedError
        max_percentage, argm = torch.max(prob, dim=1)
        entropy = torch.sum(-prob*torch.log(prob), dim=1)
        uni = utils.test_unimodality(prob.cpu().numpy())
        return argm, max_percentage, entropy, vacuity, uni, logits
        
    def train(self, loader_tr, loader_va,loader_te):
        """Training pipeline."""
        # Switch model into train mode.
        self.model.train()
        best_va_acc = 0.0 # Record the best validation metrics.
        best_va_loss = 10.0
        best_va_acc_supcon = 0.0
        forget_rate = 0.0625
        best_cont_loss = 1000

        gradient_accumulation_steps = 9
            
        for epoch in range(self.config['num_epochs']):
            #losses_AS = []
            #losses_B = []
            losses = []
            losses_vid = []
            losses_tab = []
            losses_ca_emb = []
            losses_npair = []
            losses_frame_att = []
            losses_vid_entropy = []
            losses_multimodal_entropy = []

            correct_predictions = 0
            total_samples = 0
            print('Epoch: ' + str(epoch) + ' LR: ' + str(self.optimizer.param_groups[0]["lr"])) #get_lr()))
            
            with tqdm(total=len(loader_tr)) as pbar:
                for index, data in enumerate(loader_tr):
                    cine = data[0]
                    tab_data = data[1]
                    target_AS = data[2]
                    target_B = data[3]

                    # Cross Entropy Training
                    if self.config['cotrastive_method'] == 'CE' or self.config['cotrastive_method'] == "Linear":
                        # Transfer data from CPU to GPU.
                        if self.config['use_cuda']:
                            if self.config['model'] == 'slowfast':
                                cine = [c.cuda() for c in cine]
                            else:
                                cine = cine.cuda()
                            tab_data = tab_data.cuda()
                            target_AS = target_AS.cuda()
                            target_B = target_B.cuda()
                        if self.config['model'] == "FTC_TAD":
                            pred_AS,entropy_attention,outputs, att_weight, ca_preds, learned_emb, ca_embed, ca_att_weight, multimodal_att_entropy = self.model(cine, tab_data, split='Train') # Bx3xTxHxW
                            
                            # Calculate loss between learned joint embeddings
                            ca_emb_loss, _, _ = self.embed_loss_cos(learned_emb, ca_embed, target_AS)

                            # Calculating temporal coherent npair loss
                            similarity_matrix = (torch.bmm(outputs,outputs.permute((0,2,1)))/1024)
                            self.pos_e = self.pos.repeat(len(pred_AS),1,1)
                            self.neg_e = self.neg.repeat(len(pred_AS),1,1)
                            npair_loss = torch.mean(-torch.sum(self.pos_e*similarity_matrix,dim =2) + 
                               torch.log(torch.exp(torch.sum(self.pos_e*similarity_matrix,dim=2))+
                               torch.sum(self.neg_e*torch.exp(self.neg_e*similarity_matrix),dim=2)), dim = 1)
                            
                            # remove video entropy loss
                            if not self.config["video_entropy"]:
                                entropy_attention = entropy_attention * 0

                            # Alignment between video and multimodal attention weights 
                            frame_att_loss_weight = self.config["frame_att_loss_weight"]
                            if not self.config["multimodal_att_entropy"]:
                                    multimodal_att_entropy = multimodal_att_entropy * 0  

                            if self.config["frame_attention_loss"] == "cosine_sim":
                                cos = torch.nn.CosineSimilarity()
                                frame_att_loss = cos(ca_att_weight, att_weight).mean().item()                                 
                            elif self.config["frame_attention_loss"] == "kl_div":
                                ## model outputs need to be log probabilities, targets need to be probabilities
                                att_weight = torch.log(att_weight)
                                frame_att_loss = F.kl_div(input=att_weight, target=ca_att_weight, log_target=False, reduction='mean')
                                
                            else:
                                frame_att_loss = torch.zeros(1).cuda()

                            loss_vid_weight = self.config["loss_vid_weight"]
                            loss_tab_weight = self.config["loss_tab_weight"]

                            if self.config["coteaching"] == True:
                                if self.config['abstention'] == True:
                                    loss_vid, loss_tab = loss_coteaching(pred_AS, ca_preds, target_AS, forget_rate, self.abstention_loss)
                                else:    
                                    loss_vid, loss_tab = loss_coteaching(pred_AS, ca_preds, target_AS, forget_rate)
                            else:
                                loss_vid = self._get_loss(pred_AS, target_AS, self.num_classes_AS)
                                loss_tab = self._get_loss(ca_preds, target_AS, self.num_classes_AS)

                            loss = frame_att_loss_weight*frame_att_loss + 0.5*ca_emb_loss + loss_vid_weight*loss_vid + loss_tab_weight*loss_tab + 0.05*(torch.mean(entropy_attention)) + \
                                   0.1*torch.mean(npair_loss) + 0.05*(torch.mean(multimodal_att_entropy))
                            
                            losses += [loss] 
                            losses_vid += [loss_vid_weight*loss_vid]
                            losses_tab += [loss_tab_weight*loss_tab]
                            losses_ca_emb += [0.5*ca_emb_loss]
                            losses_npair += [0.1*torch.mean(npair_loss)]
                            losses_frame_att += [frame_att_loss_weight*frame_att_loss]
                            losses_vid_entropy +=[0.05*(torch.mean(entropy_attention))]
                            losses_multimodal_entropy += [0.05*(torch.mean(multimodal_att_entropy))]

                        else:
                            pred_AS = self.model(cine, tab_data, split='Train') # Bx3xTxHxW
                            loss = self._get_loss(pred_AS, target_AS, self.num_classes_AS)
                            losses += [loss]
                    # Contrastive Learning
                    else:
                        cines = torch.cat([cine[0], cine[1]], dim=0)
                        if self.config['use_cuda']:
                            cines = cines.cuda()
                            tab_data = tab_data.cuda()
                            target_AS = target_AS.cuda()
                            target_B = target_B.cuda()
                        bsz = target_AS.shape[0]
                        features = self.model(cines, tab_data, split='Train') # Bx3xTxHxW
                        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                        if self.config['cotrastive_method'] == 'SupCon':
                            loss = self._get_loss(features, target_AS, self.num_classes_AS)
                        elif self.config['cotrastive_method'] == 'SimCLR':
                            loss = self._get_loss(features, target_AS, self.num_classes_AS)
                        else:
                            raise ValueError('contrastive method not supported: {}'.
                                             format(self.config['cotrastive_method']))

                        losses += [loss]
                        
                    # Calculate the training accuracy
                    _, predicted = torch.max(pred_AS, 1)
                    correct_predictions += (predicted == target_AS).sum().item()
                    total_samples += target_AS.size(0)

                    # Calculate the gradient.
                    # grad_loss = loss / gradient_accumulation_steps
                    loss.backward()

                    # Update the parameters according to the gradient.
                    #if (index + 1) % gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    # Zero the parameter gradients in the optimizer
                    self.optimizer.zero_grad() 
                    pbar.set_postfix_str("loss={:.4f}".format(loss.item()))
                    pbar.update()

            #loss_avg_AS = torch.mean(torch.stack(losses_AS)).item()
            #loss_avg_B = torch.mean(torch.stack(losses_B)).item()
            loss_avg = torch.mean(torch.stack(losses)).item() 
            loss_avg_vid = torch.mean(torch.stack(losses_vid)).item()
            loss_avg_tab = torch.mean(torch.stack(losses_tab)).item()
            loss_avg_ca_emb = torch.mean(torch.stack(losses_ca_emb)).item()
            loss_avg_npair = torch.mean(torch.stack(losses_npair)).item()
            loss_avg_frame_att = torch.mean(torch.stack(losses_frame_att)).item()
            loss_avg_vid_entropy = torch.mean(torch.stack(losses_vid_entropy)).item()
            loss_avg_multimodal_entropy = torch.mean(torch.stack(losses_multimodal_entropy)).item()
            #acc_AS, f1_B, val_loss = self.test(loader_va, mode="val")
            if self.config['cotrastive_method'] == 'CE' or self.config['cotrastive_method'] == 'Linear':
                ## Validation Data Evaluation 
                acc_AS, val_total_loss, val_loss_avg_vid, val_loss_avg_tab, val_loss_avg_ca_emb, val_loss_avg_npair, val_loss_avg_frame_att, val_loss_avg_vid_entropy, val_loss_avg_multimodal_entropy = self.test(loader_va, mode="val") #TODO - log all losses
                if self.config['use_wandb']:
                    wandb.log({##training losses
                               "tr_loss-total":loss_avg, 
                               "tr_loss-vid": loss_avg_vid,
                               "tr_loss-tab": loss_avg_tab,
                               "tr_loss-clip": loss_avg_ca_emb,
                               "tr_loss-npair": loss_avg_npair,
                               "tr_loss-frame_att": loss_avg_frame_att,
                               "tr_loss-vid_entropy": loss_avg_vid_entropy,
                               "tr_loss-multimodal-entropy": loss_avg_multimodal_entropy,
                               "tr_acc": (correct_predictions / total_samples),
                               "LR": self.optimizer.param_groups[0]["lr"],
                               ##val losses
                               "val_loss":val_total_loss, 
                               "val_loss-vid": val_loss_avg_vid,
                               "val_loss-tab": val_loss_avg_tab,
                               "val_loss-clip": val_loss_avg_ca_emb,
                               "val_loss-npair": val_loss_avg_npair,
                               "val_loss-frame_att": val_loss_avg_frame_att,
                               "val_loss-vid_entropy": val_loss_avg_vid_entropy,
                               "val_loss-multimodal-entropy": val_loss_avg_multimodal_entropy,
                               "val_AS_acc":acc_AS})
                    # wandb.log({"tr_loss_AS":loss_avg_AS, "tr_loss_B":loss_avg_B, "tr_loss":loss_avg,
                    #            "val_loss":val_loss, "val_B_f1":f1_B, "val_AS_acc":acc_AS})
                ## Test Data Evaluation    
                acc_AS_te, te_total_loss, _, _, _, _, _, _, _ = self.test(loader_te, mode="val")
                if self.config['use_wandb']:
                    wandb.log({ "te_loss":te_total_loss, "test_AS_acc":acc_AS_te})

                # Save model every epoch.
                self._save(self.checkpts_file)

                # Early stopping strategy.
                if acc_AS > best_va_acc:
                    # Save model with the best accuracy on validation set.
                    best_va_acc = acc_AS
                    #best_B_f1 = f1_B
                    self._save(self.bestmodel_acc)
                
                if val_total_loss < best_va_loss:
                    # Save model with lowest loss on val set
                    best_va_loss = val_total_loss
                    self._save(self.bestmodel_loss)
                # print(
                #     "Epoch: %3d, loss: %.5f/%.5f, val loss: %.5f, acc: %.5f/%.5f, top AS acc: %.5f/%.5f"
                #     % (epoch, loss_avg_AS, loss_avg_B, val_loss, acc_AS, f1_B, best_va_acc, best_B_f1)
                # )
                print(
                    "Epoch: %3d, loss: %.5f, train acc: %.5f, val loss: %.5f, val acc: %.5f, top AS acc: %.5f"
                    % (epoch, loss_avg, (correct_predictions / total_samples), val_total_loss, acc_AS, best_va_acc)
                ) 

                # Recording training losses and validation performance.
                self.train_losses += [loss_avg]
                self.valid_oas += [acc_AS]
                self.idx_steps += [epoch]
                
            elif self.config['cotrastive_method'] == 'SupCon' or self.config['cotrastive_method'] == 'SimCLR':
                if epoch%5==0:
                    val_acc = validation_constructive(self.model,self.config)
                    if self.config['use_wandb']:
                        wandb.log({"AMI":val_acc['AMI'],"NMI":val_acc['NMI'],
                                   "precision_at_1":val_acc['precision_at_1']})
                if (val_acc['precision_at_1'] > best_va_acc_supcon and self.config['cotrastive_method'] == 'SupCon'):
                    # Save model with the best accuracy on validation set.
                    best_va_acc_supcon = val_acc['precision_at_1']
                    self._save(self.bestmodel_file_contrastive)
                    print('Model Saved')
                
                if self.config['cotrastive_method'] == 'SimCLR':
                    self._save(self.bestmodel_file_contrastive)
                    print('Model Saved')
                    
                if self.config['use_wandb']:
                    wandb.log({"contrastive_loss":loss_avg})
                    
                print(
                    "Epoch: %3d, loss: %.5f,precision_at_1: %.5f"
                    % (epoch, loss_avg,val_acc['precision_at_1'])
                ) 
               
            
            # modify the learning rate
            self.scheduler.step(val_total_loss) ##TODO - could also modify based only on the validation loss-vid metric    

    @torch.no_grad()
    def test(self, loader_te, mode="test"):
        """Estimating the performance of model on the given dataset."""
        # Choose which model to evaluate.
        if mode=="test":
            self._restore(self.bestmodel_acc)
        # Switch the model into eval mode.
        self.model.eval()

        conf_AS = np.zeros((self.num_classes_AS, self.num_classes_AS))
        #conf_B = np.zeros((2,2))
        forget_rate = 0.0625
        losses = []
        losses_vid = []
        losses_tab = []
        losses_ca_emb = []
        losses_npair = []
        losses_frame_att = []
        losses_vid_entropy = []
        losses_multimodal_entropy = []
        preds = []
        gt = []
        for data in tqdm(loader_te):
            cine = data[0]
            tab_data = data[1]
            target_AS = data[2]
            target_B = data[3]
            # Transfer data from CPU to GPU.
            # Cross Entropy Training
            if self.config['cotrastive_method'] == 'CE' or self.config['cotrastive_method'] == "Linear":
                # Transfer data from CPU to GPU.
                if self.config['use_cuda']:
                    if self.config['model'] == 'slowfast':
                        cine = [c.cuda() for c in cine]
                    else:
                        cine = cine.cuda()
                    tab_data = tab_data.cuda()
                    target_AS = target_AS.cuda()
                    target_B = target_B.cuda()
                    
                if self.config['model'] == "FTC_TAD":
                    pred_AS,entropy_attention,outputs, att_weight, ca_preds, learned_emb, ca_embed, ca_att_weight, multimodal_att_entropy = self.model(cine, tab_data, split='Train') # Bx3xTxHxW #Train split allows us to get multimodal outputs to calculate losses
                else:
                    pred_AS = self.model(cine, tab_data, split='Test') # Bx3xTxHxW

                # Calculate loss between learned joint embeddings
                ca_emb_loss, _, _ = self.embed_loss_cos(learned_emb, ca_embed, target_AS)

                # Calculating temporal coherent npair loss
                similarity_matrix = (torch.bmm(outputs,outputs.permute((0,2,1)))/1024)
                self.pos_e = self.pos.repeat(len(pred_AS),1,1)
                self.neg_e = self.neg.repeat(len(pred_AS),1,1)
                npair_loss = torch.mean(-torch.sum(self.pos_e*similarity_matrix,dim =2) + 
                    torch.log(torch.exp(torch.sum(self.pos_e*similarity_matrix,dim=2))+
                    torch.sum(self.neg_e*torch.exp(self.neg_e*similarity_matrix),dim=2)), dim = 1)

                # remove video entropy loss
                if not self.config["video_entropy"]:
                    entropy_attention = entropy_attention * 0
                
                # Alignment between video and multimodal attention weights 
                frame_att_loss_weight = self.config["frame_att_loss_weight"]
                if not self.config["multimodal_att_entropy"]:
                        multimodal_att_entropy = multimodal_att_entropy * 0  
                if self.config["frame_attention_loss"] == "cosine_sim":
                    cos = torch.nn.CosineSimilarity()
                    frame_att_loss = cos(ca_att_weight, att_weight).mean().item()                                 
                elif self.config["frame_attention_loss"] == "kl_div":
                    ## model outputs need to be log probabilities, targets need to be probabilities
                    att_weight = torch.log(att_weight)
                    frame_att_loss = F.kl_div(input=att_weight, target=ca_att_weight, log_target=False, reduction='mean')
                else:
                    frame_att_loss = torch.zeros(1).cuda()

                # Coteaching + CE losses
                loss_vid_weight = self.config["loss_vid_weight"]
                loss_tab_weight = self.config["loss_tab_weight"]

                if self.config["coteaching"] == True:
                    if self.config['abstention'] == True:
                        loss_vid, loss_tab = loss_coteaching(pred_AS, ca_preds, target_AS, forget_rate, self.abstention_loss)
                    else:    
                        loss_vid, loss_tab = loss_coteaching(pred_AS, ca_preds, target_AS, forget_rate)
                else:
                    loss_vid = self._get_loss(pred_AS, target_AS, self.num_classes_AS)
                    loss_tab = self._get_loss(ca_preds, target_AS, self.num_classes_AS)
         
                loss = frame_att_loss_weight*frame_att_loss + 0.5*ca_emb_loss + loss_vid_weight*loss_vid + loss_tab_weight*loss_tab + 0.05*(torch.mean(entropy_attention)) + \
                        0.1*torch.mean(npair_loss) + 0.05*(torch.mean(multimodal_att_entropy))
                losses += [loss]
                losses_vid += [loss_vid_weight*loss_vid]
                losses_tab += [loss_tab_weight*loss_tab]
                losses_ca_emb += [0.5*ca_emb_loss]
                losses_npair += [0.1*torch.mean(npair_loss)]
                losses_frame_att += [frame_att_loss_weight*frame_att_loss]
                losses_vid_entropy +=[0.05*(torch.mean(entropy_attention))]
                losses_multimodal_entropy += [0.05*(torch.mean(multimodal_att_entropy))]

            # Contrastive Learning
            else:
                cines = torch.cat([cine[0], cine[1]], dim=0)
                if self.config['use_cuda']:
                    cines = cines.cuda()
                    tab_data = tab_data.cuda()
                    target_AS = target_AS.cuda()
                    target_B = target_B.cuda()
                bsz = target_AS.shape[0]
                features = self.model(cines) # Bx3xTxHxW
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                if self.config['cotrastive_method'] == 'SupCon':
                    loss = self._get_loss(features, target_AS, self.num_classes_AS)
                elif self.config['cotrastive_method'] == 'SimCLR':
                    loss = self._get_loss(features, target_AS, self.num_classes_AS)
                else:
                    raise ValueError('contrastive method not supported: {}'.
                                     format(self.config['cotrastive_method']))
                losses += [loss]
            
            if self.config["abstention"]:
                pred_AS = pred_AS[:, : self.num_classes_AS]

            #argmax_pred_AS = torch.argmax(pred_AS, dim=1)
            #argmax_pred_B = torch.argmax(pred_B, dim=1)
            if self.config['cotrastive_method'] == 'CE' or self.config['cotrastive_method'] == 'Linear':
                argm_AS, _, _, _, _, _ = self._get_prediction_stats(pred_AS, self.num_classes_AS)
                #argm_B, _, _, _, _ = self._get_prediction_stats(pred_B, 2)
                conf_AS = utils.update_confusion_matrix(conf_AS, target_AS.cpu(), argm_AS.cpu())
                #conf_B = utils.update_confusion_matrix(conf_B, target_B.cpu(), argm_B.cpu())
            
            preds.append(argm_AS.cpu().numpy())
            gt.append(target_AS.cpu().numpy())
        if self.config['cotrastive_method'] == 'CE' or self.config['cotrastive_method'] == 'Linear':    
            total_loss_avg = torch.mean(torch.stack(losses)).item()
            loss_avg_vid = torch.mean(torch.stack(losses_vid)).item()
            loss_avg_tab = torch.mean(torch.stack(losses_tab)).item()
            loss_avg_ca_emb = torch.mean(torch.stack(losses_ca_emb)).item()
            loss_avg_npair = torch.mean(torch.stack(losses_npair)).item()
            loss_avg_frame_att = torch.mean(torch.stack(losses_frame_att)).item()
            loss_avg_vid_entropy = torch.mean(torch.stack(losses_vid_entropy)).item()
            loss_avg_multimodal_entropy = torch.mean(torch.stack(losses_multimodal_entropy)).item()
            preds = [x for xs in preds for x in xs]
            gt = [x for xs in gt for x in xs]
            acc_AS = balanced_accuracy_score(gt, preds) #utils.balanced_acc_from_confusion_matrix(conf_AS)
            print(conf_AS)
            #f1_B = utils.f1_from_confusion_matrix(conf_B)

            # Switch the model into training mode
            self.model.train()
            return acc_AS, total_loss_avg, loss_avg_vid, loss_avg_tab, loss_avg_ca_emb, loss_avg_npair, loss_avg_frame_att, loss_avg_vid_entropy, loss_avg_multimodal_entropy
        else:
            loss_avg = torch.mean(torch.stack(losses)).item()
            self.model.train()
            return loss_avg
    
    @torch.no_grad()
    def test_comprehensive(self, loader, split, mode="test",record_embeddings=False):
        """Logs the network outputs in dataloader
        computes per-patient preds and outputs result to a DataFrame"""
        print('NOTE: test_comprehensive mode uses batch_size=1 to correctly display metadata')
        # Choose which model to evaluate.
        if mode=="test":
            self._restore(self.bestmodel_acc)
            #self._restore(self.bestmodel_loss)
        # Switch the model into eval mode.
        self.model.eval()
        fn, patient, echo, view, age, lv, as_label,bicuspid = [], [], [], [], [], [],[], []
        target_AS_arr, target_B_arr, pred_AS_arr, pred_logits_arr = [], [], [], []
        max_AS_arr, entropy_AS_arr, vacuity_AS_arr, uni_AS_arr = [], [], [], []
        att_weight_arr, ca_att_weight_arr = [], []
        #max_B_arr, entropy_B_arr, vacuity_B_arr = [], [], []
        predicted_qual = []
        embeddings = []
        
        for cine, tab_info, target_AS, target_B, data_info, cine_orig in tqdm(loader):
            # collect the label info
            target_AS_arr.append(int(target_AS[0]))
            target_B_arr.append(int(target_B[0]))
            # Transfer data from CPU to GPU.
            if self.config['use_cuda']:
                cine = cine.cuda()
                tab_info = tab_info.cuda()
                target_AS = target_AS.cuda()
                target_B = target_B.cuda()
                
            # collect metadata from data_info
            fn.append(data_info['path'][0])
            patient.append(int(data_info['patient_id'][0]))
            echo.append(int(data_info['Echo ID#'][0]))
            view.append(data_info['view'][0])
            age.append(int(data_info['age'][0]))
            #lv.append(float(data_info['LVMass indexed'][0]))
            as_label.append(data_info['as_label'][0])
            bicuspid.append(data_info['Bicuspid'][0])
            # pvq = (data_info['predicted_view_quality'][0] * 
            #        data_info['predicted_view_probability'][0]).cpu().numpy()
            # predicted_qual.append(pvq)
            
            # get the model prediction
            # pred_AS, pred_B = self.model(cine) #1x3xTxHxW
            if self.config['model'] == "FTC_TAD":
                pred_AS,entropy_attention,outputs, att_weight, _, _, embedding, ca_att_weight, multimodal_att_entropy = self.model(cine, tab_info, split='Train') #TODO CHANGE
                # Bx3xTxHxW
            else:
                pred_AS = self.model(cine, tab_info, split='Test') # Bx3xTxHxW
            # collect the model prediction info
            argm, max_p, ent, vac, uni, logits = self._get_prediction_stats(pred_AS, self.num_classes_AS)
            pred_AS_arr.append(argm.cpu().numpy()[0])
            max_AS_arr.append(max_p.cpu().numpy()[0])
            entropy_AS_arr.append(ent.cpu().numpy()[0])
            ca_att_weight_arr.append(ca_att_weight.cpu().numpy()[0])
            att_weight_arr.append(att_weight.cpu().numpy()[0])

            if self.loss_type == 'evidential':
                vacuity_AS_arr.append(vac.cpu().numpy()[0])
            else:
                vacuity_AS_arr.append(vac)
            uni_AS_arr.append(uni[0])
            pred_logits_arr.append(logits.cpu().numpy()[0])
            
            if record_embeddings:
                embeddings += [embedding[0].squeeze().cpu().numpy()]

                
        # compile the information into a dictionary
        d = {'path':fn, 'id':patient, 'echo_id': echo, 'view':view, 'age':age, 'as':as_label, 'bicuspid': bicuspid ,
             'GT_AS':target_AS_arr, 'pred_AS':pred_AS_arr, 'max_AS':max_AS_arr,
             'ent_AS':entropy_AS_arr, 'vac_AS':vacuity_AS_arr, 'uni_AS':uni_AS_arr,
             'pred_logits_AS': pred_logits_arr, 
             'att_weight_arr':att_weight_arr, 'ca_att_weight_arr':ca_att_weight_arr
             # 'GT_B':target_B_arr, 'pred_B':pred_B_arr, 'max_B':max_B_arr,
             # 'ent_B':entropy_B_arr, 'vac_B':vacuity_B_arr, 
             }
        df = pd.DataFrame(data=d)
        # save the dataframe
        # test_results_file = os.path.join(self.log_dir, mode+".csv") 
        test_results_file = os.path.join(self.log_dir, split + "_fixed" +".csv") 
        df.to_csv(test_results_file)
        if record_embeddings:
            embeddings = np.array(embeddings)
            print(embeddings.shape)
            num_batches, b, d = embeddings.shape
            print(num_batches, b, d)
            embeddings = np.reshape(embeddings, (num_batches, b*d))
            tsne_save_file =  os.path.join(self.log_dir, mode+"_tsne.html")
            plot_tsne_visualization(X=embeddings, y=as_label, info=fn, title=tsne_save_file , b = bicuspid)


# if __name__ == "__main__":
#     """Main for mock testing."""
#     from get_config import get_config
#     from dataloader.as_dataloader_revision import get_as_dataloader
#     from get_model import get_model

#     config = get_config()
    
#     if config['use_wandb']:
#         run = wandb.init(project="as_v2", entity="guangnan", config=config)
    
#     model = get_model(config)
#     net = Network(model, config)
#     dataloader_tr = get_as_dataloader(config, split='train', mode='train')
#     dataloader_va = get_as_dataloader(config, split='val', mode='val')
#     dataloader_te = get_as_dataloader(config, split='test', mode='test')
    
#     if config['mode']=="train":
#         net.train(dataloader_tr, dataloader_va)
#         net.test_comprehensive(dataloader_te, mode="test")
#     if config['mode']=="test":
#         net.test_comprehensive(dataloader_te, mode="test")
#     if config['use_wandb']:
#         wandb.finish()
