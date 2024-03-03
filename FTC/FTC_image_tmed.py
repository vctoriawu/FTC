from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.dropout import Dropout
from transformers import BertConfig, BertModel
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn

class wrn_video(nn.Module):
    def __init__(self, num_classes_diagnosis, num_classes_view):
        super(wrn_video, self).__init__()
        self.num_classes_diagnosis = num_classes_diagnosis

        self.model = models.wide_resnet50_2(pretrained=True)
        # new layer
        self.relu = nn.ReLU(inplace=True)
        self.fc_view = nn.Linear(1000, num_classes_view)
        self.fc_diagnosis = nn.Linear(1000, num_classes_diagnosis)
        self.attention = nn.Linear(1000, 1)

    def forward(self, x):        
        # [B, C, F, H, W] -> [B, F, C, H, W]
        x = x.permute(0,2,1,3,4)

        nB, nF, nC, nH, nW = x.shape
        
        # Merge batch and frames dimension
        x = x.contiguous().view(nB*nF,nC,nH,nW)
        
        out = self.relu(self.model(x))
        out = out.view(-1, nF, 1000)
        pred_view = self.fc_view(out)
        pred_as = self.fc_diagnosis(out) #[B, F, 4]
        att = self.attention(out)
        
        # View Analysis
        view_prob = torch.nn.functional.softmax(pred_view,dim=2)
        relevance = torch.cumsum(view_prob,dim=2)[:,:,1] #[B, 32]
        relevance  = relevance.unsqueeze(2).repeat(1,1,self.num_classes_diagnosis) #[B, 32, 1] -> [B, 32, 4]

        # attention weights B x F x 1
        att_weight = nn.functional.softmax(att, dim=1) 
        # B x T x 4   =>  B x 4
        as_prediction = ((relevance*pred_as) * att_weight).sum(1, keepdim=False)

        #B x T x 2 => B x 2
        pred_view = torch.mean(pred_view, dim=1)
        
        return pred_view , as_prediction
