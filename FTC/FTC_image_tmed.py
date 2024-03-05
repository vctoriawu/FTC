from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.dropout import Dropout
from transformers import BertConfig, BertModel
from FTC.util.misc import construct_ASTransformer
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
from einops import rearrange

class CrossAttention(nn.Module):
    """
    Apply cross attention between x tensor and tab_x tensor.

    Arguments:
        dim: 
        context_dim:
        heads: number of attention heads.
        dim_head: dimension of q, k, v 
    


    """
    def __init__(
        self,
        dim=1000, #dimension of img input
        context_dim=1000, #dimension of tab input
        heads = 12,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.tab_norm = nn.LayerNorm(context_dim)
        self.vid_norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias = False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias = False)
        self.norm_q = nn.LayerNorm(inner_dim)
        self.norm_k = nn.LayerNorm(inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tab_x):
        #get shape of video input
        b, f, _ = x.shape
        h = self.heads
        kv_input = self.tab_norm(tab_x)
        # q, k, v = self.to_q(x), self.to_k(kv_input), self.to_v(kv_input)
        q = self.norm_q(self.to_q(self.vid_norm(x)))
        k = self.norm_k(self.to_k(kv_input))
        v = self.to_v(kv_input)
        q, k, v = map(lambda t: rearrange(tensor=t, pattern='b f (h d) -> b h f d', h=h), (q, k, v))
        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        # attention
        attn = sim.softmax(dim = -1) 
        attn = self.dropout(attn)

        # aggregate
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # merge heads
        out = rearrange(out, 'b h f d -> b f (h d)')
        
        return self.to_out(out)
    
class EmbeddingMappingFunction(nn.Module):
    def __init__(self, video_dim, hidden_dim, tabular_dim, num_heads=8, num_transformer_layers=2):
        super(EmbeddingMappingFunction, self).__init__()
        self.fc1 = nn.Linear(video_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, tabular_dim)
        #self.fc3 = nn.Linear(hidden_dim, tabular_dim)
        
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x  # Preserve the input as the residual

        x = self.fc1(x)
        x = self.layernorm(x)
        x = self.relu(x)
        x = self.fc2(x)
        #x = self.layernorm(x)
        #x = self.relu(x)
        #x = self.fc3(x)

        # Add the residual connection
        x += residual
        return x

class wrn_video(nn.Module):
    def __init__(self, num_classes_diagnosis, num_classes_view, emb_dim, 
                loaded_parameters):
        super(wrn_video, self).__init__()
        self.num_classes_diagnosis = num_classes_diagnosis

        self.model = models.wide_resnet50_2(pretrained=True)
        # new layer
        self.relu = nn.ReLU(inplace=True)
        self.fc_view = nn.Linear(1000, num_classes_view)
        self.fc_diagnosis = nn.Linear(1000, num_classes_diagnosis)
        self.attention = nn.Linear(1000, 1)

        # Map video embeddings to video+tab embeddings
        embedding_dim = emb_dim
        self.map_embed = EmbeddingMappingFunction(embedding_dim, embedding_dim, embedding_dim)
        self.vt_proj_head = EmbeddingMappingFunction(embedding_dim, embedding_dim, embedding_dim)

        # Get the parameters for the tabular model
        loaded_categories = loaded_parameters['categories']
        loaded_num_continuous = loaded_parameters['num_continuous']
        loaded_dim = loaded_parameters['dim']
        loaded_depth = loaded_parameters['depth']
        loaded_heads = loaded_parameters['heads']
        loaded_dim_head = loaded_parameters['dim_head']
        loaded_dim_out = loaded_parameters['dim_out']
        loaded_num_special_tokens = loaded_parameters['num_special_tokens']
        loaded_attn_dropout = loaded_parameters['attn_dropout']
        loaded_ff_dropout = loaded_parameters['ff_dropout']
        loaded_hidden_dim = loaded_parameters['hidden_dim']
        loaded_numerical_features = loaded_parameters['numerical_features']
        loaded_classification = loaded_parameters['classification']
        loaded_emb_type = loaded_parameters['emb_type']
        loaded_numerical_bins = loaded_parameters['numerical_bins']

        self.tab_embed = construct_ASTransformer(loaded_categories, loaded_num_continuous, loaded_numerical_features,
                                                loaded_dim, loaded_depth, loaded_heads, loaded_dim_head, loaded_dim_out,
                                                loaded_num_special_tokens, loaded_attn_dropout, loaded_ff_dropout,
                                                loaded_hidden_dim, loaded_classification,
                                                loaded_numerical_bins, loaded_emb_type)
        self.tab_embed.load_state_dict(torch.load('../as_transformer.pth'))
        
        # Set requires_grad to False for all parameters
        for param in self.tab_embed.parameters():
            param.requires_grad = False

        tab_emb_dims=[16, 32, 72]
        self.cross_attention = CrossAttention(dim=1000, 
                                context_dim=tab_emb_dims[2], 
                                heads = 12,
                                dim_head = 72,
                                dropout = 0.)

    def forward(self, x, tab_x, split):        
        # [B, C, F, H, W] -> [B, F, C, H, W]
        x = x.permute(0,2,1,3,4)

        nB, nF, nC, nH, nW = x.shape
        # Merge batch and frames dimension
        x = x.contiguous().view(nB*nF,nC,nH,nW)

        outputs = self.model(x)
        if split=='Train':
            b, _ = tab_x.shape
            tab_x = torch.unsqueeze(tab_x, dim=-1) 
                
            # B x F x 1
            #Tranform x feature values to higher dim using a shared mlp layer
            _, tab_x = self.tab_embed(tab_x)

            outputs = outputs.view(-1, nF, 1000)
            #cross attention between video and tabular embeddings
            ca_outputs = self.cross_attention(outputs, tab_x)
            ca_outputs = self.vt_proj_head(ca_outputs)
            b, f, d = ca_outputs.shape

            # we want to get ca_outputs from outputs using our embedding mapping function
            learned_joint_emb = self.map_embed(outputs)

            ca_outputs = ca_outputs.view(b*f, d)
            learned_joint_emb = learned_joint_emb.view(b*f, d)
        else:
            print("Cross-attention module has been skipped.")
            outputs = outputs.view(-1, nF, 1000)
            b, f, d = outputs.shape
            outputs = self.map_embed(outputs)
            learned_joint_emb = outputs
            ca_outputs = outputs
            ca_outputs = ca_outputs.view(b*f, d)
            learned_joint_emb = learned_joint_emb.view(b*f, d)

        # cross attention preds
        ca_out = self.relu(ca_outputs)
        out = self.relu(learned_joint_emb)

        out = out.view(-1, nF, 1000)
        ca_out = ca_out.view(-1, nF, 1000)
        pred_view = self.fc_view(out)
        pred_as = self.fc_diagnosis(out) #[B, F, 4]
        pred_as_ca = self.fc_diagnosis(ca_out)
        att = self.attention(out)
        ca_att = self.attention(ca_out)

        # View Analysis
        view_prob = torch.nn.functional.softmax(pred_view,dim=2)
        relevance = torch.cumsum(view_prob,dim=2)[:,:,1] #[B, 32]
        relevance  = relevance.unsqueeze(2).repeat(1,1,self.num_classes_diagnosis) #[B, 32, 1] -> [B, 32, 4]

        # attention weights B x F x 1
        att_weight = nn.functional.softmax(att, dim=1) 
        ca_att_weight = nn.functional.softmax(ca_att, dim=1) 
        # B x T x 4   =>  B x 4
        as_prediction = ((relevance*pred_as) * att_weight).sum(1, keepdim=False)
        as_ca_prediction = ((relevance*pred_as_ca) * ca_att_weight).sum(1, keepdim=False)

        #B x T x 2 => B x 2
        pred_view = torch.mean(pred_view, dim=1)

        return pred_view , as_prediction, as_ca_prediction, learned_joint_emb, ca_outputs