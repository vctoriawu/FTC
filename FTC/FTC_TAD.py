from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.dropout import Dropout
from transformers import BertConfig, BertModel
import torch
import torch.nn as nn
import torchvision
#from ResNetAE.ResNetAE import ResNetAE
from FTC.posembedding import build_position_encoding
from FTC.deformable_transformer import DeformableTransformer
from FTC.util.misc import NestedTensor, construct_ASTransformer
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
        dim=1024, #dimension of img input
        context_dim=1024, #dimension of tab input
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
        self.layernorm = nn.LayerNorm(hidden_dim)

        # Self-attention layers
        '''self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=video_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, 
            num_layers=num_transformer_layers
        )

        # Feed-forward layer
        self.fc = nn.Linear(video_dim, tabular_dim)
        self.layernorm = nn.LayerNorm(video_dim)'''

    def forward(self, x):
        residual = x  # Preserve the input as the residual

        x = self.fc1(x)
        x = self.layernorm(x)
        x = self.relu(x)
        x = self.fc2(x)

        # Add the residual connection
        x += residual

        # Apply self-attention using Transformer
        '''x = self.transformer_encoder(x)

        # Feed-forward layer
        x = self.fc(x)
        # x = self.layernorm(x)'''

        return x
    
class Reduce(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        
        # sorted_scores,_ = x.sort(descending=True, dim=1)
        # topk_scores = sorted_scores[:, :9, :]
        # x = torch.mean(topk_scores, dim=1)
        x = x.mean(dim=1)
        return x

class Inception(nn.Module):
    def __init__(self, emb_dim, in_channels=1, out_channels_per_conv=16):
        super().__init__()
        
        self.five       = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_per_conv, kernel_size=(5,emb_dim), stride=(1,1), padding=(2,0), bias=False)
        self.three      = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_per_conv, kernel_size=(3,emb_dim), stride=(1,1), padding=(1,0), bias=False)
        self.stride2    = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_per_conv, kernel_size=(3,emb_dim), stride=(1,2), padding=(1,0), bias=False)
    
    def forward(self,x):
        out_five    = self.five(x)
        out_three   = self.three(x)
        out_stride2 = self.stride2(x)
        
        output = torch.cat((out_five, out_three, out_stride2),dim=1).permute(0,3,2,1) # Cat on channel dim
        return output
    

        
# New Multi Branch Auto Encoding Transformer
class FTC(nn.Module):
    def __init__(self, 
                AE, 
                pos_embed, 
                transformer, 
                embedding_dim, 
                use_tab, 
                cross_attention, 
                tab_input_dim, 
                tab_emb_dims,
                loaded_parameters, 
                pretrained = False):
        super(FTC, self).__init__()
        
        last_features = 4
        self.pretrained = pretrained
        self.use_tab = use_tab
        self.tab_input_dim = tab_input_dim
        self.tab_emb_dims = tab_emb_dims
        
        self.AE = AE
        if self.pretrained == True:
            from pathlib import Path
            checkpoint = torch.load(Path('/AS_Neda/FTC/logs/resnet18/best_model_cont.pth'))
            prefix = 'module.model.'
            n_clip = len(prefix)
            adapted_dict = {k[n_clip:]: v for k, v in checkpoint["model"].items()
                            if k.startswith(prefix)}
            self.AE.load_state_dict(adapted_dict, strict=False)
            print('Stage1 weights Loaded res18')

        self.pos_embed = pos_embed
        self.transformer = transformer

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

        self.aorticstenosispred = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim//2, bias=True),
            nn.LayerNorm(embedding_dim//2),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Linear(in_features=embedding_dim//2, out_features=4, bias=True),
            # nn.Softmax(dim=2),
            # Reduce(),
            )
        
        self.attentionweights = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim//2, bias=True),
            nn.LayerNorm(embedding_dim//2),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Linear(in_features=embedding_dim//2, out_features=1, bias=True),
            # nn.Softmax(dim=2),
            # Reduce(),
            )

        #self.vf = frames_per_video
        self.em = embedding_dim

        # load the pretrained weights TODO: Make this a config
        # self.load_state_dict(torch.load("best_model.pth")["model"], strict=False)

        self.cross_attention = cross_attention

        self.tab_embed = construct_ASTransformer(loaded_categories, loaded_num_continuous, loaded_numerical_features,
                                                 loaded_dim, loaded_depth, loaded_heads, loaded_dim_head, loaded_dim_out,
                                                 loaded_num_special_tokens, loaded_attn_dropout, loaded_ff_dropout,
                                                 loaded_hidden_dim, loaded_classification,
                                                 loaded_numerical_bins, loaded_emb_type)
        self.tab_embed.load_state_dict(torch.load('../as_transformer.pth'))
        
        # Set requires_grad to False for all parameters
        for param in self.tab_embed.parameters():
            param.requires_grad = False

        # Map video embeddings to video+tab embeddings
        self.map_embed = EmbeddingMappingFunction(embedding_dim, embedding_dim, embedding_dim)

    def forward(self, x, tab_x, split):   
        # Video dimension (B x F x C x H x W)
        x = x.permute(0,2,1,3,4)
        
        nB, nF, nC, nH, nW = x.shape
        # Merge batch and frames dimension
        x = x.contiguous().view(nB*nF,nC,nH,nW)
        
        if self.pretrained:
            # with torch.no_grad():
            # (BxF) x C x H x W => (BxF) x Emb
            embeddings = self.AE(x).squeeze()
            #embeddings = torch.nn.functional.normalize(embeddings, dim=0)
        else:
            # (BxF) x C x H x W => (BxF) x Emb
            embeddings = self.AE(x).squeeze()
            
            
        embeddings_reshaped = embeddings.view(nB, nF, self.em )
        embeddings_reshaped = embeddings_reshaped.permute(0,2,1)
            
        # Creatining porsitional embedding and nested tensor
        mask = torch.ones((nB, nF), dtype=torch.bool).cuda()
        #embeddings_reshaped = embeddings_reshaped.cuda()
        samples = NestedTensor(embeddings_reshaped, mask)
        pos = self.pos_embed(samples).cuda()
        
        #  B x F x Emb
        outputs = self.transformer([embeddings_reshaped],[pos],[mask])

        #integrate tab data
        if split=='Train':
            if self.use_tab:
                tab_x = torch.unsqueeze(tab_x, dim=-1) 
                
                # B x F x 1
                #Tranform x feature values to higher dim using a shared mlp layer
                _, tab_x = self.tab_embed(tab_x)

                #cross attention between video and tabular embeddings
                ca_outputs = self.cross_attention(outputs, tab_x)

                # we want to get ca_outputs from outputs using our embedding mapping function
                learned_joint_emb = self.map_embed(outputs)
        else:
            print("Cross-attention module has been skipped.")
            outputs = self.map_embed(outputs)
            learned_joint_emb = outputs
            ca_outputs = outputs

        method = "attention"   # "average","emb_mag","attention",
        if method == "average":
            # B x F x Emb => B x T x 4
            as_prediction = self.aorticstenosispred(outputs)
            # B x T x 4   =>  B x 4
            as_prediction = as_prediction.mean(dim=1)
            
        elif method == "emb_mag":
            # B x F x Emb => B x T x 4
            as_prediction = self.aorticstenosispred(outputs)
            # B x T x 4   =>  B x K x 4
            idx_act_feat = max_index[:,:9].unsqueeze(2).expand([-1, -1, 4])
            as_prediction = as_prediction.gather(1,idx_act_feat)
            as_prediction = as_prediction.mean(dim=1)
            
        elif method == "attention":
            # B x F x Emb => B x T x 4
            as_prediction = self.aorticstenosispred(outputs) #learned_joint_emb)

            # cross attention preds
            if split=='Train':
                if self.use_tab:
                    as_ca_predictions = self.aorticstenosispred(ca_outputs)
            else:
                as_ca_predictions = as_prediction
            
            # attention weights B x F x 1
            att_weight = self.attentionweights(outputs) #learned_joint_emb)
            #print(att_weight.shape,outputs.shape)
            att_weight = nn.functional.softmax(att_weight, dim=1)
            # B x T x 4   =>  B x 4
            as_prediction = (as_prediction * att_weight).sum(1)
            as_ca_predictions = (as_ca_predictions * att_weight).sum(1)
            # Calculating the entropy for attention
            entropy_attention = torch.sum(-att_weight*torch.log(att_weight), dim=1)
            
        elif method == "attention_resbranch":
            # B x F x Emb => B x T x 4
            as_prediction = self.aorticstenosispred(outputs)
            
            # attention weights B x F x 1
            att_weight = self.attentionweights(embeddings_reshaped.permute(0,2,1))
            att_weight = nn.functional.softmax(att_weight, dim=1)
            # B x T x 4   =>  B x 4
            as_prediction = (as_prediction * att_weight).sum(1)
            
            # Calculating the entropy for attention
            entropy_attention = torch.sum(-att_weight*torch.log(att_weight), dim=1)

        return as_prediction,entropy_attention,outputs,att_weight,as_ca_predictions, learned_joint_emb, ca_outputs

def get_model_tad(emb_dim, 
              tab_input_dim, 
              tab_emb_dims, 
              img_per_video, 
              num_hidden_layers = 16,
              intermediate_size = 8192,
              rm_branch = None,
              use_conv = False,
              attention_heads=16,
              use_tab=True
              ):
    
    model_res = torchvision.models.resnet18()
    dim_in = model_res.fc.in_features
    model_res.fc =  nn.Linear(dim_in, 1024)

    # Parameters for tabular model
    loaded_parameters = torch.load("../model_parameters.pth")
    
    # Setup model
    pos_embed = build_position_encoding(1024)
    transformer = DeformableTransformer(
            d_model=1024,
            nhead=8,
            num_encoder_layers=2,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            return_intermediate_dec=True,
            num_feature_levels=1,
            dec_n_points=4,
            enc_n_points=4)
    
    cross_attention = CrossAttention(dim=1024, 
        context_dim=tab_emb_dims[2], 
        heads = 12,
        dim_head = 72,
        dropout = 0.)
        
    model = FTC(AE=model_res,
                pos_embed=pos_embed,
                transformer=transformer, 
                embedding_dim=emb_dim, 
                use_tab=use_tab, 
                cross_attention=cross_attention, 
                tab_input_dim=tab_input_dim, 
                tab_emb_dims=tab_emb_dims,
                loaded_parameters=loaded_parameters, 
                pretrained=False,) 
    
    return model