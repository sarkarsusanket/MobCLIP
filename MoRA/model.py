import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# from satclip.sentinel.terramind import emb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

AUX_MOD = ["ACS", "ASR", "BSUM", "ESRI", "URBANICITY", "DHC", "RETAILDEMAND", "Text"]
INPUT_DIMS = {
    "ACS": 2486,
    "ASR": 1703,
    "BSUM": 255,
    "ESRI": 2566,
    "URBANICITY": 43,
    "DHC": 1883,
    "RETAILDEMAND": 127,
    "Text": 1024,
}

class LightGCN(nn.Module):
    def __init__(self,all_mob_features, num_layers):
        super(LightGCN, self).__init__()
        
        self.num_layers = num_layers
        self.ebds = nn.Parameter(all_mob_features)   
        
        
    def forward(self, adj):
        embeds = self.ebds
        embeds_list = [embeds]
        for layer in range(self.num_layers):
            embeddings = torch.spmm(adj, embeds_list[-1])
            embeds_list.append(embeddings)

        # Aggregate embeddings from all layers
        all_embeddings = torch.stack(embeds_list, dim=0)
        all_embeddings = torch.sum(all_embeddings, dim=0)
        self.final_embeds = all_embeddings

        return all_embeddings

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        # Ensure hidden_dims is a list
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        all_dims = [input_dim] + hidden_dims + [output_dim]

        self.layers = nn.ModuleList([
            nn.Linear(all_dims[i], all_dims[i+1])
            for i in range(len(all_dims) - 1)
        ])

        self.activation = nn.GELU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation to all layers except the last
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x



class LinearHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearHead, self).__init__()
        self.head = nn.Linear(input_dim, output_dim)  
   

    def forward(self, x):
        x = self.head(x)
        return x
    

class MobCLIP(nn.Module):
    def __init__(
        self,
        embedding_dim,
        mob_features,
        gnn_layers,
        scale,
    ):
        super(MobCLIP, self).__init__()

        self.embedding_dim = embedding_dim
        self.mob_features = mob_features
        self.gnn_layers = gnn_layers
        self.scale = scale

        self.mob_lightgcn = LightGCN(
            all_mob_features=self.mob_features,
            num_layers=self.gnn_layers,
        )

        # Better as a buffer or parameter, so it moves with the module
        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1 / self.scale), dtype=torch.float32)
        )

        # Use ModuleDict so Lightning/DDP knows about these submodules
        self.mod_encoders = nn.ModuleDict()
        for mod in AUX_MOD:
            if mod in ["Text", "Image"]:
                self.mod_encoders[mod] = LinearHead(
                    input_dim=INPUT_DIMS[mod],
                    output_dim=embedding_dim,
                )
            else:
                self.mod_encoders[mod] = MLPEncoder(
                    input_dim=INPUT_DIMS[mod],
                    hidden_dims=[512, 256],
                    output_dim=embedding_dim,
                )

        self.apply(self.init_weights)
 
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):

            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
            


    
    def forward(self, batch, mob_adj, global_indices = None
               ):
        
        logits = {}
        clip_embeddings = {}


        global_mob_ebd = self.mob_lightgcn(mob_adj)
        mob_ebd = global_mob_ebd[global_indices]
        mob_embeddings = mob_ebd / mob_ebd.norm(dim=1, keepdim=True) 
        clip_embeddings["mob"] = mob_embeddings

        logit_scale = self.logit_scale.exp()

        print("Training on the following modalities: ", *AUX_MOD)

        for mod in AUX_MOD:
            features = batch[mod]
            if mod == "Text":
                nan_mask = torch.isnan(features).any(dim=1)
                features = features[~nan_mask]
            ebd = self.mod_encoders[mod](features)
            embeddings = ebd/ebd.norm(dim=1, keepdim=True)
            clip_embeddings[mod] = embeddings
            if mod == "Text":
                logits[f"logits_per_mob_{mod}"] = logit_scale * clip_embeddings['mob'][~nan_mask] @ clip_embeddings[mod].t()
                logits[f"logits_per_{mod}_mob"] = logits[f"logits_per_mob_{mod}"].t()
            else:
                logits[f"logits_per_mob_{mod}"] = logit_scale * clip_embeddings['mob'] @ clip_embeddings[mod].t()
                logits[f"logits_per_{mod}_mob"] = logits[f"logits_per_mob_{mod}"].t()

        return logits, mob_ebd
    
   

    
        
       
    
        