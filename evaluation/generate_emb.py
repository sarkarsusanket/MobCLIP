import torch
import numpy as np
import os
import sys
sys.path.append(r"/home/susanket/MobCLIP/src/MoRA/MoRA")
from main import MobCLIPLightningModule
from data import FeatureDataModule

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = "/home/susanket/mobclip_logs/US/Hex6/checkpoints/epoch=174-val_loss=4.76.ckpt"
config_fn = "/home/susanket/MobCLIP/src/MoRA/MoRA/configs/default_USA_Hex6.yaml"  # adjust path if needed

# Recreate datamodule (so we can get the adjacency)
dm = FeatureDataModule(
    mob_path="/data/susanket/mobclip/mobility_line_embeddings_hex6.npz",
    feature_paths="/data/susanket/mobclip/preprocessed_gdbs/hex6_features.npz",
    mob_graph_path="/data/susanket/mobclip/mobility_adjacency_mat.npz",
    batch_size=32,
    num_workers=8,
)
dm.setup("fit")
mob_adj = dm.dataset.get_mob_graph().to(device)

# Load trained model
lit_model = MobCLIPLightningModule.load_from_checkpoint(
    ckpt_path,
    mob_features_path="/data/susanket/mobclip/mobility_line_embeddings_hex6.npz",
    map_location=device,
)
lit_model.eval().to(device)
model = lit_model.model  # this is your MobCLIP

with torch.no_grad():
    # Global LightGCN embeddings for *all* hexbins
    global_mob_ebd = model.mob_lightgcn(mob_adj)             # shape [N_hex, embedding_dim]

    # Optionally, use the CLIP-normalized version (what you use in logits)
    mob_clip_emb = global_mob_ebd / global_mob_ebd.norm(dim=1, keepdim=True)

# Move to CPU / NumPy if needed
hex_embeddings = mob_clip_emb.cpu().numpy()   # hex_embeddings[i] is embedding for hexbin i
node_ids = np.load("/data/susanket/mobclip/mobility_line_embeddings_hex6.npz")['node_ids']

np.savez_compressed("/data/susanket/mobclip/hex6_mobclip_embeddings_US.npz", embs = hex_embeddings, ids = node_ids)
