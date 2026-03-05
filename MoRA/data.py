import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

import torch
from torch.utils.data import Dataset,DataLoader
from typing import Dict
import lightning.pytorch as pl
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp


AUX_MOD = ["ACS", "ASR", "BSUM", "ESRI", "URBANICITY", "DHC", "RETAILDEMAND", "Text"]

class FeatureDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        mob_path, 
        feature_paths, # A list of all the features, this can be any N number of feats
        mob_graph_path,
        batch_size: int=64,
        num_workers=4, 
        val_random_split_fraction = 0.1,
        ):

        super().__init__()
        self.mob_path = mob_path
        self.feature_paths = feature_paths
        self.mob_graph_path = mob_graph_path

        
        self.batch_size= batch_size
        self.num_workers = num_workers
        self.val_random_split_fraction = val_random_split_fraction
        
        self.save_hyperparameters()

    def setup(self, stage= None):
        
        self.dataset = CustomDataset(self.mob_path, self.feature_paths, self.mob_graph_path)
        N_val = int(len(self.dataset) * self.val_random_split_fraction)
        N_train = len(self.dataset) - N_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [N_train, N_val])

        self.mob_adj = self.dataset.mob_adj  
        self.mob_features = self.dataset.mob_features

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    

class CustomDataset(Dataset):
    def __init__(self, mob_path, feature_paths, mob_graph_path):
        # Mobility
        mob_data = np.load(mob_path)
        # if isinstance(mob_data, np.lib.npyio.NpzFile):
        #     if len(mob_data.files) == 1:
        #         self.mob_features = mob_data[mob_data.files[0]]
        #     else:
        #         # the mob_data should have a key called embeddings and their node_ids
        #         embs = mob_data['embeddings']
        #         node_ids = mob_data['node_ids']
        #         max_id = node_ids.max()
        #         dim = embs.shape[1]

        #         self.mob_features = torch.full((max_id + 1, dim), 1e-8, dtype=torch.float32)

        #         self.mob_features[node_ids] = torch.from_numpy(embs)

        #         print(f"Loaded {max_id + 1} nodes with dimension {dim}")
                

        #     mob_data.close()
        # else:
        self.mob_features = torch.from_numpy(mob_data['embeddings']).to(torch.float32)

        # Aux features: load all keys into memory
        aux_archive = np.load(feature_paths)
        self.aux_features = {k: aux_archive[k] for k in aux_archive.files}
        aux_archive.close()

        # Graph
        self.mob_adj = self._load_mob_adj(mob_graph_path)
       
    def _load_npy(self, path):
        return np.load(path)    
        
    def _normalize_adj(self, mat):
        """Laplacian normalization for mat in coo_matrix

        Args:
            mat (scipy.sparse.coo_matrix): the un-normalized adjacent matrix

        Returns:
            scipy.sparse.coo_matrix: normalized adjacent matrix
        """
        degree = np.array(mat.sum(axis=-1)) + 1e-10
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)

        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()
    
    def _load_mob_adj(self, mob_graph_path):
        """
        Load a mobility graph adjacency matrix from a file, normalize it, 
        and convert it to a PyTorch sparse tensor.

        Args:
            mob_graph_path (str): 
                The file path to the mobility graph data stored in `.npz` format. 

        Returns:
            torch.sparse.Tensor: 
                A normalized adjacency matrix in PyTorch sparse tensor format. 
        """
        
        loaded = np.load(mob_graph_path, allow_pickle=True)
        # print("LOADED keys", loaded.keys())
        # M = len(loaded["valid_nodes"])
        from_array = np.array([int(x, 16)-1 for x in loaded["from_"]], dtype=np.int32)
        to_array   = np.array([int(x, 16)-1 for x in loaded["to"]], dtype=np.int32)
        M = max(from_array.max(), to_array.max())+1
        mob_adj_coo_mat = coo_matrix((loaded["weight"], (from_array, to_array)), shape=(M, M))  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
        print(f"Total edges : {mob_adj_coo_mat.nnz}")  # Number of stored values, including explicit zeros.
        normalized_adj_mat = self._normalize_adj(mob_adj_coo_mat)
        
        idxs = torch.from_numpy(np.vstack([normalized_adj_mat.row, normalized_adj_mat.col]).astype(np.int64))
        vals = torch.from_numpy(normalized_adj_mat.data.astype(np.float32))
        shape = torch.Size(normalized_adj_mat.shape)

        return torch.sparse_coo_tensor(idxs, vals, size=shape)
   
    def __len__(self):
        return len(self.mob_features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mob = self.mob_features[idx] 
        result = {
            "mob": torch.tensor(mob, dtype=torch.float),
            "index": idx
        } 
        for mod in AUX_MOD:
            result[mod] = torch.tensor(self.aux_features[mod][idx], dtype = torch.float)

        return result
            

    def get_mob_graph(self):
        return self.mob_adj


