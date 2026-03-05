import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

import torch
torch.backends.cudnn.enabled=False
import numpy as np
from pathlib import Path
import lightning.pytorch
from lightning.pytorch.cli import LightningCLI

from model import MobCLIP
from loss import Loss
from data import FeatureDataModule


# def load_mobility(mob_features_path):
#     mob_data = np.load(mob_features_path)
#     embs = mob_data['embeddings']
#     node_ids = mob_data['node_ids']
#     max_id = len(node_ids)
#     dim = embs.shape[1]
#     mob_features = torch.full((max_id + 1, dim), 1e-8, dtype=torch.float32)
#     mob_features[node_ids] = torch.from_numpy(embs).to(torch.float32)
#     return mob_features

class MobCLIPLightningModule(lightning.pytorch.LightningModule):
    def __init__(
        self,
        embedding_dim = 128,
        mob_features_path = None,
        gnn_layers=2,
        scale=0.2,
        learning_rate=1e-4,
        weight_decay=0.01,
    ) -> None:
        super().__init__()

        data = np.load(mob_features_path)
        self.mob_features = torch.from_numpy(data["embeddings"]).to(torch.float32)
        # self.mob_features_id = data["node_ids"]
        self.model = MobCLIP(
            embedding_dim=embedding_dim,
            mob_features = self.mob_features,
            gnn_layers=gnn_layers,
            scale=scale
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fun = Loss()
        self.save_hyperparameters()

    def common_step(self, batch):
        
        global_indices = batch["index"]  
        device = global_indices.device  
        mob_adj  = self.trainer.datamodule.dataset.get_mob_graph().to(device)
        logits, _ = self.model(batch, mob_adj, global_indices=global_indices) 

        return self.loss_fun(logits)

    def training_step(self, batch, batch_idx):

        loss = self.common_step(batch)
        self.log("train_loss", loss)
        self.print(f"train_loss: {loss.item()}")
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("val_loss", loss)
        self.print(f"val_loss: {loss.item()}")
        
        return loss

    def configure_optimizers(self):
        
        exclude = (
            lambda n, p: p.ndim < 2
            or "bias" in n
        )
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.model.named_parameters())
        bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": bias_params, "weight_decay": 0.0},
                {
                    "params": rest_params,
                    "weight_decay": self.weight_decay,
                },  # specify in configs/default.yaml
            ],
            lr=self.learning_rate,  # specify in configs/default.yaml
        )

        return optimizer



class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--watchmodel", action="store_true")
        


def cli_main(default_config_filename= "./configs/default.yaml"):
    save_config_fn = default_config_filename.replace(".yaml", "-latest.yaml")
        # modify configs/default.yaml for learning rate etc.

    cli = MyLightningCLI(
        model_class=MobCLIPLightningModule,
        datamodule_class=FeatureDataModule,

        save_config_kwargs=dict(
            config_filename=save_config_fn,
            overwrite=True,
        ),
        trainer_defaults={
            "accumulate_grad_batches": 1,
            "log_every_n_steps": 1,
        },
        parser_kwargs={"default_config_files": [default_config_filename]},
        seed_everything_default=0,
        run=False,
    )
    


    if cli.trainer.logger is not None:
        cli.trainer.logger.log_hyperparams(cli.datamodule.hparams)

    # Create folder to log configs
    # NOTE: Lightning does not handle config paths with subfolders
    dirname_cfg = Path(default_config_filename).parent   #Path("./configs")
    dir_log_cfg = Path(cli.trainer.log_dir) / dirname_cfg
    dir_log_cfg.mkdir(parents=True, exist_ok=True)

    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.datamodule,
        # ckpt_path = r"/home/susanket/mobclip_logs/US/Hex6-random-init/checkpoints/epoch=94-val_loss=5.01.ckpt"
        
    )
    

if __name__ == "__main__":
    config_fn =  "/home/susanket/MobCLIP/src/MoRA/MoRA/configs/default_USA_Hex6.yaml"
    torch.multiprocessing.set_sharing_strategy('file_system')

    if torch.cuda.get_device_name(device=0)=='NVIDIA A100 80GB PCIe':
        torch.set_float32_matmul_precision("highest")
        print('Model go vroom! 🚀')
    else:
        torch.set_float32_matmul_precision("high")

    cli_main(config_fn)
