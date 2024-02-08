import os
from pathlib import Path
import argparse 
import logging

import wandb
import torch
import torchvision
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from cem.data.CUB200 import cub_loader
from protocbm.model import *
from protocbm._config import *


class ConceptModel(L.LightningModule):
    def __init__(self, 
                 x2c_model: torch.nn.Module,
                 optimiser: str                         = "adam",
                 lr: float                              = 1e-5,
                 weight_decay: float                    = 0.0,
                 plateau_lr_scheduler_enable: bool      = False,
                 plateau_lr_scheduler_monitor: str      = "val_x2c_acc",
                 plateau_lr_scheduler_mode: str         = "min",
                 plateau_lr_scheduler_patience: int     = 10,
                 plateau_lr_scheduler_factor: float     = 0.1,
                 plateau_lr_scheduler_min_lr: float     = 1e-6,
                 plateau_lr_scheduler_threshold: float  = 0.01,
                 plateau_lr_scheduler_cooldown: int     = 0):
        super().__init__()
        self.x2c_model = x2c_model
        self.optimiser = optimiser
        self.lr = lr
        self.weight_decay = weight_decay
        self.plateau_lr_scheduler_enable = plateau_lr_scheduler_enable
        self.plateau_lr_scheduler_monitor = plateau_lr_scheduler_monitor
        self.plateau_lr_scheduler_mode = plateau_lr_scheduler_mode
        self.plateau_lr_scheduler_patience = plateau_lr_scheduler_patience
        self.plateau_lr_scheduler_factor = plateau_lr_scheduler_factor
        self.plateau_lr_scheduler_min_lr = plateau_lr_scheduler_min_lr
        self.plateau_lr_scheduler_threshold = plateau_lr_scheduler_threshold
        self.plateau_lr_scheduler_cooldown = plateau_lr_scheduler_cooldown
        self.accuracy = Accuracy("binary")
        self.save_hyperparameters()
        
    def training_step(self, batch, batch_idx):
        x, _, c = batch
        x2c_output = self.x2c_model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x2c_output, c)
        acc = self.accuracy(x2c_output, c)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _, c = batch
        x2c_output = self.x2c_model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x2c_output, c)
        acc = self.accuracy(x2c_output, c)
        self.log("val_acc", acc, on_step=True, on_epoch=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, _, c = batch
        x2c_output = self.x2c_model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x2c_output, c)
        acc = self.accuracy(x2c_output, c)
        self.log("test_acc", acc, on_step=True, on_epoch=True)
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        objects = {}
        objects['optimizer'] = get_optimiser(self.optimiser)(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.plateau_lr_scheduler_enable:
            objects['lr_scheduler'] = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(objects['optimizer'],
                                                                        mode=self.plateau_lr_scheduler_mode,
                                                                        patience=self.plateau_lr_scheduler_patience,
                                                                        factor=self.plateau_lr_scheduler_factor,
                                                                        min_lr=self.plateau_lr_scheduler_min_lr,
                                                                        threshold=self.plateau_lr_scheduler_threshold,
                                                                        cooldown=self.plateau_lr_scheduler_cooldown,),
                "monitor": self.plateau_lr_scheduler_monitor,
                "strict": False,
            }
        return objects


def create_wandb_logger(args):
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        group=args.group,
        job_type=args.job_type,
        tags=args.tags,
        notes=args.notes,
        # reinit=True,
        log_model=args.wandb_log_model,
        settings=wandb.Settings(start_method="thread"))
    wandb_logger.experiment.config.update(args)  # add configuration file
    wandb.run.name = args.run_name + args.suffix_wand_run_name

    return wandb_logger

def construct_backbone(arch, n_classes, pretrained=True):
    arch = arch.lower().strip()
    if arch.startswith("resnet"):
        if arch == "resnet18":
            backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif arch == "resnet34":
            backbone = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None)
        elif arch == "resnet50":
            backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)
        elif arch == "resnet101":
            backbone = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT if pretrained else None)
        elif arch == "resnet152":
            backbone = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT if pretrained else None)
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, n_classes)        
    elif arch.startswith("efficientnet-v2"):
        if arch == "efficientnet-v2-s":
            backbone = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        elif arch == "efficientnet-v2-m":
            backbone = torchvision.models.efficientnet_v2_m(weights=torchvision.models.EfficientNet_V2_M_Weights.DEFAULT if pretrained else None)
        elif arch == "efficientnet-v2-l":
            backbone = torchvision.models.efficientnet_v2_l(weights=torchvision.models.EfficientNet_V2_L_Weights.DEFAULT if pretrained else None)
        
        backbone.classifier[1] = torch.nn.Linear(backbone.classifier[1].in_features, n_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return backbone


def main(
    n_concepts: int,
    # dataloader settings
    cub_dir,
    pkl_dir,
    batch_size: int = 64,
    num_workers: int = 4,
    use_cbm_concept_subset: bool = False,
    # model settings
    optimiser: str = "adam",
    x2c_arch: str = "resnet50",
    # DKNN settings
    # optimiser settings
    lr: float = 0.01,
    weight_decay = 0.0005,
    # trainer settings
    precision: int = 32,
    max_epochs: int = 20,
    profiler: str = None,
    # tensorboard settings
    tb_log_dir: str = None,
    tb_name: str = "porotcbm",
    # wandb logging
    wandb_logger: WandbLogger = None,
    plateau_lr_scheduler_enable: bool = False,
    plateau_lr_scheduler_monitor: str = "val_x2c_acc",
    plateau_lr_scheduler_mode: str = "min",
    plateau_lr_scheduler_patience: int = 10,
    plateau_lr_scheduler_factor: float = 0.1,
    plateau_lr_scheduler_min_lr: float = 1e-6,
    plateau_lr_scheduler_threshold: float = 0.01,
    plateau_lr_scheduler_cooldown: int = 0,
    early_stop_enable: bool = True,
    early_stop_monitor: str = "val_x2c_acc",
    early_stop_mode: str = "max",
    early_stop_patience: int = 3,
    log_level: str = "INFO",
    checkpoint_dir: str = None,
    checkpoint_name: str = None,
    checkpoint_monitor: str = None,
    checkpoint_n_epochs: int = None,
    **kwargs,
):  
    # dataset preparation
    cub_dir = Path(cub_dir)
    pkl_dir = Path(pkl_dir)
    
    def path_transform(path):
        path_parts = Path(path)
        idx = path_parts.parts.index('CUB200')
        rel_parts = path_parts.parts[idx+1:]
        return cub_dir / Path(*rel_parts)
    
    train_pkl = str(pkl_dir / "train.pkl")
    test_pkl = str(pkl_dir / "test.pkl")
    val_pkl = str(pkl_dir / "val.pkl")
    
    train_dl = cub_loader.load_data([train_pkl],
                                    use_attr=True,
                                    no_img=False,
                                    root_dir=cub_dir,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    path_transform=path_transform,
                                    is_training=True,
                                    use_cbm_concept_subset=use_cbm_concept_subset)
    
    test_dl = cub_loader.load_data([test_pkl],
                                    use_attr=True,
                                    no_img=False,
                                    root_dir=cub_dir,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    path_transform=path_transform,
                                    is_training=False,
                                    use_cbm_concept_subset=use_cbm_concept_subset)
    
    val_dl = cub_loader.load_data([val_pkl],
                                    use_attr=True,
                                    no_img=False,
                                    root_dir=cub_dir,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    path_transform=path_transform,
                                    is_training=False,
                                    use_cbm_concept_subset=use_cbm_concept_subset)
    
    print("=" * 20)
    print("Train set size: ", len(train_dl.dataset))
    print("Test set size: ", len(test_dl.dataset))
    print("Val set size: ", len(val_dl.dataset))
    print("=" * 20)
    
    # set logging level
    logging.getLogger("lightning.pytorch").setLevel(log_level)
    
    # model preparation
    x2c_model = construct_backbone(x2c_arch, pretrained=True, n_classes=n_concepts)
    model = ConceptModel(x2c_model,
                         optimiser=optimiser,
                         lr=lr,
                         weight_decay=weight_decay,
                         plateau_lr_scheduler_enable=plateau_lr_scheduler_enable,
                         plateau_lr_scheduler_monitor=plateau_lr_scheduler_monitor,
                         plateau_lr_scheduler_mode=plateau_lr_scheduler_mode,
                         plateau_lr_scheduler_patience=plateau_lr_scheduler_patience,
                         plateau_lr_scheduler_factor=plateau_lr_scheduler_factor,
                         plateau_lr_scheduler_min_lr=plateau_lr_scheduler_min_lr,
                         plateau_lr_scheduler_threshold=plateau_lr_scheduler_threshold,
                         plateau_lr_scheduler_cooldown=plateau_lr_scheduler_cooldown)
 
    callbacks = []
    if early_stop_enable:
        early_stop = EarlyStopping(monitor=early_stop_monitor,
                                   mode=early_stop_mode,
                                   patience=early_stop_patience,
                                   verbose=True,
                                   strict=False)
        callbacks.append(early_stop)
        
    if checkpoint_dir is not None:
        checkpoint_cb = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir,
                                                     filename=checkpoint_name,
                                                     monitor=checkpoint_monitor,
                                                     every_n_epochs=checkpoint_n_epochs,)
        callbacks.append(checkpoint_cb)
    
    loggers = [TensorBoardLogger(tb_log_dir, tb_name)]
    if wandb_logger is not None:
        loggers.append(wandb_logger)
    
    trainer = L.Trainer(max_epochs=max_epochs,
                        logger=loggers, 
                        precision=precision, 
                        num_sanity_val_steps=0, 
                        callbacks=callbacks,
                        profiler=profiler)
    trainer.fit(model, train_dl, val_dl)
    trainer.test(model, test_dl)
    
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # dataset settings
    parser.add_argument("--cub_dir", type=str, required=True)
    parser.add_argument("--pkl_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    # model settings
    parser.add_argument("--n_concepts", type=int, required=True)
    parser.add_argument("--x2c_arch", type=str, default="resnet50")
    
    # optimiser settings
    parser.add_argument("--optimiser", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    
    # plateau lr scheduler settings
    parser.add_argument("--plateau_lr_scheduler_enable", action="store_true")
    parser.add_argument("--plateau_lr_scheduler_monitor", type=str, default="val_c2y_acc")
    parser.add_argument("--plateau_lr_scheduler_mode", type=str, default="max")
    parser.add_argument("--plateau_lr_scheduler_patience", type=int, default=10)
    parser.add_argument("--plateau_lr_scheduler_factor", type=float, default=0.1)
    parser.add_argument("--plateau_lr_scheduler_min_lr", type=float, default=1e-6)
    parser.add_argument("--plateau_lr_scheduler_threshold", type=float, default=0.01)
    parser.add_argument("--plateau_lr_scheduler_cooldown", type=int, default=0)
    
    # early stop settings
    parser.add_argument("--early_stop_enable", action="store_true")
    parser.add_argument("--early_stop_monitor", type=str, default="val_c2y_acc")
    parser.add_argument("--early_stop_mode", type=str, default="max")
    parser.add_argument("--early_stop_patience", type=int, default=3)
    
    # train settings
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--profiler", type=str, default=None)
    
    # tensorboard settings
    parser.add_argument("--tb_log_dir", type=str, default=None)
    parser.add_argument("--tb_name", type=str, default="protocbm")
    
    # wandb settings
    parser.add_argument("--run_name", type=str, default="protocbm-cub")
    parser.add_argument('--group', type=str, help="Group runs in wand")
    parser.add_argument('--job_type', type=str, help="Job type for wand")
    parser.add_argument('--notes', type=str, help="Notes for wandb logging.")
    parser.add_argument('--tags', nargs='+', type=str, default=[], help='Tags for wandb')
    parser.add_argument('--suffix_wand_run_name',
                        type=str,
                        default="",
                        help="Suffix for run name in wand")
    parser.add_argument('--wandb_log_model',
                        action='store_true',
                        dest='wandb_log_model',
                        help='True for storing the model checkpoints in wandb')
    parser.set_defaults(wandb_log_model=False)
    parser.add_argument('--disable_wandb',
                        action='store_true',
                        dest='disable_wandb',
                        help='True if you dont want to crete wandb logs.')
    parser.set_defaults(disable_wandb=False)
    
    # checkpoint settings
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_monitor", type=str, default=None)
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--checkpoint_n_epochs", type=int, default=None)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    L.seed_everything(773)
    
    args = parse_arguments()
    if not args.disable_wandb:
        wandb_logger = create_wandb_logger(args)
    else:
        wandb_logger = None
    
    main(**vars(args), wandb_logger=wandb_logger)
