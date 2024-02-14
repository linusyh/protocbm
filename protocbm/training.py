from argparse import ArgumentParser
from pathlib import Path
import logging

import wandb
import torch
from torch.utils.data import DataLoader
import torchvision
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from cem.data.CUB200 import cub_loader
from protocbm.model import *
from protocbm._config import *


def create_wandb_logger(args: dict):
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        group=args.get("group", None),
        job_type=args.get("job_type", None),
        tags=args.get("tags", []),
        notes=args.get("notes", None),
        # reinit=True,
        log_model=args.get("notes", False),
        settings=wandb.Settings(start_method="thread"))
    wandb_logger.experiment.config.update(args)  # add configuration file
    wandb.run.name = args['run_name'] + args['suffix_wand_run_name']

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

def protocbm_train_loop(
    n_concepts: int,
    n_classes: int,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    # model settings
    concept_loss_weight: float = 1.0,
    proto_loss_weight: float = 1.0,
    x2c_arch: str = "resnet50",
    c_activation="sigmoid",
    # DKNN settings
    dknn_k: int = 1,
    dknn_tau: float = 1.0,
    dknn_method: str = "determinstic",
    dknn_num_samples: int = -1,
    dknn_similarity='euclidean',
    dknn_loss_type='bce',
    x2c_only_epochs: int = 0,
    epochs_proto_recompute: int = 1,
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
    plateau_lr_scheduler_monitor: str = "val_c2y_acc",
    plateau_lr_scheduler_mode: str = "min",
    plateau_lr_scheduler_patience: int = 10,
    plateau_lr_scheduler_factor: float = 0.1,
    plateau_lr_scheduler_min_lr: float = 1e-6,
    plateau_lr_scheduler_threshold: float = 0.01,
    plateau_lr_scheduler_cooldown: int = 0,
    early_stop_enable: bool = True,
    early_stop_monitor: str = "val_c2y_acc",
    early_stop_mode: str = "max",
    early_stop_patience: int = 3,
    log_level: str = "INFO",
    checkpoint_dir: str = None,
    checkpoint_name: str = None,
    checkpoint_n_epochs: int = None,
    disable_wandb: bool = False,
    use_wandb_version_for_ckpt: bool = True,
    batch_process_fn = None,
    **kwargs,
):  
    # Welcome Screen
    print("=" * 20)
    print("Train set size: ", len(train_dl.dataset))
    print("Test set size: ", len(test_dl.dataset))
    print("Val set size: ", len(val_dl.dataset))
    print("=" * 20)
    
    loggers = [TensorBoardLogger(tb_log_dir, tb_name)]
    if wandb_logger is not None:
        loggers.append(wandb_logger)
    
    # setup wandb logging
    if not disable_wandb:
        wandb_logger = create_wandb_logger(kwargs)
        loggers.append(wandb_logger)
    
    # set logging level
    logging.getLogger("lightning.pytorch").setLevel(log_level)
    
    # model preparation
    x2c_model = construct_backbone(x2c_arch, pretrained=True, n_classes=n_concepts)
    protocbm_model = ProtoCBMDKNNJoint(
        n_concepts= n_concepts,
        n_classes=n_classes,
        x2c_model=x2c_model,
        concept_loss_weight=concept_loss_weight,
        proto_loss_weight=proto_loss_weight,
        dknn_k=dknn_k,
        dknn_tau=dknn_tau,
        dknn_method=dknn_method,
        dknn_num_samples=dknn_num_samples,
        dknn_similarity=dknn_similarity,
        dknn_loss_type=dknn_loss_type,
        c_activation=c_activation,
        proto_train_dl=train_dl,
        learning_rate=lr,
        weight_decay=weight_decay,
        epoch_proto_recompute=epochs_proto_recompute,
        x2c_only_epochs=x2c_only_epochs,
        plateau_lr_scheduler_enable=plateau_lr_scheduler_enable,
        plateau_lr_scheduler_monitor=plateau_lr_scheduler_monitor,
        plateau_lr_scheduler_mode=plateau_lr_scheduler_mode,
        plateau_lr_scheduler_patience=plateau_lr_scheduler_patience,
        plateau_lr_scheduler_factor=plateau_lr_scheduler_factor,
        plateau_lr_scheduler_min_lr=plateau_lr_scheduler_min_lr,
        plateau_lr_scheduler_threshold=plateau_lr_scheduler_threshold,
        plateau_lr_scheduler_cooldown=plateau_lr_scheduler_cooldown,
        batch_process_fn=batch_process_fn
    )
 
    # Careate callbacks
    callbacks = []
    if early_stop_enable:
        early_stop = EarlyStopping(monitor=early_stop_monitor,
                                   mode=early_stop_mode,
                                   patience=early_stop_patience,
                                   verbose=True,
                                   strict=False)
        callbacks.append(early_stop)
    
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        if use_wandb_version_for_ckpt and not disable_wandb:
            checkpoint_dir = checkpoint_dir / str(wandb_logger.experiment.id)
        
        checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=checkpoint_name,
            every_n_epochs=checkpoint_n_epochs,
        )
        callbacks.append(checkpoint)
    
    
    trainer = L.Trainer(max_epochs=max_epochs,
                        logger=loggers, 
                        precision=precision, 
                        num_sanity_val_steps=0, 
                        callbacks=callbacks,
                        profiler=profiler)
    trainer.fit(protocbm_model, train_dl, val_dl)
    trainer.test(protocbm_model, test_dl)
    
    
def protocbm_add_common_args(parser: ArgumentParser,
                             default_monitor="val_c2y_acc_cls",
                             default_run_name="protocbm-cub"):
    # model settings
    parser.add_argument("--n_concepts", type=int, required=True)
    parser.add_argument("--n_classes", type=int, required=True)
    parser.add_argument("--x2c_arch", type=str, default="resnet50")
    parser.add_argument("--c_activation", type=str, default="sigmoid")
    
    parser.add_argument("--concept_loss_weight", type=float, default=1.0)
    parser.add_argument("--proto_loss_weight", type=float, default=1.0)
    
    # DKNN settings
    parser.add_argument("--dknn_k", type=int, default=1)
    parser.add_argument("--dknn_tau", type=float, default=1.0)
    parser.add_argument("--dknn_method", type=str, default="deterministic", choices=["deterministic", "stochastic"])
    parser.add_argument("--dknn_num_samples", type=int, default=-1)
    parser.add_argument("--dknn_similarity", type=str, default="euclidean")
    parser.add_argument("--dknn_loss_type", type=str, default="bce")
    parser.add_argument("--epochs_proto_recompute", type=int, default=1)
    parser.add_argument("--x2c_only_epochs", type=int, default=0)
    
    # optimiser settings
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    
    # plateau lr scheduler settings
    parser.add_argument("--plateau_lr_scheduler_enable", action="store_true")
    parser.add_argument("--plateau_lr_scheduler_monitor", type=str, default=default_monitor)
    parser.add_argument("--plateau_lr_scheduler_mode", type=str, default="max")
    parser.add_argument("--plateau_lr_scheduler_patience", type=int, default=10)
    parser.add_argument("--plateau_lr_scheduler_factor", type=float, default=0.1)
    parser.add_argument("--plateau_lr_scheduler_min_lr", type=float, default=1e-6)
    parser.add_argument("--plateau_lr_scheduler_threshold", type=float, default=0.01)
    parser.add_argument("--plateau_lr_scheduler_cooldown", type=int, default=0)
    
    # early stop settings
    parser.add_argument("--early_stop_enable", action="store_true")
    parser.add_argument("--early_stop_monitor", type=str, default=default_monitor)
    parser.add_argument("--early_stop_mode", type=str, default="max")
    parser.add_argument("--early_stop_patience", type=int, default=3)
    
    # train settings
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--profiler", type=str, default=None)
    
    # tensorboard settings
    parser.add_argument("--tb_log_dir", type=str, default=None)
    parser.add_argument("--tb_name", type=str, default="protocbm")
    
    #checkpoint settings
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--checkpoint_n_epochs", type=int, default=None)
    
    # Enabled only if wandb is enabled
    parser.add_argument('--use_wandb_version_for_ckpt', action='store_true')
    
    # wandb settings
    parser.add_argument("--run_name", type=str, default=default_run_name)
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
    
    return parser