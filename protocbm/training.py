from omegaconf import DictConfig, OmegaConf
import hydra
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
from protocbm._config import *
from protocbm.models.protocbm import ProtoCBM
from protocbm.models.protocem import ProtoCEM

def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary.

    Parameters:
    - d: The dictionary to flatten
    - parent_key: The base key to use for flattening (used internally)
    - sep: The separator between nested keys

    Returns:
    A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) or isinstance(v, DictConfig):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def create_wandb_logger(args: DictConfig):
    args = dict(args.copy())
    wandb_args = args.pop("wandb")
    
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        group=wandb_args.get("group", None),
        job_type=wandb_args.get("job_type", None),
        tags=wandb_args.get("tags", []),
        notes=wandb_args.get("notes", None),
        # reinit=True,
        log_model=wandb_args.get("notes", False),
        settings=wandb.Settings(start_method="thread"))
    
    
    logging.debug(flatten_dict(args))
    wandb_logger.experiment.config.update(flatten_dict(args))  # add configuration file
    wandb.run.name = wandb_args.run_name + wandb_args.suffix
    return wandb_logger


def get_proto_model(name: str):
    lookup = {
        "protocbm": ProtoCBM,
        "protocem": ProtoCEM
    }
    return lookup[name.lower().strip()]  


def construct_model(config: DictConfig, 
                    proto_dl: DataLoader,
                    batch_process_fn=None):
    
    model_config_dict = dict(config.model.copy())
    model_type = model_config_dict.pop("type")
    model_config_dict.pop('proto')
    
    ModelClass = get_proto_model(model_type)
    
    args = dict(
        n_concepts= config.dataset.n_concepts,
        n_tasks=config.dataset.n_classes,
        proto_train_dl=proto_dl,
        batch_process_fn=batch_process_fn,
        **model_config_dict
    )
    
    # DKNN config
    if config.model.proto.type.upper() == "DKNN":
        d = dict(config.model.proto.copy())
        d.pop('type')
        for key, val in d.items():
            args[f"dknn_{key}"] = val
    
    # optimiser config
    optimiser_dict = dict(config.optimiser.copy())
    opt_type = optimiser_dict.pop("type")
    args["optimiser"] = opt_type
    args["optimiser_params"] = optimiser_dict
    
    # lr scheduler config
    lrs_dict = dict(config.lr_scheduler.copy())
    lrs_type = lrs_dict.pop("type")
    if lrs_type =="plateau":
        monitor = lrs_dict.pop("monitor")
        args["plateau_lr_scheduler_enable"] = True
        args["plateau_lr_scheduler_monitor"] = monitor
        args["plateau_lr_scheduler_params"] = lrs_dict
    
    model = ModelClass(
        **args
    )
    return model


def train_loop(
    train_dl: DataLoader,
    test_dl: DataLoader,
    val_dl: DataLoader,
    config: DictConfig,
    batch_process_fn = None,
):  
    
    # Build model
    model = construct_model(config, train_dl, batch_process_fn)
    
    logging.debug(str(model))
    
    # Welcome Screen
    logging.info("=" * 20)
    logging.info(f"Train set size: {len(train_dl.dataset)}")
    logging.info(f"Test set size: {len(test_dl.dataset)}")
    logging.info(f" Val set size: {len(val_dl.dataset)}")
    logging.info("=" * 20)
    
    loggers = []
    
    if "tensorboard" in config.keys():
        loggers.append(TensorBoardLogger(config.tensorboard.dir, name=config.tensorboard.name))
        
    # setup wandb logging
    wandb_logger = None
    if config.wandb.enable:
        wandb_logger = create_wandb_logger(config)
        loggers.append(wandb_logger)
    
    # set logging level
    logging.getLogger("lightning.pytorch").setLevel(config.log_level)    
 
    # Careate callbacks
    callbacks = []
    if config.early_stop.enable:
        early_stop = EarlyStopping(monitor=config.early_stop.monitor,
                                   mode=config.early_stop.mode,
                                   patience=config.early_stop.patience,
                                   verbose=True,
                                   strict=False)
        callbacks.append(early_stop)
    
    if config.ckpt.enable:
        checkpoint_dir = Path(config.ckpt.dir)
        if config.ckpt.use_wandb_version and config.wandb.enable:
            checkpoint_dir = checkpoint_dir / str(wandb_logger.experiment.id)
        
        checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=config.ckpt.name,
            every_n_epochs=config.ckpt.n_epochs,
        )
        callbacks.append(checkpoint)
    
    
    trainer = L.Trainer(max_epochs=config.trainer.max_epochs,
                        logger=loggers, 
                        num_sanity_val_steps=0, 
                        callbacks=callbacks,
                        precision=config.trainer.precision, 
                        profiler=config.trainer.profiler,)
    trainer.fit(model, train_dl, val_dl)
    trainer.test(model, test_dl)
    
    
def protocbm_add_common_args(parser: ArgumentParser,
                             default_monitor="val_c2y_acc_cls",
                             default_run_name="protocbm-cub"):
    # model settings
    parser.add_argument("--n_concepts", type=int, required=True)
    parser.add_argument("--n_classes", type=int, required=True)
    parser.add_argument("--x2c_arch", type=str, default="resnet50")
    parser.add_argument("--concept_from_logit", action="store_true")
    
    parser.add_argument("--concept_loss_weight", type=float, default=1.0)
    parser.add_argument("--proto_loss_weight", type=float, default=1.0)
    
    # DKNN settings
    parser.add_argument("--dknn_k", type=int, default=1)
    parser.add_argument("--dknn_tau", type=float, default=1.0)
    parser.add_argument("--dknn_method", type=str, default="deterministic", choices=["deterministic", "stochastic"])
    parser.add_argument("--dknn_num_samples", type=int, default=-1)
    parser.add_argument("--dknn_max_neighbours", type=int, default=-1)
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