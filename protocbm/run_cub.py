from pathlib import Path
import argparse 

import torch
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from cem.data.CUB200 import cub_loader
from protocbm.model import *
from protocbm._config import *
from protocbm.training import *


def main(
    # dataloader settings
    cub_dir,
    pkl_dir,
    batch_size: int = 64,
    num_workers: int = 4,
    use_cbm_concept_subset: bool = False,
    **kwargs,    
):
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
    
    return protocbm_train_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        **kwargs
    )

def parse_arguments():
    DEFAULT_MONITOR = "val_c2y_acc_cls"
    parser = argparse.ArgumentParser()
    
    # dataset settings
    parser.add_argument("--cub_dir", type=str, required=True)
    parser.add_argument("--pkl_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_cbm_concept_subset", action="store_true")    
    
    protocbm_add_common_args(parser, DEFAULT_MONITOR, "protocbm_cub")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    L.seed_everything(773)
    
    args = parse_arguments()    
    main(**vars(args))
