import argparse
import torch

from protocbm.datasets.celeba_gender_age import load_celeba_subsets
from protocbm.model import *
from protocbm._config import *
from protocbm.training import *

def main(
    # dataloader settings
    celeba_dir,
    subset_indices_file,
    batch_size: int = 64,
    num_workers: int = 4,
    **kwargs,    
):    
    config = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'train_shuffle': False,
        'pin_memory': torch.cuda.is_available(),
    }
    
    train_dl, val_dl, test_dl = load_celeba_subsets(
        root=celeba_dir,
        subset_indices_file=subset_indices_file,
        config=config
    )
    
    def celeba_batch_fn(batch):
        x, (y, c) = batch
        return x, y, c
    
    return protocbm_train_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        batch_process_fn=celeba_batch_fn,
        **kwargs
    )
    
def parse_arguments():
    DEFAULT_MONITOR = "val_c2y_acc_cls"
    parser = argparse.ArgumentParser()
    
    # dataset settings
    parser.add_argument("--celeba_dir", type=str, required=True)
    parser.add_argument("--subset_indices_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    
    protocbm_add_common_args(parser, DEFAULT_MONITOR, "protocbm_celeba_subset")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    L.seed_everything(773)
    
    args = parse_arguments()    
    main(**vars(args))