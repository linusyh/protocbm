from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import lightning as L
import lightning.pytorch as pl

from protocbm.model import *
from protocbm._config import *
from protocbm.training import *
from protocbm.datasets.builder import build_dataset

def initialise():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    # L.seed_everything(773)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig):
    initialise()

    # Load the dataset
    train_dl, val_dl, test_dl = build_dataset(cfg.dataset)
    
    train_loop(train_dl=train_dl,
               test_dl=test_dl,
               val_dl=val_dl,
               config=cfg)


if __name__ == "__main__":
    main()
