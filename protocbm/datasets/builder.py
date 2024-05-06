from protocbm.datasets import celeba_cem, cub
from cem.data.synthetic_loaders import generate_xor_data, generate_trig_data, generate_dot_data
from protocbm.datasets.celeba_gender_age import load_celeba_subsets

import logging
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from omegaconf import DictConfig


def build_cub(dataset_config: DictConfig):
    logging.debug("Building CUB200 dataset")
    cub_dir = Path(dataset_config.root_dir)
    pkl_dir = Path(dataset_config.pkl_dir)
    
    def path_transform(path):
        path_parts = Path(path)
        idx = path_parts.parts.index('CUB200')
        rel_parts = path_parts.parts[idx+1:]
        return cub_dir / Path(*rel_parts)
    
    train_pkl = str(pkl_dir / "train.pkl")
    test_pkl = str(pkl_dir / "test.pkl")
    val_pkl = str(pkl_dir / "val.pkl")
    
    use_cbm_concept_subset = False
    batch_size = dataset_config.batch_size
    num_workers = dataset_config.num_workers
    
    if dataset_config.name == "CUB_200_112":
        use_cbm_concept_subset = True
        logging.debug("Using CBM concept subset")
    
    train_dl = cub.load_data([train_pkl],
                             use_attr=True,
                             no_img=False,
                             root_dir=cub_dir,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             path_transform=path_transform,
                             is_training=True,
                             use_cbm_concept_subset=use_cbm_concept_subset)
    
    test_dl = cub.load_data([test_pkl],
                            use_attr=True,
                            no_img=False,
                            root_dir=cub_dir,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            path_transform=path_transform,
                            is_training=False,
                            use_cbm_concept_subset=use_cbm_concept_subset)
    
    val_dl = cub.load_data([val_pkl],
                           no_img=False,
                           use_attr=True,
                           root_dir=cub_dir,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           path_transform=path_transform,
                           is_training=False,
                           use_cbm_concept_subset=use_cbm_concept_subset)
    logging.debug(f"Loaded CUB200: train={len(train_dl.dataset)}, val={len(val_dl.dataset)}, test={len(test_dl.dataset)}")
    return train_dl, val_dl, test_dl


def build_celeba_cem(dataset_config: DictConfig):
    cem_config = dict(
        image_size=dataset_config.get('image_size', 64),
        num_classes=dataset_config.get('n_classes', 1000),
        label_binary_width=dataset_config.get('label_binary_width', 1),
        num_concepts=dataset_config.get('n_concepts',6),
        num_hidden_concepts=dataset_config.get('num_hidden_concepts', 2),
        label_dataset_subsample=dataset_config.get('label_dataset_subsample', 12),
        use_binary_vector_class=dataset_config.get('use_binary_vector_class', True),
        use_imbalance=dataset_config.get('use_imbalance', True),
        weight_loss=dataset_config.get('weight_loss', False),
        batch_size=dataset_config.batch_size,
        num_workers=dataset_config.num_workers,
    )
    
    (train_dl, 
     val_dl, 
     test_dl, 
     imbalance) = celeba_cem.generate_data(
        config=cem_config,
        root_dir=dataset_config.root_dir,
        train_shuffle=False
     )
     
    logging.debug(f"Loaded CELEBA CEM: train={len(train_dl.dataset)}, val={len(val_dl.dataset)}, test={len(test_dl.dataset)}")
    logging.debug(f"Imbalance: {imbalance}")
    return train_dl, val_dl, test_dl


def build_celeba_genderage(dataset_config: DictConfig):
    config = {
        'batch_size': dataset_config.batch_size,
        'num_workers': dataset_config.num_workers,
        'train_shuffle': False,
        'pin_memory': torch.cuda.is_available(),
    }
    
    train_dl, val_dl, test_dl = load_celeba_subsets(
        root=dataset_config.root_dir,
        subset_indices_file=dataset_config.subset_file,
        config=config
    )
    logging.debug(f"Loaded CELEBA subsets: train={len(train_dl.dataset)}, val={len(val_dl.dataset)}, test={len(test_dl.dataset)}")
    return train_dl, val_dl, test_dl


def build_synthetic_dataset(dataset_config: DictConfig):
    name = dataset_config.name.upper().lower()
    if name == "XOR":
        generate_data = generate_xor_data
    elif name == "TRIG":
        generate_data = generate_trig_data
    elif name == "DOT":
        generate_data = generate_dot_data

    dataset_size = dataset_config.dataset_size
    batch_size = dataset_config.batch_size
    num_workers = dataset_config.num_workers
    random_seed = dataset_config.get("random_seed")
    if random_seed is not None:
        np.random.seed(random_seed)
    
    train_ds = TensorDataset(*generate_data(int(dataset_size * 0.7)))
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    
    test_ds = TensorDataset(*generate_data(int(dataset_size * 0.2)))
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
    
    val_ds = TensorDataset(*generate_data(int(dataset_size * 0.1)))
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    return train_dl, val_dl, test_dl
    
    
def build_dataset(dataset_config: DictConfig):
    ds_name = dataset_config.name.upper().strip()
    logging.info(f"Building dataset: {ds_name}")
    if ds_name.startswith("CUB_200"):
        return build_cub(dataset_config)
    elif ds_name.startswith("CELEBA_GENDERAGE"):
        return build_celeba_genderage(dataset_config)
    elif ds_name.startswith("CELEBA_CEM"):
        return build_celeba_cem(dataset_config)
    elif ds_name in ["XOR", "TRIG", "DOT"]:
        return build_synthetic_dataset(dataset_config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_config.name}")