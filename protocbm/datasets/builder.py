from cem.data.CUB200 import cub_loader

from pathlib import Path
from omegaconf import DictConfig

def build_dataset(dataset_config: DictConfig):
    if dataset_config.name.startswith("CUB_200"):
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

        return train_dl, val_dl, test_dl
    else:
        raise ValueError(f"Unknown dataset: {dataset_config.name}")