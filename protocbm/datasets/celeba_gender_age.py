from typing import *

import pickle
import torch
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms
import lightning as L


SELECTED_CONCEPTS = [20, 39]
CONCEPT_NAMES = ['Young_Male', 'Young_Female', 'Old_Male', 'Old_Female']
NUM_CONCEPTS = 40

def load_celeba_subsets(root: str,
                        config: Dict,
                        subset_indices_file: str,
                        selected_concepts=None,
                        train_transform=None,
                        val_test_transform=None,
                        download: bool = False,
                        x_mean=[0.485, 0.456, 0.406],
                        x_std=[0.229, 0.224, 0.225]):
    
    with open(subset_indices_file, 'rb') as f:
        sel_indices = pickle.load(f)
        
    image_size = config.get('image_size', 64)
    
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop((image_size,image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=x_mean, std=x_std)
        ])
    
    if val_test_transform is None:
        val_test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=x_mean, std=x_std)
        ])
    
    sel_attribute_positive = torch.tensor([False] * NUM_CONCEPTS)
    
    selected_concepts = selected_concepts or SELECTED_CONCEPTS
    print(f"selected_concepts: {selected_concepts}")

    for sel_concept_idx in selected_concepts:
        sel_attribute_positive[sel_concept_idx] = True
    
    def label_transform(attr_tensor):
        # attr_tensor is a 40-dim tensor
        sel_concepts = attr_tensor[sel_attribute_positive]
        c = attr_tensor[~sel_attribute_positive]
        
        binariser = torch.tensor([2**i for i in range(len(selected_concepts))])
        y = torch.dot(sel_concepts, binariser)
        return y, c
    
    train_ds = CelebA(root, 
                      split='train',
                      target_type='attr',
                      transform=train_transform,
                      target_transform=label_transform,
                      download=download)
    
    val_ds = CelebA(root, 
                      split='valid',
                      target_type='attr',
                      transform=val_test_transform,
                      target_transform=label_transform,
                      download=download)
    
    test_ds = CelebA(root, 
                      split='test',
                      target_type='attr',
                      transform=val_test_transform,
                      target_transform=label_transform,
                      download=download)
    
    train_ds = Subset(train_ds, sel_indices[0])
    val_ds = Subset(val_ds, sel_indices[1])
    test_ds = Subset(test_ds, sel_indices[2])

    batch_size = config.get('batch_size', 64)
    train_shuffle = config.get('train_shuffle', True)
    num_workers = config.get('num_workers', 0)
    pin_memory = config.get('pin_memory', torch.cuda.is_available())
    
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_dl, val_dl, test_dl
