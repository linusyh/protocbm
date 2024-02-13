from typing import *

import pickle
import torch
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms

SELECTED_CONCEPTS = [20, 39]
CONCEPT_NAMES = ['Young_Male', 'Young_Female', 'Old_Male', 'Old_Female']
NUM_CONCEPTS = 40

def load_celeba_subsets(root: str,
                        config: Dict,
                        subset_indices_file: str,
                        train_transform=None,
                        val_test_transform=None,
                        download: bool = False,):
    
    with open(subset_indices_file, 'rb') as f:
        sel_indices = pickle.load(f)
    
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.CenterCrop((320,320)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    if val_test_transform is None:
        val_test_transform = transforms.Compose([
            transforms.CenterCrop((320,320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    sel_attribute_positive = torch.tensor([False] * NUM_CONCEPTS)
    for sel_concept_idx in SELECTED_CONCEPTS:
        sel_attribute_positive[sel_concept_idx] = True
    
    def label_transform(attr_tensor):
        # attr_tensor is a 40-dim tensor
        sel_concepts = attr_tensor[sel_attribute_positive]
        c = attr_tensor[~sel_attribute_positive]
        
        binariser = torch.tensor([2**i for i in range(len(SELECTED_CONCEPTS))])
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

    train_dl = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=config['train_shuffle'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
    )
    
    val_dl = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
    )
    
    test_dl = DataLoader(
        test_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
    )
    
    return train_dl, val_dl, test_dl
