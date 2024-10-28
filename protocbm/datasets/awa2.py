from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
import json


class AWA2(Dataset):
    def __init__(self, 
                 root,
                 split='train',
                 image_size=320,
                 split_file_name='split.json',
                 x_mean=[0.485, 0.456, 0.406],
                 x_std=[0.229, 0.224, 0.225]):
        
        root = Path(root)
        assert root.exists(), f"Path {root} does not exist."
        assert root.is_dir(), f"Path {root} is not a directory."
        
        split_file = root / split_file_name
        assert split_file.exists(), f"Split file {split_file} does not exist."
        self.split_file = split_file

        with open(split_file, 'r') as f:
            split_dict = json.load(f)
            self.image_files = split_dict[split]
        
        # Load class list
        class_file = root / f"classes.txt"
        with open(class_file, 'r') as f:
            self.classes = [l.strip().split("\t")[-1] for l in f.readlines()]
            self.class_to_index = {c: i for i, c in enumerate(self.classes)}
        
        # Load attributes
        attr_file = root / f"predicate-matrix-binary.txt"
        self.attributes = []
        with open(attr_file, 'r') as f:
            for l in f.readlines():
                attr = np.array([float(x) for x in l.split(" ")], dtype=np.float32)
                self.attributes.append(attr)
        
        self.root = root
        self.split = split
        self.image_size = image_size
        self.x_mean = x_mean
        self.x_std = x_std
        if split.startswith("train"):
            self.transform = transforms.Compose([
                transforms.CenterCrop((image_size,image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=x_mean, std=x_std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop((image_size,image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=x_mean, std=x_std)
            ])
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        relative_path, class_name = self.image_files[idx]
        image = Image.open(self.root / relative_path).convert("RGB")
        image = self.transform(image)
        label = self.class_to_index[class_name]
        attributes = self.attributes[label]
        
        return image, label, attributes

# CEM compatiability
def generate_data(config, root_dir, seed, output_dataset_vars=True):
    random_sampling_p = config.get("sampling_percent", 1.0)
    
    train_ds = AWA2(root_dir, split='train') 
    test_ds = AWA2(root_dir, split='test')
    val_ds = AWA2(root_dir, split='val')
 
    if  0 < random_sampling_p < 1:
        train_ds = Subset(train_ds, np.random.permutation(len(train_ds))[:int(np.ceil(random_sampling_p * len(train_ds)))])
        test_ds = Subset(test_ds, np.random.permutation(len(test_ds))[:int(np.ceil(random_sampling_p * len(test_ds)))])
        val_ds = Subset(val_ds, np.random.permutation(len(val_ds))[:int(np.ceil(random_sampling_p * len(val_ds)))])

    train_dl = DataLoader(train_ds, 
                          batch_size=config['batch_size'],
                          num_workers=config['num_workers'],
                          shuffle=True)
    test_dl = DataLoader(test_ds,
                         batch_size=config['batch_size'],
                         num_workers=config['num_workers'],
                         shuffle=True)
    val_dl = DataLoader(val_ds,
                        batch_size=config['batch_size'],
                        num_workers=config['num_workers'],
                        shuffle=True)                     
    
    n_concepts = 85
    n_classes = 50 
    concept_group_map = None
    imbalance = None
    
    if not output_dataset_vars:
        return train_dl, val_dl, test_dl, imbalance
    return (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        (n_concepts, n_classes, concept_group_map),
    )
