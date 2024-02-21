import sys
import logging
from abc import ABC, abstractmethod
from functools import partial

import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics import Accuracy
import lightning as L

from protocbm.dknn.dknn_layer import *
from tqdm import tqdm


def get_activation_fn(name="relu"):
    if name == "relu":
        return torch.nn.ReLU
    elif name == "leakyrelu":
        return torch.nn.LeakyReLU
    elif name == "sigmoid":
        return torch.nn.Sigmoid
    elif name == "selu":
        return torch.nn.SELU
    elif name == "tanh":
        return torch.nn.Tanh
    elif name in ["none", "identity", None]:
        return torch.nn.Identity
    else:
        raise ValueError(name)


def get_optimiser(name: str):
    lookup = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "adamax": torch.optim.Adamax,
    }
    return lookup.get(name.lower(), torch.optim.Adam)
        
        
class ProtoCBM(L.LightningModule, ABC):
    def __init__(
        self,
        n_concepts,
        concept_loss_weight=1,
        proto_loss_weight=1,
        x2c_model=None,
        c_activation="sigmoid",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05, 
        batch_process_fn=None,
        plateau_lr_scheduler_enable=False,
        plateau_lr_scheduler_monitor="val_c2y_acc",
        plateau_lr_scheduler_mode="max",
        plateau_lr_scheduler_patience=10,
        plateau_lr_scheduler_factor=0.1,
        plateau_lr_scheduler_min_lr=1e-6,
        plateau_lr_scheduler_threshold=0.01,
        plateau_lr_scheduler_cooldown=0
    ):
        super().__init__()
        self.n_concepts = n_concepts
        if x2c_model is not None:
            self.x2c_model = x2c_model
        else:
            self.x2c_model = resnet50(weights=ResNet50_Weights.DEFAULT, num_classes=n_concepts)
        
        self.c_act = get_activation_fn(c_activation)()
            
        self.proto_loss_weight = proto_loss_weight
        self.concept_loss_weight = concept_loss_weight
        self.concept_loss_fn = torch.nn.BCEWithLogitsLoss()
        
        self.batch_process_fn = batch_process_fn
        
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.plateau_lr_scheduler_enable = plateau_lr_scheduler_enable
        self.plateau_lr_scheduler_monitor = plateau_lr_scheduler_monitor
        self.plateau_lr_scheduler_mode = plateau_lr_scheduler_mode
        self.plateau_lr_scheduler_patience = plateau_lr_scheduler_patience
        self.plateau_lr_scheduler_factor = plateau_lr_scheduler_factor
        self.plateau_lr_scheduler_min_lr = plateau_lr_scheduler_min_lr
        self.plateau_lr_scheduler_threshold = plateau_lr_scheduler_threshold
        self.plateau_lr_scheduler_cooldown = plateau_lr_scheduler_cooldown
    
    def _process_batch(self, batch):
        if self.batch_process_fn is not None:
            return self.batch_process_fn(batch)
        else:
            return batch
        
    def prepare_prototypes(self):
        print("Preparing prototypes")
        # compute a cache of concept activations as prototypes
        list_c_activations = []
        list_cls_labels = []
        
        for batch in tqdm(self.proto_dataloader):
            img_data, class_label, _ = self._process_batch(batch)
            
            with torch.no_grad():
                c_activations = self.c_act(self.x2c_model(img_data.to(self.device)))
                list_c_activations.append(c_activations)
                list_cls_labels.append(class_label.to(self.device))
                
            # Ensure consistent printing of progress bar
            sys.stdout.flush()
            sys.stderr.flush()
                
        self.register_buffer('proto_concepts', torch.concat(list_c_activations))
        self.register_buffer('proto_classes', torch.concat(list_cls_labels))
    
    def has_prototypes(self):
        return hasattr(self, "proto_concepts") and hasattr(self, "proto_classes")
    
    def clear_prototypes(self):
        if hasattr(self, "proto_concepts"):
            delattr(self, "proto_concepts")
        if hasattr(self, "proto_classes"):
            delattr(self, "proto_classes")
        
    @abstractmethod
    def _forward(self, batch, batch_idx, mask_prototype=False):
        raise NotImplementedError()
    
    def training_step(self, batch, batch_idx):
        results = self._forward(batch, batch_idx, mask_prototype=True)
        for name, value in results.items():
            self.log(f"train_{name}", value, on_epoch=True, prog_bar=True)
        
        return results['loss']
    
    def validation_step(self, batch, batch_idx):
        results = self._forward(batch, batch_idx, mask_prototype=False)
        for name, value in results.items():
            self.log(f"val_{name}", value, on_epoch=True, prog_bar=True)
        
        return results['loss']
    
    def test_step(self, batch, batch_idx):
        results = self._forward(batch, batch_idx, mask_prototype=False)
        for name, value in results.items():
            self.log(f"test_{name}", value, on_epoch=True, prog_bar=True)
        
        return results['loss']
            
    def configure_optimizers(self):
        objects = {}
        objects['optimizer'] = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.plateau_lr_scheduler_enable:
            objects['lr_scheduler'] = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(objects['optimizer'],
                                                                        mode=self.plateau_lr_scheduler_mode,
                                                                        patience=self.plateau_lr_scheduler_patience,
                                                                        factor=self.plateau_lr_scheduler_factor,
                                                                        min_lr=self.plateau_lr_scheduler_min_lr,
                                                                        threshold=self.plateau_lr_scheduler_threshold,
                                                                        cooldown=self.plateau_lr_scheduler_cooldown,),
                "monitor": self.plateau_lr_scheduler_monitor,
                "strict": False,
            }
        return objects
    
    @abstractmethod
    def calculate_proto_loss(self, query_x, neighbour_x, query_y, neighbour_y):
        raise NotImplemented()
        
        
class SequentialCBM:
    def __init__(self):
        self._training_stage = "x2c"
        
    @property
    def training_mode(self):
        return 'sequential'
    
    @property
    def training_stage(self):
        return self._training_stage
    
    @training_stage.setter
    def training_stage(self, stage):
        assert stage in ["x2c", "c2y"]
        self._training_stage = stage
    
    @abstractmethod
    def complete_x2c(self, proto_train_dl):
        for p in self.x2c_model.parameters():
            p.require_grad = False
        self.training_stage = "c2y"
        self.proto_dataloader = proto_train_dl
        
        
class JointCBM:
    @property
    def training_mode(self):
        return 'joint'
    
    
class ProtoCBMDKNN(ProtoCBM, ABC):
    def __init__(self,
        n_concepts,
        n_classes,
        concept_loss_weight=1,
        proto_loss_weight=1,
        x2c_model=None,
        dknn_k=2,
        dknn_tau=1,
        dknn_method='deterministic',
        dknn_num_samples=-1,
        dknn_similarity='euclidean',
        dknn_loss_type='minus_count',
        dknn_max_neighbours=-1,
        c_activation="sigmoid",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        plateau_lr_scheduler_enable=False,
        plateau_lr_scheduler_monitor="val_c2y_acc",
        plateau_lr_scheduler_mode="max",
        plateau_lr_scheduler_patience=10,
        plateau_lr_scheduler_factor=0.1,
        plateau_lr_scheduler_min_lr=1e-6,
        plateau_lr_scheduler_threshold=0.01,
        plateau_lr_scheduler_cooldown=0,
        batch_process_fn=None):
        super().__init__(
            n_concepts=n_concepts,
            concept_loss_weight=concept_loss_weight,
            proto_loss_weight=proto_loss_weight,
            x2c_model=x2c_model,
            c_activation=c_activation,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
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
        
        self.dknn_k = dknn_k
        self.dknn_tau = dknn_tau
        self.dknn_method = dknn_method
        self.dknn_num_samples = dknn_num_samples
        self.dknn_simiarity = dknn_similarity
        self.dknn_loss_type = dknn_loss_type        
        self.dknn_loss_function = dknn_loss_factory(dknn_loss_type)
        self.dknn_max_neighbours = dknn_max_neighbours

        self.x2c_accuracy = Accuracy("binary")
        
        self.n_classes = n_classes
        self.proto_model = DKNN(k=dknn_k,
                                tau=dknn_tau,
                                hard=False,
                                method=dknn_method,
                                num_samples=dknn_num_samples,
                                similarity=dknn_similarity,)
        
    def calculate_proto_loss(self, scores, query_y, neighbour_y):
        query_y_one_hot = F.one_hot(query_y, self.n_classes)  # (B, C)
        neighbour_y_one_hot = F.one_hot(neighbour_y, self.n_classes)  # (N, C)
        correct = (query_y_one_hot.unsqueeze(1) * neighbour_y_one_hot.unsqueeze(0)).sum(-1).float() # (B, N)
        
        loss = self.dknn_loss_function(scores, correct)
        
        results = dknn_results_analysis(scores, neighbour_y, query_y, self.dknn_k)
        results['loss'] = loss
        
        return results


class ProtoCBMDKNNJoint(ProtoCBMDKNN, JointCBM):
    def __init__(self,
        n_classes,
        n_concepts,
        proto_train_dl,
        concept_loss_weight=1,
        proto_loss_weight=1,
        x2c_model=None,
        dknn_k=2,
        dknn_tau=1,
        dknn_method="deterministic",
        dknn_num_samples=-1,
        dknn_similarity="euclidean",
        dknn_loss_type="minus_count",
        dknn_max_neighbours=-1,
        c_activation="sigmoid",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        x2c_only_epochs=0,
        epoch_proto_recompute=1,
        plateau_lr_scheduler_enable=False,
        plateau_lr_scheduler_monitor="val_c2y_acc",
        plateau_lr_scheduler_mode="max",
        plateau_lr_scheduler_patience=10,
        plateau_lr_scheduler_factor=0.1,
        plateau_lr_scheduler_min_lr=1e-6,
        plateau_lr_scheduler_threshold=0.01,
        plateau_lr_scheduler_cooldown=0,
        batch_process_fn=None):
        
        super().__init__(
            n_concepts=n_concepts,
            n_classes=n_classes,
            concept_loss_weight=concept_loss_weight,
            proto_loss_weight=proto_loss_weight,
            x2c_model=x2c_model,
            dknn_k=dknn_k,
            dknn_tau=dknn_tau,
            dknn_method=dknn_method,
            dknn_num_samples=dknn_num_samples,
            dknn_similarity=dknn_similarity,
            dknn_loss_type=dknn_loss_type,
            dknn_max_neighbours=dknn_max_neighbours,
            c_activation=c_activation,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
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
        JointCBM.__init__(self)
        self.proto_dataloader = proto_train_dl
        self._epoch_counter = 0
        self.epoch_proto_recompute = max(1, epoch_proto_recompute)
        self._x2c_only_epochs = max(0, x2c_only_epochs)
        self.save_hyperparameters(ignore=["proto_train_dl", "x2c_model"])
    
    def x2c_only(self):
        return self._epoch_counter < self._x2c_only_epochs
            
    def _forward(self, batch, batch_idx, mask_prototype=True):
        logging.debug("_forward")
        x, y, c = self._process_batch(batch)
        pred_c_logit = self.x2c_model(x)
        pred_c = self.c_act(pred_c_logit)
        pred_c_scores = F.sigmoid(pred_c_logit)
        x2c_loss = self.concept_loss_fn(pred_c_logit, c.float())
        x2c_accuracy = self.x2c_accuracy(pred_c_scores, c)
        
        if self.x2c_only():
            return {
                "loss": x2c_loss * self.concept_loss_weight,
                "x2c_loss": x2c_loss,
                "x2c_acc": x2c_accuracy,
            }
            
        if not self.has_prototypes():
            self.prepare_prototypes()
        
        if mask_prototype:
            bsz = self.proto_dataloader.batch_size
            idx_start = batch_idx * bsz
            idx_end = min(idx_start + bsz, self.proto_concepts.shape[0])
            
            mask = torch.ones_like(self.proto_classes, dtype=torch.bool)
            mask[idx_start: idx_end] = 0
            
            proto_x = self.proto_concepts[mask, ...]
            proto_y = self.proto_classes[mask]
        else:
            proto_x = self.proto_concepts
            proto_y = self.proto_classes
            
        if self.dknn_max_neighbours > 0 and proto_x.shape[0] > self.dknn_max_neighbours:
            idx = torch.randperm(proto_x.shape[0])[:self.dknn_max_neighbours]
            proto_x = proto_x[idx, ...]
            proto_y = proto_y[idx]    
        
        pred_y = self.proto_model(pred_c, proto_x)
        proto_results = self.calculate_proto_loss(pred_y, y, proto_y)
        
        loss = self.concept_loss_weight * x2c_loss + \
            self.proto_loss_weight * proto_results['loss']
        
        return {
            "loss":         loss,
            "x2c_loss":     x2c_loss,
            "x2c_acc":      x2c_accuracy,
            "c2y_loss":     proto_results['loss'],
            "c2y_acc":      proto_results['neighbour_accuracy'],
            "c2y_acc_cls":  proto_results['class_accuracy']
        }
        
    def on_train_epoch_end(self) -> None:
        self._epoch_counter += 1
        if (self._epoch_counter % self.epoch_proto_recompute) == 0:
            self.clear_prototypes()