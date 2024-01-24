import sys
import logging
from abc import ABC, abstractmethod
from functools import partial

import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics import Accuracy
import lightning as L

from proto_models.dknn.dknn_layer import *
from cem.models.cbm import ConceptBottleneckModel
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
    else:
        raise ValueError(name)


def get_optimiser(name: str):
    lookup = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam
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
        self.concept_loss_fn = torch.nn.BCELoss()
        
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
    
    def _process_batch(self, batch):
        img_data, class_label, concept_label = batch
        return img_data, class_label, concept_label
        
    def _prepare_prototypes(self):
        print("Preparing prototypes")
        # compute a cache of concept activations as prototypes
        list_c_activations = []
        list_cls_labels = []
        
        for batch in tqdm(self.proto_dataloader):
            img_data, class_label, _ = self._process_batch(batch)
            
            with torch.no_grad():
                c_activations = self.x2c_model(img_data.to(self.device))
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
        delattr(self, "proto_concepts")
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
        optimiser = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimiser
    
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
        c_activation="sigmoid",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        top_k_accuracy_c2y=1):
        super().__init__(
            n_concepts=n_concepts,
            concept_loss_weight=concept_loss_weight,
            proto_loss_weight=proto_loss_weight,
            x2c_model=x2c_model,
            c_activation=c_activation,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        self.dknn_k = dknn_k
        self.dknn_tau = dknn_tau
        self.dknn_method = dknn_method
        self.dknn_num_samples = dknn_num_samples
        self.dknn_simiarity = dknn_similarity
        
        self.n_classes = n_classes
        self.proto_model = DKNN(k=dknn_k,
                                tau=dknn_tau,
                                method=dknn_method,
                                num_samples=dknn_num_samples,
                                similarity=dknn_similarity)
        self.accuracy_c2y = Accuracy(task="multiclass",
                                     num_classes=n_classes,
                                     top_k=top_k_accuracy_c2y)
        
    def calculate_proto_loss(self, scores, query_y, neighbour_y):
        query_y_one_hot = F.one_hot(query_y, self.n_classes)
        neighbour_y_one_hot = F.one_hot(neighbour_y, self.n_classes)
        
        # top_k_ness = self.get_proto_model()(query_x, neighbour_x)  # (B*X, N*X) => (B, N)
        correct = (query_y_one_hot.unsqueeze(1) * neighbour_y_one_hot.unsqueeze(0)).sum(-1) # (B, N)
        correct_in_top_k = (correct * scores).sum(-1)  # [B]
        loss = -correct_in_top_k
        
        accuracy = dknn_results_analysis(scores, neighbour_y, query_y, self.dknn_k)
        
        return {
            "loss": loss.mean(), "accuracy": accuracy
        }
    
class ProtoCBMDKNNSequential(ProtoCBMDKNN, SequentialCBM):
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
        c_activation="sigmoid",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05):
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
            c_activation=c_activation,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        SequentialCBM.__init__(self)
        self.save_hyperparameters()
    
    def on_train_epoch_start(self) -> None:
        if self.training_stage == "c2y":
            if not self.has_prototypes():
                self._prepare_prototypes()
                
    def on_validation_epoch_start(self) -> None:
        if self.training_stage == "c2y":
            if not self.has_prototypes():
                self._prepare_prototypes()
    
    def _forward(self, batch, batch_idx, mask_prototype=True, log_prefix=""):
        logging.debug("_forward")
        img_data, truth_y, truth_c = self._process_batch(batch)
        
        if self.training_stage == "x2c":
            pred_c = self.x2c_model(img_data)
            pred_c = self.c_act(pred_c)
            x2c_loss = self.concept_loss_fn(pred_c, truth_c)
            return {
                "loss": x2c_loss,
                "x2c_loss": x2c_loss,
            }
        elif self.training_stage == "c2y":
            assert self.has_prototypes()
            
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
            
            pred_c = self.x2c_model(img_data)
            pred_c = self.c_act(pred_c)
            pred_y = self.proto_model(pred_c, proto_x)
            proto_results = self.calculate_proto_loss(pred_y, truth_y, proto_y)
            return {
                "loss": proto_results['loss'],
                "c2y_loss": proto_results['loss'],
                "c2y_acc": proto_results['accuracy']
            }
        else:
            raise ValueError(self.training_stage)
        

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
        dknn_method='deterministic',
        dknn_num_samples=-1,
        dknn_similarity='euclidean',
        c_activation="sigmoid",
        epoch_proto_recompute=1,
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,):
        
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
            c_activation=c_activation,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        JointCBM.__init__(self)
        self.proto_dataloader = proto_train_dl
        self._epoch_counter = 0
        self.epoch_proto_recompute = epoch_proto_recompute
        self.save_hyperparameters()
            
    def _forward(self, batch, batch_idx, mask_prototype=True):
        logging.debug("_forward")
        img_data, truth_y, truth_c = self._process_batch(batch)
        
        if not self.has_prototypes():
            self._prepare_prototypes()
        pred_c = self.c_act(self.x2c_model(img_data))
        x2c_loss = self.concept_loss_fn(pred_c, truth_c)
        
        # Masking identical samples from training set (only used for training)
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
        
        pred_y = self.proto_model(pred_c, proto_x)
        proto_results = self.calculate_proto_loss(pred_y, truth_y, proto_y)
        
        loss = self.concept_loss_weight * x2c_loss + \
            self.proto_loss_weight * proto_results['loss']
        
        return {
            "x2c_loss": x2c_loss,
            "c2y_loss": proto_results['loss'],
            "c2y_acc": proto_results['accuracy'],
            "loss": loss
        }
        
    def on_train_epoch_end(self) -> None:
        self._epoch_counter += 1
        if (self._epoch_counter % self.epoch_proto_recompute) == 0:
            self.clear_prototypes()