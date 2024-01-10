from functools import partial

import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import lightning as L

from cem.train.utils import utils
from proto_models.dknn.dknn_layer import DKNN, dknn_loss
from cem.models.cbm import ConceptBottleneckModel


def get_proto_model_and_loss(name="dknn"):
    if name.startswith("dknn"):
        parts = name.split("_")
        if len(parts) == 1:
            return DKNN, dknn_loss
        else:
            sampling_method = parts[-1]
            return DKNN, partial(dknn_loss, method=sampling_method)
            
    else:
        raise ValueError(name)


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

class ProtoCBM(L.LightningModule):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        proto_dataloader: DataLoader,
        concept_loss_weight=1,
        proto_loss_weight=1,
        
        x2c_model=None,
        # x2c_arch=utils.wrap_pretrained_model(resnet50),
        c2y_model=None,
        c2y_loss=None,
        c2y_proto="dknn_deterministic",
        c_activation="leaklyrelu",
        
        optimiser="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-65,
        
        top_k_accuracy=None,
        training_mode="sequential"
    ):
        super().__init__(self)
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.proto_dataloader = proto_dataloader
        
        if x2c_model is not None:
            self.x2c_model = x2c_model
        else:
            # TODO: Replace this with a proper wraper function
            self.x2c_model = resnet50(weights=ResNet50_Weights.DEFAULT, num_classes=n_concepts)
        
        self.c_act = get_activation_fn(c_activation)()
        
        if c2y_model is not None and c2y_loss is not None:
            self.c2y_model = c2y_model
            self.c2y_loss = c2y_loss
        else:
            c2y_arch, self.c2y_loss = get_proto_model_and_loss(c2y_proto)
            c2y_model = c2y_arch()
            
        self.proto_loss_weight = proto_loss_weight
        self.concept_loss_weight = concept_loss_weight
        self.training_mode = training_mode
        if self.training_mode == "sequential":
            self._training_stage = "x2c"
        else:
            raise ValueError(training_mode)
    
    def _process_batch(self, batch):
        img_data, class_label, concept_label = batch
        return img_data, class_label, concept_label
        
    def _prepare_prototypes(self):
        # compute a cache of concept activations as prototypes
        list_c_activations = []
        list_cls_labels = []
        
        for batch in self.proto_dataloader:
            img_data, class_label, _ = self._process_batch(batch)
            
            with torch.no_grad():
                c_activations = self.c2y_model(img_data.to(self.device))
                list_c_activations.append(c_activations)
                list_cls_labels.append(class_label.to(self.device))
                
        self.register_buffer('proto_concepts', torch.concat(list_c_activations))
        self.register_buffer('proto_classes', torch.concat(list_cls_labels))
    
    def _assert_has_prototypes(self):
        assert hasattr(self, "proto_concepts")
        assert hasattr(self, "proto_classes")
    
    def complete_x2c(self, train_dl):
        assert self.training_mode == "sequential"
        # freeze x2c model
        for p in self.c2y_model.parameters():
            p.require_grad = False
        self.training_stage = "y2c"
        
    def on_train_epoch_start(self):
        if self.training_mode == "sequential" and self.training_stage == "x2c":
            return
        self._prepare_prototypes()
        
    def _forward(self, batch):
        img_data, y_truth, c_truth = self._process_batch(batch)
        
        if self.training_mode == "sequential":
            if self.training_stage == "x2c":
                c_hat = self.x2c_model(img_data)
                c_hat = self.c_cact(c_hat)
                return self.concept_loss_weight * torch.nn.functional.cross_entropy(c_hat, c_truth)
            elif self.training_stage == "y2c":
                self._assert_has_prototypes()
                with torch.no_grad():
                    c_hat = self.x2c_model(img_data)
                    c_hat = self.c_act(c_hat)
                return self.proto_loss_weight * self.c2y_loss(self.c2y_model, c_hat, self.proto_concepts, y_truth, self.proto_classes)
            else:
                raise ValueError(self.training_stage)
        elif self.training_mode == "joint":
            self._assert_has_prototypes()
            c_hat = self.x2c_model(img_data)
            c_hat = self.c_cact(c_hat)
            concept_loss = self.concept_loss_weight * torch.nn.functional.cross_entropy(c_hat, c_truth)
            
            proto_loss = self.proto_loss_weight * self.c2y_loss(self.c2y_model, c_hat, self.proto_concepts, y_truth, self.proto_classes)
            return concept_loss + proto_loss
        else:
            raise ValueError(self.training_mode)

    def forward(self, batch):
        return self._forward(batch)

    def training_step(self, batch, batch_idx):
        return self._forward(batch)
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._forward(batch)
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._forward(batch)
    
    def predict_step(self, batch, batch_idx):
        img_data, _, _ = self._process_batch(batch)
        
        self._assert_has_prototypes()
        with torch.no_grad():
            c_hat = self.x2c_model(img_data)
            c_hat = self.c_cact(c_hat)
            y_pred = self.c2y_model()