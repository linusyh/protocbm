from typing import *
import torch
import torch.nn.functional as F
import lightning as L
import torch.utils
import torchmetrics
from tqdm import tqdm

from protocbm.models.utils import *
from protocbm.dknn.dknn_layer import *

class ProtoTaskLearner(L.LightningModule):
    def __init__(self, 
                 n_concepts: int,
                 n_classes: int, 
                 hidden_layers: List[int],
                 proto: DKNN,
                 proto_loss: torch.nn.Module,
                 optimiser,
                 lr_scheduler: Mapping = None,
                 proto_dataloader: torch.utils.data.DataLoader = None,
                 max_neighbours: int = -1,
                 metric_prefix: str = "c2y_",
                 metric_translate: Dict[str, str] = None,
                 **kwargs):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        logging.debug(f"[ProtoTaskLearner] Using hidden layers: {hidden_layers}")
        layers = [torch.nn.Linear(self.n_concepts, hidden_layers[0])]
        for i in range(1, len(hidden_layers)):
            layers.append(torch.nn.BatchNorm1d(hidden_layers[i-1]))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        self.pre_proto_model = torch.nn.Sequential(*layers)

        self.proto_model = proto
        self.proto_loss_fn = proto_loss

        self.optimiser = optimiser
        self.lr_scheduler = lr_scheduler
        self.proto_dataloader = proto_dataloader
        self.max_neighbours = max_neighbours

        if metric_translate is None:
            self.metric_translate = {}
            self.metric_translate["neighbour_accuracy"] = "acc"
            self.metric_translate["class_accuracy"] = "acc_cls"
        self.metric_prefix = metric_prefix

    def _unpack_batch(self, batch):
        x = batch[0]
        if isinstance(batch[1], list):
            y, c = batch[1]
        else:
            y, c = batch[1], batch[2]
        c = c.float()
        return x, y, c
    
    def set_proto_dataloader(self, proto_dataloader):
        self.proto_dataloader = proto_dataloader
    
    def prepare_prototypes(self):
        prog_bar = tqdm(self.proto_dataloader, desc="Preparing prototypes")
        list_c_activations = []
        list_cls_labels = []
        
        for batch in prog_bar:
            x, y, c = self._unpack_batch(batch)
            list_c_activations.append(c)
            list_cls_labels.append(y)

        self.register_buffer('proto_concepts', torch.concat(list_c_activations).to(self.device))
        self.register_buffer('proto_classes', torch.concat(list_cls_labels).to(self.device))
    
    def has_prototypes(self):
        return hasattr(self, "proto_concepts") and hasattr(self, "proto_classes")
    
    def clear_prototypes(self):
        if hasattr(self, "proto_concepts"):
            delattr(self, "proto_concepts")
        if hasattr(self, "proto_classes"):
            delattr(self, "proto_classes")

    def forward(self, x):
        logit = self.backbone(x)
        prob = F.sigmoid(logit)
        return logit, prob

    def proto_loss(self, proto_out, true_y, proto_y):
        correct = torch.eq(true_y.unsqueeze(1), proto_y.unsqueeze(0)).float()
        loss = self.proto_loss_fn(proto_out, correct)
        return loss
    
    def proto_metrics(self, proto_out, true_y, proto_y):
        return dknn_results_analysis(proto_out, proto_y, true_y, self.proto_model.k)
    
    def _forward(self, batch, batch_idx, step_type):
        x, y, c = self._unpack_batch(batch)

        if not self.has_prototypes():
            self.prepare_prototypes()

        if step_type == "train":
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

        if self.max_neighbours > 0:
            if proto_x.shape[0] > self.max_neighbours:
                idx = torch.randperm(proto_x.shape[0])[:self.max_neighbours]
                proto_x = proto_x[idx]
                proto_y = proto_y[idx]
        
        pred_h = self.pre_proto_model(c)
        proto_h  = self.pre_proto_model(proto_x)
        proto_out = self.proto_model(pred_h, proto_h)

        loss = self.proto_loss(proto_out, y, proto_y)
        metrics = self.proto_metrics(proto_out, y, proto_y)

        return loss, metrics
    
    def _run_step(self, batch, batch_idx, step_type):
        loss, metrics = self._forward(batch, batch_idx, step_type)
        self.log(f"{step_type}_{self.metric_prefix}loss", loss, prog_bar=True)
        for k, v in metrics.items():
            if k in self.metric_translate:
                k = self.metric_translate[k]
            self.log(f"{step_type}_{self.metric_prefix}{k}", v, prog_bar=True)
        return loss, metrics
    
    def training_step(self, batch, batch_idx):
        loss, metrics = self._run_step(batch, batch_idx, "train")
        return loss
   
    def validation_step(self, batch, batch_idx):
        loss, metrics = self._run_step(batch, batch_idx, "val")
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, metrics = self._run_step(batch, batch_idx, "test")
        return loss
    
    def predict_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "predict")
    
    def configure_optimizers(self):
        optimiser = self.optimiser(params=self.parameters())
        scheduler = self.lr_scheduler
        if scheduler is None:
            return optimiser
        
        scheduler["scheduler"] = scheduler["scheduler"](optimizer=optimiser)
        print(scheduler["scheduler"])
        return {
            "optimizer": optimiser,
            "lr_scheduler": dict(scheduler),
        }