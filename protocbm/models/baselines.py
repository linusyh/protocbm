from typing import *
import torch
import torch.nn.functional as F
import lightning as L
import torchmetrics
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from protocbm.models.utils import *


class StandardResNet(L.LightningModule):
    def __init__(self,
                 n_tasks: int,
                 arch: str = 'resnet18',
                 optimiser = None,
                 lr_scheduler: Mapping = None,
                 metrics = None,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters('n_tasks', 'arch')
        self.n_tasks = n_tasks
        self.backbone = get_backbone(arch, n_tasks)
        if metrics is None:
            self.metrics = torch.nn.ModuleDict({
                "acc": torchmetrics.Accuracy(task='multiclass', num_classes=n_tasks), 
                "f1":  torchmetrics.F1Score(task='multiclass', num_classes=n_tasks),
                "auc": torchmetrics.AUROC(task='multiclass', num_classes=n_tasks), 
            })
        self.optimiser = optimiser or torch.optim.Adam
        self.lr_scheduler = lr_scheduler
        self.loss = CrossEntropyLoss()
        
    def forward(self, x):
        logit = self.backbone(x)
        prob = F.sigmoid(logit)
        return prob
    
    def _unpack_batch(self, batch):
        x = batch[0]
        if isinstance(batch[1], list):
            y, c = batch[1]
        else:
            y, c = batch[1], batch[2]
        c = c.float()
        return x, y, c
    
    def _run_step(self, batch, mode):
        x, y = self._unpack_batch(batch)[:2]
        y_pred = self(x)
        
        for name, metric_fn in self.metrics.items():
            self.log(f"{mode}/y_{name}", metric_fn(y_pred, y), prog_bar=True)
        
        loss = self.loss(y_pred, y)
        self.log(f"{mode}/y_loss", loss)
        self.log(f"{mode}/loss", loss)
        return loss
    
    def training_step(self, batch):
        return self._run_step(batch, mode="train")
    
    def validation_step(self, batch):
        return self._run_step(batch, mode="val")
    
    def test_step(self, batch):
        return self._run_step(batch, mode="test")
    
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
        
        
class MultiTaskResNet(L.LightningModule):
    def __init__(self,
                 n_concepts: int,
                 n_tasks: int,
                 arch: str = 'resnet18',
                 optimiser = None,
                 lr_scheduler: Mapping = None,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters('n_tasks', 'arch')
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.backbone = get_backbone(arch, n_concepts+n_tasks)
        self.concept_metrics = torch.nn.ModuleDict({
            "acc": torchmetrics.Accuracy(task='binary'), 
            "f1":  torchmetrics.F1Score(task='binary'),
            "auc": torchmetrics.AUROC(task='binary'), 
        })
        
        self.task_metrics = torch.nn.ModuleDict({
            "acc": torchmetrics.Accuracy(task='multiclass', num_classes=n_tasks), 
            "f1":  torchmetrics.F1Score(task='multiclass', num_classes=n_tasks),
            "auc": torchmetrics.AUROC(task='multiclass', num_classes=n_tasks), 
        })
        self.optimiser = optimiser or torch.optim.Adam
        self.lr_scheduler = lr_scheduler
        self.concept_loss = BCEWithLogitsLoss()
        self.task_loss = CrossEntropyLoss()
        
    def forward(self, x):
        logit = self.backbone(x)
        return logit
    
    def _unpack_batch(self, batch):
        x = batch[0]
        if isinstance(batch[1], list):
            y, c = batch[1]
        else:
            y, c = batch[1], batch[2]
        c = c.float()
        return x, y, c
    
    def _run_step(self, batch, mode):
        x, y, c = self._unpack_batch(batch)
        logit = self(x)
        c_logit = logit[:, :self.n_concepts]
        c_pred = F.sigmoid(c_logit)
        y_pred = F.sigmoid(logit[:, self.n_concepts:])
        for name, metric_fn in self.concept_metrics.items():
            self.log(f"{mode}/c_{name}", metric_fn(c_pred, c), prog_bar=True)
            
        for name, metric_fn in self.task_metrics.items():
            self.log(f"{mode}/y_{name}", metric_fn(y_pred, y), prog_bar=True)
        
        c_loss = self.concept_loss(c_logit, c)
        y_loss = self.task_loss(y_pred, y)
        loss = c_loss + y_loss
        self.log(f"{mode}/c_loss", c_loss)
        self.log(f"{mode}/y_loss", y_loss)
        self.log(f"{mode}/loss", loss)
        
        return loss
    
    def training_step(self, batch):
        return self._run_step(batch, mode="train")
    
    def validation_step(self, batch):
        return self._run_step(batch, mode="val")
    
    def test_step(self, batch):
        return self._run_step(batch, mode="test")
    
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