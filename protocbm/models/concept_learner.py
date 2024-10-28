from typing import *
import torch
from torch.nn import Sequential, Dropout
import torch.nn.functional as F
from torchvision.models.resnet import ResNet
import lightning as L
import torchmetrics
from protocbm.models.utils import *


def resnet_add_dropout_final(model, dropout_rate):
    assert isinstance(model, ResNet), "Attempting to add dropout to unsupported model (!= Resnet)"
    fc = model.fc
    new_fc = Sequential(
                Dropout(dropout_rate),
                fc
            )
    model.fc = new_fc
    return model


class ConceptLearner(L.LightningModule):
    def __init__(self, 
                 n_concepts: int,
                 dropout_rate: Optional[float] = None,
                 optimiser: torch.optim.Optimizer=None,
                 lr_scheduler: Mapping = None,
                 arch: Optional[str] = None, 
                 model: torch.nn.Module = None,
                 metrics = None,
                 prog_bar_metrics: Optional[Sequence[str]] = None,
                 **kwargs):
        super().__init__()
        hyperparams = ('n_concepts', 'dropout_rate')
        if model is not None:
            self.backbone = model
        elif arch is not None:
            self.backbone = get_backbone(arch, n_concepts)
            hyperparams += ('arch', )
        else:
            raise ValueError("Either model or arch must be provided")

        self.save_hyperparameters(*hyperparams)
        self.optimiser = optimiser or torch.optim.Adam
        self.lr_scheduler = lr_scheduler
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.backbone = resnet_add_dropout_final(self.backbone, dropout_rate)

        if metrics is None:
            self.metrics = torch.nn.ModuleDict({
                "acc": torchmetrics.Accuracy(task='binary'), 
                "f1":  torchmetrics.F1Score(task='binary'),
                "auc": torchmetrics.AUROC(task='binary'), 
            })
        else:
            self.metrics = torch.nn.ModuleDict(metrics)
        
        self.prog_bar_metrics = prog_bar_metrics if prog_bar_metrics is not None else ["acc"]

    def _unpack_batch(self, batch):
        x = batch[0]
        if isinstance(batch[1], list):
            y, c = batch[1]
        else:
            y, c = batch[1], batch[2]
        c = c.float()
        return x, y, c
    
    def forward(self, x):
        logit = self.backbone(x)
        prob = F.sigmoid(logit)
        return logit, prob
    
    def _run_step(self, batch, step_type):
        x, y, c = self._unpack_batch(batch)
        c_logits, c_prob = self(x)
        
        loss = self.loss_fn(c_logits, c)
        self.log(f"{step_type}/loss", loss)
        for name, metric_fn in self.metrics.items():
            prog_bar = name in self.prog_bar_metrics
            self.log(f"{step_type}/{name}", metric_fn(c_prob, c), prog_bar=prog_bar)
        return loss
    
    def training_step(self, batch):
        return self._run_step(batch, "train")
    
    def validation_step(self, batch):
        return self._run_step(batch, "val")
    
    def test_step(self, batch):
        return self._run_step(batch, "test")
    
    def predict_step(self, batch):
        x, y, c = self._unpack_batch(batch)
        c_logits, c_prob = self(x)
        
        return {
            "c": c,
            "c_prob": c_prob,
            "c_emb": c_logits,
            "y": y        
        }
    
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


class CEMConceptLearner(ConceptLearner):
    def __init__(self, 
                 n_concepts: int,
                 optimiser: torch.optim.Optimizer,
                 lr_scheduler: Mapping,
                 arch: Optional[str] = None, 
                 model: torch.nn.Module = None,
                 metrics: Optional[Mapping[str, torchmetrics.Metric]] = None,
                 emb_size: int = 32,
                 shared_prob_gen: bool = True,
                 activation: str = "leakyrelu",
                 **kwargs):
        super().__init__(n_concepts=None,
                         optimiser=optimiser,
                         lr_scheduler=lr_scheduler,
                         arch=arch,
                         model=model,
                         metrics=metrics,
                         **kwargs)
        
        self.n_concepts = n_concepts
        self.emb_size = emb_size
        self.shared_prob_gen = shared_prob_gen
        self.activation = get_activation_fn(activation)()
        
        self.concept_context_generators = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(list(self.backbone.modules())[-1].out_features, 2 * self.emb_size),
                                self.activation)
            for _ in range(n_concepts)
        ])
        
        self.concept_prob_generators = torch.nn.ModuleList([
            torch.nn.Linear(2 * self.emb_size, 1)
            for _ in range(1 if shared_prob_gen else n_concepts)
        ])
        
    def forward(self, x):
        intermid = self.backbone(x)
        
        logits_l = []
        probs_l = []
        
        for i, context_gen in enumerate(self.concept_context_generators):
            if self.shared_prob_gen:
                prob_gen = self.concept_prob_generators[0]
            else:
                prob_gen = self.concept_prob_generators[i]
            
            context = context_gen(intermid)
            logit = prob_gen(context)
            logits_l.append(logit)
            prob = F.sigmoid(logit)
            probs_l.append(prob)
        
        logits = torch.cat(logits_l, dim=1)
        probs = torch.cat(probs_l)
        return logits, probs
    
