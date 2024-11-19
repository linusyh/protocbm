from protocbm.models.cbm import *
from protocbm.model import *
from protocbm.models.utils import *
from protocbm.dknn.dknn_layer import *

import sys
from functools import partial
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import *


def compute_concept_accuracy(c_pred, c_true):
    c_pred = (c_pred.cpu().detach().numpy() >= 0.5).astype(np.int32)
    c_true = (c_true.cpu().detach().numpy() > 0.5).astype(np.int32)

    c_accuracy = c_auc = c_f1 = 0
    for i in range(c_true.shape[-1]):
        true_vars = c_true[:, i]
        pred_vars = c_pred[:, i]
        c_accuracy += sklearn.metrics.accuracy_score(
            true_vars, pred_vars
        ) / c_true.shape[-1]

        if len(np.unique(true_vars)) == 1:
            c_auc += np.mean(true_vars == pred_vars)/c_true.shape[-1]
        else:
            c_auc += sklearn.metrics.roc_auc_score(
                true_vars,
                pred_vars,
            )/c_true.shape[-1]
        c_f1 += sklearn.metrics.f1_score(
            true_vars,
            pred_vars,
            average='macro',
        )/c_true.shape[-1]
    
    return c_accuracy, c_auc, c_f1


class ProtoCBM(ConceptBottleneckModel):
    def __init__(self,
                 n_concepts,
                 n_tasks,
                 proto_train_dl: DataLoader,
                 concept_loss_weight=0.01,
                 task_loss_weight=1,
 
                 extra_dims=0,
                 bool=False,
                 sigmoidal_extra_capacity=True,
                 bottleneck_nonlinear=None,
                 output_latent=False,
                 x2c_model=None,
                 x2c_arch="resnet18",
                 weighted_concepts=Optional[Literal['local', 'global']],
                 
                 optimiser="adam",
                 optimiser_params={},
                 task_class_weights=None,
 
                 active_intervention_values=None,
                 inactive_intervention_values=None,
                 intervention_policy=None,
                 output_interventions=False,
                 use_concept_groups=False,
                 include_certainty=True,
                 top_k_accuracy=None,
                 
                 batch_process_fn=None,
                 concept_from_logit=False,
                 dknn_k=2,
                 dknn_tau=1,
                 dknn_method="deterministic",
                 dknn_num_samples=-1,
                 dknn_similarity="euclidean",
                 dknn_loss_type="minus_count",
                 dknn_loss_params={},
                 dknn_max_neighbours=-1,
                 dknn_hidden_layers=None,
                 x2c_only_epochs=0,
                 epoch_proto_recompute=1,
                 plateau_lr_scheduler_enable=False,
                 plateau_lr_scheduler_monitor="val_c2y_acc",
                 plateau_lr_scheduler_params={}):
        
       
        proto_model = DKNN(k=dknn_k,
                           tau=dknn_tau,
                           hard=False,
                           method=dknn_method,
                           num_samples=dknn_num_samples,
                           similarity=dknn_similarity,) 
        super().__init__(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            concept_loss_weight=concept_loss_weight,
            task_loss_weight=task_loss_weight,
            extra_dims=extra_dims,
            bool=bool,
            sigmoidal_prob=not concept_from_logit,
            sigmoidal_extra_capacity=sigmoidal_extra_capacity,
            bottleneck_nonlinear=bottleneck_nonlinear,
            output_latent=output_latent,
            x2c_model=x2c_model,
            c_extractor_arch=partial(get_backbone, x2c_arch),
            c2y_model=proto_model,
            c2y_layers=None,
            optimizer=optimiser,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_interventions=output_interventions,
            use_concept_groups=use_concept_groups,
            include_certainty=include_certainty,
            top_k_accuracy=top_k_accuracy,
        )
        
        self.dknn_k = dknn_k
        self.dknn_tau = dknn_tau
        self.dknn_method = dknn_method
        self.dknn_num_samples = dknn_num_samples
        self.dknn_simiarity = dknn_similarity
        self.dknn_loss_type = dknn_loss_type        
        self.dknn_loss_params = dknn_loss_params
        self.dknn_loss_function = dknn_loss_factory(dknn_loss_type, k=dknn_k, **dknn_loss_params)
        self.dknn_max_neighbours = dknn_max_neighbours
        self.proto_model = proto_model
        
        self.__epoch_counter = 0
        self.epoch_proto_recompute = max(1, epoch_proto_recompute)
        self.x2c_only_epochs = max(0, x2c_only_epochs)
        
        self.proto_dataloader = proto_train_dl
        
        self.batch_process_fn = batch_process_fn
        self.concept_from_logit = concept_from_logit
        self.plateau_lr_scheduler_enable = plateau_lr_scheduler_enable
        self.plateau_lr_scheduler_monitor = plateau_lr_scheduler_monitor
        self.plateau_lr_scheduler_params = plateau_lr_scheduler_params
        
        self.optimiser = optimiser
        self.optimiser_params = optimiser_params
        
        self.weighted_concepts = weighted_concepts
        if weighted_concepts == 'local':
            self.concept_weights = torch.nn.Parameter(torch.ones(n_concepts))
        elif weighted_concepts == 'global':
            self.concept_weights_model = torch.nn.Sequential(
                torch.nn.Linear(n_concepts, n_concepts),
                torch.nn.SELU()
            )
        
        # construct hidden layers
        if dknn_hidden_layers is not None:
            layers = [torch.nn.Linear(self.n_concepts, dknn_hidden_layers[0])]
            for i in range(1, len(dknn_hidden_layers)):
                layers.append(torch.nn.BatchNorm1d(dknn_hidden_layers[i-1]))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Linear(dknn_hidden_layers[i-1], dknn_hidden_layers[i]))
            self.pre_proto_model = torch.nn.Sequential(*layers)
        else:
            self.pre_proto_model = None
    
    def _unpack_batch(self, batch):
        if self.batch_process_fn is not None:
            return self.batch_process_fn(batch)
        return super()._unpack_batch(batch)
    
    def prepare_prototypes(self):
        print("Preparing prototypes")
        # compute a cache of concept activations as prototypes
        list_c_activations = []
        list_cls_labels = []
        
        for batch in tqdm(self.proto_dataloader):
            x, y = self._unpack_batch(batch)[:2]
            
            with torch.no_grad():
                c_pred, c_sem = self._run_x2c(x.to(self.device), None)
                if self.concept_from_logit:
                    c = c_pred
                else:
                    c = c_sem
                list_c_activations.append(c)
                list_cls_labels.append(y.to(self.device))
                
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
    
    def configure_optimizers(self):
        objects = {}
        OPTIM_CLASS = get_optimiser(self.optimiser)
        objects['optimizer'] = OPTIM_CLASS(self.parameters(), **self.optimiser_params)
        if self.plateau_lr_scheduler_enable:
            objects['lr_scheduler'] = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(objects['optimizer'],
                                                                        **self.plateau_lr_scheduler_params),
                "monitor": self.plateau_lr_scheduler_monitor,
                "strict": False,
            }
        return objects
    
    def on_train_epoch_end(self) -> None:
        self.__epoch_counter += 1
        if (self.__epoch_counter % self.epoch_proto_recompute) == 0:
            self.clear_prototypes()
            
    def on_train_end(self) -> None:
        self.prepare_prototypes()
            
    def _if_x2c_only(self):
            return self.__epoch_counter < self.x2c_only_epochs
        
    def _run_c2y(self, c_pred, batch_idx, train=True):
        if self._if_x2c_only():
            return None
        
        if not self.has_prototypes():
            self.prepare_prototypes()
        
        if train:
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
        
        if self.pre_proto_model is not None:
            c_pred = self.pre_proto_model(c_pred)
            proto_x = self.pre_proto_model(proto_x)
        
        if self.weighted_concepts == 'local':
            concept_weights = self.concept_weights
        elif self.weighted_concepts == 'global':
            concept_weights = self.concept_weights_model(c_pred)
        elif self.weighted_concepts is None:
            concept_weights = None        
        
        pred_y = self.proto_model(c_pred, proto_x, concept_weights=concept_weights)
        return pred_y, proto_y
    
    def dknn_loss(self, query_y, scores, neighbour_y):
        if self._if_x2c_only():
            return 0
        else:
            # query_y_one_hot = F.one_hot(query_y.long(), self.n_tasks)  # (B, C)
            # neighbour_y_one_hot = F.one_hot(neighbour_y.long(), self.n_tasks)  # (N, C)
            # correct = (query_y_one_hot.unsqueeze(1) * neighbour_y_one_hot.unsqueeze(0)).sum(-1).float() # (B, N)
            
            correct = (query_y.long().unsqueeze(1) == neighbour_y.long().unsqueeze(0)).float()
            loss = self.dknn_loss_function(scores, correct)
            return loss
        
    def _cal_loss(self, c, c_sem, c_logits, y, y_outputs,
                  competencies, prev_interventions):
        if self.task_loss_weight != 0 and not self._if_x2c_only():
            task_loss = self.dknn_loss(y, *y_outputs)
            task_loss_scalar = task_loss.detach()
        else:
            task_loss = 0
            task_loss_scalar = 0
    
        if self.concept_loss_weight != 0:
            # We separate this so that we are allowed to
            # use arbitrary activations (i.e., not necessarily in [0, 1])
            # whenever no concept supervision is provided
            # Will only compute the concept loss for concepts whose certainty
            # values are fully given
            if self.include_certainty:
                concept_loss = self.loss_concept(c_sem, c.float())
                concept_loss_scalar = concept_loss.detach()
            else:
                c_sem_used = torch.where(
                    torch.logical_or(c == 0, c == 1),
                    c_sem,
                    c,
                ) # This forces zero loss when c is uncertain
                concept_loss = self.loss_concept(c_sem_used, c)
                concept_loss_scalar = concept_loss.detach()
            loss = self.concept_loss_weight * concept_loss + self.task_loss_weight * task_loss
        else:
            loss = task_loss
            concept_loss_scalar = 0.0
        
        return loss, concept_loss_scalar, task_loss_scalar
    
    def _cal_metrics(self, c_sem, y_outputs, c, y, loss, concept_loss_scalar, task_loss_scalar):
        c_accuracy, c_auc, c_f1 = compute_concept_accuracy(c_sem, c)
        
        result = {
            "x2c_acc": c_accuracy,
            "x2c_auc": c_auc,
            "x2c_f1": c_f1,
            "x2c_loss": concept_loss_scalar,
            "loss": loss.detach(),
        }
        
        if not self._if_x2c_only():
            scores, neighbour_y = y_outputs
            dknn_accuracies = dknn_results_analysis(scores, neighbour_y, y, self.dknn_k)
            result.update({
                "c2y_acc": dknn_accuracies["neighbour_accuracy"],
                "c2y_acc_cls": dknn_accuracies["class_accuracy"],
                "c2y_loss": task_loss_scalar
            })
                
        return result
    
    def training_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=True)
        for name, val in result.items():
            self.log(f"train_{name}", val, on_epoch=True, prog_bar=True)
        return {
            "loss": loss,
            "log": result
        }

    def validation_step(self, batch, batch_no):
        _, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("val_" + name, val, on_epoch=True, prog_bar=True)
        result = {
            "val_" + key: val
            for key, val in result.items()
        }
        return result

    def test_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("test_" + name, val, on_epoch=True, prog_bar=True)
        return result
