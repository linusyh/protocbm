from cem.models.cbm import *
from protocbm.model import *
from protocbm.dknn.dknn_layer import *

import sys
from tqdm import tqdm
from torch.utils.data import DataLoader


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
                 sigmoidal_prob=True,
                 sigmoidal_extra_capacity=True,
                 bottleneck_nonlinear=None,
                 output_latent=False,
 
                 x2c_model=None,
                 c_extractor_arch=utils.wrap_pretrained_model(resnet50),
 
                 optimizer="adam",
                 momentum=0.9,
                 learning_rate=0.01,
                 weight_decay=4e-05,
                 weight_loss=None,
                 task_class_weights=None,
 
                 active_intervention_values=None,
                 inactive_intervention_values=None,
                 intervention_policy=None,
                 output_interventions=False,
                 use_concept_groups=False,
                 include_certainty=True,
                 top_k_accuracy=None,
                 
                 batch_process_fn=None,
                 dknn_k=2,
                 dknn_tau=1,
                 dknn_method="deterministic",
                 dknn_num_samples=-1,
                 dknn_similarity="euclidean",
                 dknn_loss_type="minus_count",
                 dknn_max_neighbours=-1,
                 x2c_only_epochs=0,
                 epoch_proto_recompute=1,
                 plateau_lr_scheduler_enable=False,
                 plateau_lr_scheduler_monitor="val_c2y_acc",
                 plateau_lr_scheduler_mode="max",
                 plateau_lr_scheduler_patience=10,
                 plateau_lr_scheduler_factor=0.1,
                 plateau_lr_scheduler_min_lr=1e-6,
                 plateau_lr_scheduler_threshold=0.01,
                 plateau_lr_scheduler_cooldown=0):
        
       
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
            sigmoidal_prob=sigmoidal_prob,
            sigmoidal_extra_capacity=sigmoidal_extra_capacity,
            bottleneck_nonlinear=bottleneck_nonlinear,
            output_latent=output_latent,
            x2c_model=x2c_model,
            c_extractor_arch=c_extractor_arch,
            c2y_model=proto_model,
            c2y_layers=None,
            
            optimizer=optimizer,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_loss=weight_loss,
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
        self.dknn_loss_function = dknn_loss_factory(dknn_loss_type)
        self.dknn_max_neighbours = dknn_max_neighbours
        self.proto_model = proto_model
        
        self.__epoch_counter = 0
        self.epoch_proto_recompute = max(1, epoch_proto_recompute)
        self.x2c_only_epochs = max(0, x2c_only_epochs)
        
        self.proto_dataloader = proto_train_dl
        
        self.batch_process_fn = batch_process_fn
        self.plateau_lr_scheduler_enable = plateau_lr_scheduler_enable
        self.plateau_lr_scheduler_monitor = plateau_lr_scheduler_monitor
        self.plateau_lr_scheduler_mode = plateau_lr_scheduler_mode
        self.plateau_lr_scheduler_patience = plateau_lr_scheduler_patience
        self.plateau_lr_scheduler_factor = plateau_lr_scheduler_factor
        self.plateau_lr_scheduler_min_lr = plateau_lr_scheduler_min_lr
        self.plateau_lr_scheduler_threshold = plateau_lr_scheduler_threshold
        self.plateau_lr_scheduler_cooldown = plateau_lr_scheduler_cooldown
    
    
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
                list_c_activations.append(c_pred)
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
        objects['optimizer'] = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
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
    
    def on_train_epoch_end(self) -> None:
        self.__epoch_counter += 1
        if (self.__epoch_counter % self.epoch_proto_recompute) == 0:
            self.clear_prototypes()
            
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
            
        pred_y = self.proto_model(c_pred, proto_x)
        return pred_y, proto_y
    
    def dknn_loss(self, query_y, scores, neighbour_y):
        if self._if_x2c_only():
            return 0
        else:
            query_y_one_hot = F.one_hot(query_y.long(), self.n_tasks)  # (B, C)
            neighbour_y_one_hot = F.one_hot(neighbour_y.long(), self.n_tasks)  # (N, C)
            correct = (query_y_one_hot.unsqueeze(1) * neighbour_y_one_hot.unsqueeze(0)).sum(-1).float() # (B, N)
            
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
                concept_loss = self.loss_concept(c_sem, c)
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