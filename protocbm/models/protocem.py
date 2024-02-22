from cem.models.cbm import *
from protocbm.models.protocbm import ProtoCBM
from protocbm.dknn.dknn_layer import *
from .utils import get_activation_fn, get_optimiser

import sys
from torch.utils.data import DataLoader
from torchvision.models import *
from pytorch_lightning import LightningModule

class ProtoCEM(ProtoCBM):
    def __init__(self,
                 n_concepts,
                 n_tasks,
                 proto_train_dl: DataLoader,
                 x2c_model=None,
                 emb_size=16,
                 training_intervention_prob=0.25,
                 embedding_activation="leakyrelu",
                 shared_prob_gen=False,
                 concept_loss_weight=0.01,
                 task_loss_weight=1,
                 output_latent=False,
                #  extra_dims=0,
                #  bool=False,
                #  sigmoidal_prob=True,
                #  sigmoidal_extra_capacity=True,
                #  bottleneck_nonlinear=None,
                 optimiser="adam",
                 optimiser_params={},
 
                 active_intervention_values=None,
                 inactive_intervention_values=None,
                 intervention_policy=None,
                 output_interventions=False,
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
                 plateau_lr_scheduler_params={}):
    
        super().__init__(n_concepts=n_concepts,
                         n_tasks=n_tasks,
                         proto_train_dl=proto_train_dl,
                         concept_loss_weight=concept_loss_weight,
                         task_loss_weight=task_loss_weight,
                         x2c_model=torch.nn.Identity(),  # Not needed for ProtoCEM, hence filler
                         concept_from_logit=True, # Very important, logits = actual embedding in CEM
                         optimiser=optimiser,
                         optimiser_params=optimiser_params,
                         batch_process_fn=batch_process_fn,
                         dknn_k=dknn_k,
                         dknn_tau=dknn_tau,
                         dknn_method=dknn_method,
                         dknn_similarity=dknn_similarity,
                         dknn_loss_type=dknn_loss_type,
                         dknn_max_neighbours=dknn_max_neighbours,
                         x2c_only_epochs=x2c_only_epochs,
                         epoch_proto_recompute=epoch_proto_recompute,
                         plateau_lr_scheduler_enable=plateau_lr_scheduler_enable,
                         plateau_lr_scheduler_monitor=plateau_lr_scheduler_monitor,
                         plateau_lr_scheduler_params=plateau_lr_scheduler_params)
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.concept_loss_weight = concept_loss_weight
        self.task_loss_weight = task_loss_weight
        self.emb_size = emb_size
        self.training_intervention_prob = training_intervention_prob
        self.embedding_activation = embedding_activation
        self.shared_prob_gen = shared_prob_gen
        self.proto_dataloader = proto_train_dl
        self.batch_process_fn = batch_process_fn
        self.output_latent = output_latent
        
        self.include_certainty = include_certainty
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.training_intervention_prob = training_intervention_prob
        
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)
        
        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.ones(n_concepts)
        self.pre_concept_model = x2c_model
        
        
        # X2C setup
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        self.shared_prob_gen = shared_prob_gen
        self.top_k_accuracy = top_k_accuracy
        for i in range(n_concepts):
            self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            2 * emb_size,
                        ),
                        get_activation_fn(embedding_activation)()
                    ])
                )
            if self.shared_prob_gen and len(self.concept_prob_generators) == 0:
                # Then we will use one and only one probability generator which
                # will be shared among all concepts. This will force concept
                # embedding vectors to be pushed into the same latent space
                self.concept_prob_generators.append(torch.nn.Linear(
                    2 * emb_size,
                    1,
                ))
            elif not self.shared_prob_gen:
                self.concept_prob_generators.append(torch.nn.Linear(
                    2 * emb_size,
                    1,
                ))
        self.sig = torch.nn.Sigmoid()
        self.loss_concept = torch.nn.BCELoss()
        
        # DKNN Settings
        self.proto_model = DKNN(k=dknn_k,
                                tau=dknn_tau,
                                hard=False,
                                method=dknn_method,
                                num_samples=dknn_num_samples,
                                similarity=dknn_similarity) 
        self.dknn_k = dknn_k
        self.dknn_tau = dknn_tau
        self.dknn_method = dknn_method
        self.dknn_num_samples = dknn_num_samples
        self.dknn_simiarity = dknn_similarity
        self.dknn_loss_type = dknn_loss_type        
        self.dknn_loss_function = dknn_loss_factory(dknn_loss_type)
        self.dknn_max_neighbours = dknn_max_neighbours
        
        self.x2c_only_epochs = x2c_only_epochs
        self.epoch_proto_recompute = epoch_proto_recompute
        
    def prepare_prototypes(self):
        print("Preparing prototypes")
        # compute a cache of concept activations as prototypes
        list_c_activations = []
        list_cls_labels = []
        
        for batch in tqdm(self.proto_dataloader):
            x, y = self._unpack_batch(batch)[:2]
            
            with torch.no_grad():
                c, _ = self._run_x2c(x.to(self.device), None)
                list_c_activations.append(c)
                list_cls_labels.append(y.to(self.device))
            sys.stdout.flush()
            sys.stderr.flush()
                
        self.register_buffer('proto_concepts', torch.concat(list_c_activations))
        self.register_buffer('proto_classes', torch.concat(list_cls_labels))
        
    def _after_interventions(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        intervention_idxs=None,
        c_true=None,
        train=False,
        competencies=None,
    ):
        if train and (self.training_intervention_prob != 0) and (
            (c_true is not None) and
            (intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(
                self.ones * self.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )
        if (c_true is None) or (intervention_idxs is None):
            return prob, intervention_idxs
        intervention_idxs = intervention_idxs.type(torch.FloatTensor)
        intervention_idxs = intervention_idxs.to(prob.device)
        if not self.include_certainty:
            c_true = torch.where(
                torch.logical_or(c_true == 0, c_true == 1),
                c_true,
                prob,
            )
        return prob * (1 - intervention_idxs) + intervention_idxs * c_true, intervention_idxs

    
    def _run_x2c(self, x, latent):
        if latent is None:
            pre_c = self.pre_concept_model(x)
            contexts = []
            c_sem = []

            # First predict all the concept probabilities
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                context = context_gen(pre_c)
                prob = prob_gen(context)
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(self.sig(prob))
            c_sem = torch.cat(c_sem, axis=-1)
            contexts = torch.cat(contexts, axis=1)
        else:
            contexts, c_sem = latent
            
        c_emb = (
            contexts[:, :, :self.emb_size] * torch.unsqueeze(c_sem, dim=-1) +
            contexts[:, :, self.emb_size:] * (1 - torch.unsqueeze(c_sem, dim=-1))
        )
        c_emb = c_emb.view((-1, self.emb_size * self.n_concepts))

        return c_emb, c_sem
    
    def _forward(
        self,
        x,
        intervention_idxs=None,
        c=None,
        y=None,
        train=False,
        latent=None,
        competencies=None,
        prev_interventions=None,
        output_embeddings=False,
        output_latent=None,
        output_interventions=None,
        batch_idx=None,
    ):
        output_interventions = (
            output_interventions if output_interventions is not None
            else self.output_interventions
        )
        output_latent = (
            output_latent if output_latent is not None
            else self.output_latent
        )
        if latent is None:
            pre_c = self.pre_concept_model(x)
            contexts = []
            c_sem = []

            # First predict all the concept probabilities
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                context = context_gen(pre_c)
                prob = prob_gen(context)
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(self.sig(prob))
            c_sem = torch.cat(c_sem, axis=-1)
            contexts = torch.cat(contexts, axis=1)
            latent = contexts, c_sem
        else:
            contexts, c_sem = latent

        # Now include any interventions that we may want to perform!
        if (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        ):
            horizon = self.intervention_policy.num_groups_intervened
            if hasattr(self.intervention_policy, "horizon"):
                horizon = self.intervention_policy.horizon
            prior_distribution = self._prior_int_distribution(
                prob=c_sem,
                pos_embeddings=contexts[:, :, :self.emb_size],
                neg_embeddings=contexts[:, :, self.emb_size:],
                competencies=competencies,
                prev_interventions=prev_interventions,
                c=c,
                train=train,
                horizon=horizon,
            )
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=prior_distribution,
            )

        else:
            c_int = c
        if not train:
            intervention_idxs = self._standardize_indices(
                intervention_idxs=intervention_idxs,
                batch_size=x.shape[0],
            )

        # Then, time to do the mixing between the positive and the
        # negative embeddings
        probs, intervention_idxs = self._after_interventions(
            c_sem,
            pos_embeddings=contexts[:, :, :self.emb_size],
            neg_embeddings=contexts[:, :, self.emb_size:],
            intervention_idxs=intervention_idxs,
            c_true=c_int,
            train=train,
            competencies=competencies,
        )
        
        # Then time to mix!
        c_pred = (
            contexts[:, :, :self.emb_size] * torch.unsqueeze(probs, dim=-1) +
            contexts[:, :, self.emb_size:] * (1 - torch.unsqueeze(probs, dim=-1))
        )
        c_pred = c_pred.view((-1, self.emb_size * self.n_concepts))
        y = self._run_c2y(c_pred, batch_idx, train)
        tail_results = []
        if output_interventions:
            if (
                (intervention_idxs is not None) and
                isinstance(intervention_idxs, np.ndarray)
            ):
                intervention_idxs = torch.FloatTensor(
                    intervention_idxs
                ).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            tail_results.append(latent)
        if output_embeddings:
            tail_results.append(contexts[:, :, :self.emb_size])
            tail_results.append(contexts[:, :, self.emb_size:])
        return tuple([c_sem, c_pred, y] + tail_results)

           
                                       