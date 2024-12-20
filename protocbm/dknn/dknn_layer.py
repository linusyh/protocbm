'''
"Differentiable k-nearest neighbors" layer.

Given a set of M queries and a set of N neighbors,
returns an M x N matrix whose rows sum to k, indicating to what degree
a certain neighbor is one of the k nearest neighbors to the query.
At the limit of tau = 0, each entry is a binary value representing
whether each neighbor is actually one of the k closest to each query.
'''
from typing import *
import logging
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np

from protocbm.dknn.neuralsort import FasterNeuralSort, NeuralSort
from protocbm.dknn.pl import PL


class DKNNLossSparseWeighted(torch.nn.Module):
    def __init__(self, 
                 positive_weight=1.0, 
                 negative_weight=1.0,
                 num_neighbour_agnostic: bool = True,
                 **kwargs):
        super(DKNNLossSparseWeighted, self).__init__()
        logging.info(f"DKNNLossSparseWeighted initialised with pos={positive_weight} neg={negative_weight}")
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.num_neighbour_agnostic = num_neighbour_agnostic
    
    def forward(self, dknn_output, truth):
        weight_factor = self.positive_weight + self.negative_weight
        is_positive = truth.sum()
        amplification = torch.where(truth==1, 
                                    self.positive_weight * -1 / is_positive,
                                    self.negative_weight / (dknn_output.nelement()-is_positive))
        loss = (dknn_output * amplification).sum() / weight_factor
        if self.num_neighbour_agnostic:
            loss *= dknn_output.shape[1]  
        return loss
    

class DKNNLoss(torch.nn.Module):
    def __init__(self, k ,**kwargs):
        self.k = k
        super(DKNNLoss, self).__init__()
    
    def forward(self, dknn_output, truth):
        return -(dknn_output * truth).sum() / self.k


class DKNNInverseLoss(torch.nn.Module):
    def __init__(self,**kwargs):
        super(DKNNInverseLoss, self).__init__()
    
    def forward(self, dknn_output, truth):
        return (1/(dknn_output * truth)).sum()
    
class DKNNBCEwithLogitLoss(_Loss):
    def __init__(self, k: float, reduction = 'mean', **kwargs):
        super(DKNNBCEwithLogitLoss, self).__init__(reduction=reduction)
        self.k = k
    
    def forward(self, dknn_output, truth):
        return F.binary_cross_entropy_with_logits(dknn_output/self.k, truth, reduction=self.reduction)

class DKNNMSELoss(_Loss):
    def __init__(self, k: float, reduction = 'mean', **kwargs):
        super(DKNNMSELoss, self).__init__(reduction=reduction)
        self.k = k
    
    def forward(self, dknn_output, truth):
        return F.mse_loss(dknn_output/self.k, truth, reduction=self.reduction)

DKNN_LOSS_LOOKUP = {
    "dknn_original": DKNNLoss,
    "minus_count": DKNNLoss,
    "inverse_count": DKNNInverseLoss,
    "sparse_weighted": DKNNLossSparseWeighted,
    "bce": DKNNBCEwithLogitLoss,
    "mse": DKNNMSELoss
}

def dknn_loss_factory(loss_str: str, k: float, **kwargs) -> torch.nn.Module:
    loss_str = loss_str.strip().lower()
    
    if loss_str in DKNN_LOSS_LOOKUP.keys():
        return DKNN_LOSS_LOOKUP[loss_str](k=k, **kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_str}")


class DKNN(torch.nn.Module):

    def __init__(self, k, tau=1.0, hard=False, method='deterministic', num_samples=-1, similarity='euclidean'):
        super(DKNN, self).__init__()
        self.k = k
        self.soft_sort = FasterNeuralSort(k=k, tau=tau, hard=hard)
        self.method = method
        self.num_samples = num_samples
        self.similarity = similarity

    # query: M x p
    # neighbors: N x p
    # concept_weights: M x p
    # returns:
    def forward(self, query, neighbors, tau=1.0, concept_weights=None):      
        if self.similarity == 'euclidean':
            diffs = (query.unsqueeze(1) - neighbors.unsqueeze(0)) # M x N x p
            squared_diffs = diffs**2
            if concept_weights is not None:
                if concept_weights.ndim == 1 and len(concept_weights) == query.shape[-1]:
                    concept_weights = concept_weights[None, None, :]
                elif concept_weights.ndim == 2 and concept_weights.shape[-1] == query.shape[-1]:
                    concept_weights = concept_weights.unsqeeze(1)
                else:
                    raise ValueError(f'Unsupported concept_weights shape: {concept_weights.shape}')
                squared_diffs *= concept_weights
            l2_norms = squared_diffs.sum(2) # M x N
            norms = l2_norms
            scores = -norms  # B * N
            
        elif self.similarity == 'cosine':
            # scores = F.cosine_similarity(query.unsqueeze(1), neighbors.unsqueeze(0), dim=2) - 1
            scores = F.cosine_similarity(query.unsqueeze(1), neighbors.unsqueeze(0), dim=2)
        else:
            raise ValueError('Unknown similarity for DKNN: {}'.format(self.similarity))

        if self.method == 'deterministic':
            P_hat = self.soft_sort(scores)  # B*N*N
            top_k = P_hat[:, :self.k, :].sum(1)  # B*N
            return top_k
        if self.method == 'stochastic':
            if scores.min().item() <= 0:
                scores = scores - scores.min() + 1e-08
            # print("Scores: " + str((scores.min().item(), scores.max().item())))
            pl_s = PL(scores, tau, hard=False)
            P_hat = pl_s.sample((self.num_samples, ))
            top_k = P_hat[:, :, :self.k, :].sum(2)
            return top_k


def dknn_loss(dknn_layer, query, neighbors, query_label, neighbor_labels, method='deterministic'):
    # query: batch_size x p
    # neighbors: 10k x p
    # query_labels: batch_size x [10] one-hot
    # neighbor_labels: n x [10] one-hot
    if method == 'deterministic':
        # top_k_ness is the sum of top-k row of permutation matrix
        top_k_ness = dknn_layer(query, neighbors)  # (B*512, N*512) => (B, N)
        correct = (query_label.unsqueeze(1) * neighbor_labels.unsqueeze(0)).sum(-1) # (B, N)
        correct_in_top_k = (correct * top_k_ness).sum(-1)  # [B]
        loss = -correct_in_top_k
        # loss = 1 / correct_in_top_k
        return loss.mean()
    elif method == 'stochastic':
        top_k_ness = dknn_layer(query, neighbors)
        correct = (query_label.unsqueeze(1) * neighbor_labels.unsqueeze(0)).sum(-1)
        correct_in_top_k = (correct.unsqueeze(0) * top_k_ness).sum(-1)
        loss = -correct_in_top_k
        return loss.mean()
    else:
        raise ValueError(method)
    
    
def dknn_loss_warp_one_hot(dknn_layer, query, neighbors, query_label, neighbor_labels, method='deterministic', num_classes=-1):
    if num_classes == -1:
        num_classes = max(torch.max(query_label).item(), torch.max(neighbor_labels).item()) + 1
    
    return dknn_loss(dknn_layer, query, neighbors,
                     query_label=F.one_hot(query_label, num_classes),
                     neighbor_labels=F.one_hot(neighbor_labels, num_classes),
                     method=method
    )


def dknn_cal_neighbour_accuracy(dknn_output, neighbour_label, target_label, k):
    with torch.no_grad():
        total_neighbours = 0
        correct_neighbours = 0
        
        for pred, label in zip(dknn_output, target_label):
            _, top_idx = torch.topk(pred, k=k)
            neighbours = neighbour_label[top_idx]
            total_neighbours += k
            correct_neighbours += (neighbours == label).sum().item()
        
        return correct_neighbours / total_neighbours


def dknn_cal_class_accuracy(dknn_output, neighbour_label, target_label, k):
    dknn_output = dknn_output.detach().cpu().numpy()
    neighbour_label = neighbour_label.detach().cpu().numpy()
    target_label = target_label.detach().cpu().numpy()
    
    accuracies = []
    
    for pred, label in zip(dknn_output, target_label):
        top_k_idx = np.argpartition(pred, -k)[-k:]
        top_k_labels = neighbour_label[top_k_idx]
        uniques, counts = np.unique(top_k_labels, return_counts=True)
        if label in uniques:
            idx = np.where(uniques == label)[0]
            label_count = counts[idx]
            if counts.max() == label_count:
                divisor = (counts==label_count).sum()
                if divisor > 0:
                    accuracies.append(1/divisor)
    
    if len(accuracies) == 0:
        return 0
    return np.sum(accuracies) / len(dknn_output)
    
def dknn_results_analysis(dknn_output, neighbour_label, target_label, k):
    class_acc = dknn_cal_class_accuracy(dknn_output, neighbour_label, target_label, k)
    neighbour_acc = dknn_cal_neighbour_accuracy(dknn_output, neighbour_label, target_label, k)
    return {
        "class_accuracy": class_acc,
        "neighbour_accuracy": neighbour_acc
    }
