'''
"Differentiable k-nearest neighbors" layer.

Given a set of M queries and a set of N neighbors,
returns an M x N matrix whose rows sum to k, indicating to what degree
a certain neighbor is one of the k nearest neighbors to the query.
At the limit of tau = 0, each entry is a binary value representing
whether each neighbor is actually one of the k closest to each query.
'''

import torch
import torch.nn.functional as F
import lightning as L

from proto_models.dknn.neuralsort import NeuralSort
from proto_models.dknn.pl import PL


class DKNNClassifier(L.LightningModule):    
    def __init__(self,
                 num_classes: int,
                 prototype_feature: torch.Tensor,
                 prototype_label: torch.Tensor,
                 optim_cls=torch.optim.Adam,
                 optim_params={},
                 *args,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.prototype_feature = prototype_feature
        self.prototype_label = prototype_label
        self.optim_cls = optim_cls
        self.optim_params = optim_params
        
        self.dknn = DKNN(*args, **kwargs)
    
    def has_prototype(self):
        return self.prototype_feature is not None and self.prototype_label is not None
    
    def configure_optimizers(self):
        optimiser = self.optim_cls(self.dknn.parameters(), **self.optim_params)
        return dict(optimizer=optimiser)
    
    def on_train_epoch_start(self):
        if self.has_prototype():
            self.prototype_feature = self.prototype_feature.to(self.device)
            self.prototype_label = self.prototype_label.to(self.device)
    
    def training_step(self, batch, batch_idx):
        x, truth_y = batch
        
        loss = dknn_loss_warp_one_hot(self.dknn,
                                      query=x,
                                      neighbors=self.prototype_feature,
                                      query_label=truth_y,
                                      neighbor_labels=self.prototype_label,
                                      method=self.dknn.method)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss


class DKNN(torch.nn.Module):

    def __init__(self, k, tau=1.0, hard=False, method='deterministic', num_samples=-1, similarity='euclidean'):
        super(DKNN, self).__init__()
        self.k = k
        self.soft_sort = NeuralSort(tau=tau, hard=hard)
        self.method = method
        self.num_samples = num_samples
        self.similarity = similarity

    # query: M x p
    # neighbors: N x p
    #
    # returns:
    def forward(self, query, neighbors, tau=1.0):
        if self.similarity == 'euclidean':
            diffs = (query.unsqueeze(1) - neighbors.unsqueeze(0))
            squared_diffs = diffs**2
            l2_norms = squared_diffs.sum(2)
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
    
    
def dknn_results_analysis(dknn_output, neighbour_label, target_label, k):
    with torch.no_grad():
        total_neighbours = 0
        correct_neighbours = 0
        
        for pred, label in zip(dknn_output, target_label):
            _, top_idx = torch.topk(pred, k=k)
            neighbours = neighbour_label[top_idx]
            total_neighbours += k
            correct_neighbours += (neighbours == label).sum().item()
        
        return correct_neighbours / total_neighbours