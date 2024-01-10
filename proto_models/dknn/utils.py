import torch


def dknn_predict(query, neighbors, neighbor_labels, k):
    '''
    query: p
    neighbors: n x p
    neighbor_labels: n x num_classes
    '''
    query, neighbors = query.detach(), neighbors.detach()
    diffs = (query.unsqueeze(1) - neighbors.unsqueeze(0))
    # squared_diffs = diffs**2
    # norms = squared_diffs.sum(-1)
    norms = torch.norm(diffs, p=2, dim=-1)
    indices = torch.argsort(norms, dim=-1).to(neighbor_labels.device)
    labels = neighbor_labels[indices[:, :k]]  # n x k x num_classes
    label_counts = labels.sum(dim=1)  # n x num_classes
    prediction = torch.argmax(label_counts, dim=1)  # n

    return prediction, indices[:, :k]


# def dknn_predict(query, neighbors, neighbor_labels, k):
#     '''
#     query: p
#     neighbors: n x p
#     neighbor_labels: n (int)
#     '''
#     query, neighbors = query.detach(), neighbors.detach()
#     diffs = (query.unsqueeze(1) - neighbors.unsqueeze(0))
#     # squared_diffs = diffs**2
#     # norms = squared_diffs.sum(-1)
#     norms = torch.norm(diffs, p=2, dim=-1)
#     indices = torch.argsort(norms, dim=-1).to(neighbor_labels.device)
#     labels = neighbor_labels[indices[:, :k]]
#     # Compute the mode of each row (i.e., the majority label)
#     # `labels` is a tensor of shape (batch_size, k)
#     prediction, _ = torch.mode(labels, dim=1)
#     # Convert the tensor to a long tensor
#     return prediction.long()