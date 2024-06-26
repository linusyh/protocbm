import torch
import torch.nn.functional as F
from torch import Tensor


class NeuralSort(torch.nn.Module):

    def __init__(self, tau=1.0, hard=False):
        super(NeuralSort, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        scores = scores.unsqueeze(-1)
        bsize = scores.size()[0]
        dim = scores.size()[1]
        device = scores.device
        one = torch.FloatTensor(dim, 1).fill_(1).to(device)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        B = torch.matmul(A_scores, torch.matmul(one, torch.transpose(one, 0, 1)))
        scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)).type(torch.FloatTensor).to(device)
        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C - B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)

        if self.hard:
            P = torch.zeros_like(P_hat, device=device)
            b_idx = torch.arange(bsize).repeat([1, dim]).view(dim, bsize).transpose(
                dim0=1, dim1=0).flatten().type(torch.LongTensor).to(device)
            r_idx = torch.arange(dim).repeat([bsize, 1]).flatten().type(torch.LongTensor).to(device)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat
    

class FasterNeuralSort(torch.nn.Module):
    def __init__(self, 
                 k: int,
                 tau: float = 1.0, 
                 hard: bool = False):
        super(FasterNeuralSort, self).__init__()
        self.k = k
        self.hard = hard
        self.tau = tau
    
    def forward(self, scores):
        bsize, dim = scores.size()
        scores = torch.unsqueeze(scores, -1)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        B = torch.sum(A_scores, dim=-1, keepdim=True)
        scaling = (dim + 1 - 2 * (torch.arange(self.k) + 1)).type(torch.FloatTensor).to(scores.device)
        C = torch.matmul(scores, scaling.unsqueeze(0))
        P_max = (C - B).permute(0, 2, 1)
        P_hat = F.softmax(P_max / self.tau, dim=-1)
        return P_hat