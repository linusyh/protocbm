import unittest

import torch
import numpy as np
from protocbm.dknn.dknn_layer import *


class TestDKNN(unittest.TestCase):
    def test_dknn_sparse_weighted_loss_against_loop_implementation(self):
        def dknn_sparse_weighted_loss(dknn_output, truth, positive_weight, negative_weight):
            positive_count = truth.sum()
            negative_count = np.prod(truth.shape) - positive_count
            
            positive_loss = -(truth * dknn_output).sum() / positive_count
            negative_loss = ((1 - truth) * dknn_output).sum() / negative_count
            
            return positive_weight * positive_loss + negative_weight * negative_loss
        
        dknn_output = torch.rand((1, 100, 200))
        truth = torch.rand((1, 100, 200)) > 0.5
        positive_weight = np.random.rand()
        negative_weight = np.random.rand()  
        
        LossFn = dknn_loss_factory(f"sparse_weighted_{positive_weight}_{negative_weight}")
        loss1 = LossFn(dknn_output, truth).detach().cpu().item()
        loss2 = float(dknn_sparse_weighted_loss(dknn_output.detach().cpu().numpy(), truth.detach().cpu().numpy(), positive_weight, negative_weight))
        
        self.assertAlmostEqual(loss1, loss2, places=5)
        
    def test_dknn_class_accuracy(self):
        scores = torch.tensor([[0.8, 0.5, 0.4], [0.3, 0.2, 0.4], [0.1, 0.9, 0.9], [0.2, 0.3, 0.1]])
        neighbour_labels = torch.tensor([1, 2, 1])
        target_labels = torch.tensor([2, 1, 0, 0])
        
        accuracy = dknn_cal_class_accuracy(scores, neighbour_labels, target_labels, 2)
        print(accuracy)
        self.assertAlmostEqual(accuracy, 1.5/4)
        

if __name__ == "__main__":
    unittest.main
            
            
            