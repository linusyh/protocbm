from lightning import seed_everything

from torch.utils.data import Dataset
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ClusterClassLinearConcept(Dataset):
    def __init__(self, n_clusters, n_dim, n_samples_per_cluster, n_concepts, mean=0.0, std=1.0, seed=773):
        self.n_clusters = n_clusters
        self.n_dim = n_dim
        self.n_samples_per_cluster = n_samples_per_cluster
        self.n_concepts = n_concepts
        self.mean = mean
        self.std = std
        self.seed = seed
        
        self._generate_data()
        
    def _generate_data(self):
        seed_everything(self.seed)
        centroids = np.random.uniform(low=-3, high=3, size=(self.n_clusters, self.n_dim))
        print(centroids)
        concept_linear_weights = np.random.normal(size=(self.n_dim, self.n_concepts))
        print(concept_linear_weights)
        x_list = []
        c_list = []
        y_list = []
        
        for cluster_idx, centroid in enumerate(centroids):
            x = centroid + np.random.normal(loc=self.mean, scale=self.std, size=(self.n_samples_per_cluster, self.n_dim))
            c = sigmoid(x @ concept_linear_weights)
            y = cluster_idx * np.ones(self.n_samples_per_cluster, dtype=np.int32)
        
            x_list.append(x)
            c_list.append(c)
            y_list.append(y)
        
        self.x = np.concatenate(x_list, axis=0).astype(np.float32)
        self.c = np.concatenate(c_list, axis=0).astype(np.float32)
        self.y = np.concatenate(y_list, axis=0).astype(int)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.c[idx]
    
    
if __name__ == "__main__":
    N_CLUSTERS = 2
    N_DIM = 3
    N_SAMPLES_PER_CLUSTER = 10
    N_CONCEPTS = 3
    MEAN = 0.0
    STD = 1.0
    SEED = 203
    
    dataset = ClusterClassLinearConcept(N_CLUSTERS, N_DIM, N_SAMPLES_PER_CLUSTER, N_CONCEPTS, mean=MEAN, std=STD, seed=SEED)
    iterator = iter(dataset)
    
    for i in range(2):
        x, c, y = next(iterator)
        print(x)
        print(c)
        print(y)