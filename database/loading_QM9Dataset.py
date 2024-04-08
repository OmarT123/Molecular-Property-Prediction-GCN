import dgl
from dgl.data import QM9Dataset
import numpy as np
import math

print(dgl.__version__)

# Load QM9Edge dataset
dataset = QM9Dataset(['mu','alpha','homo']) #SHOULD I USE MORE THAN ONE PROP OR JUSST LOAD ONE????
size = math.ceil(len(dataset)/10_000)
# print(size)



adj = [[] for _ in range(size)]
feat = [[] for _ in range(size)]
count = 0
idx = 0

for graph, label in dataset:
    n = graph.ndata['R'].shape[0]

    # Pad the tensor to shape (50, 50)
    padded_adj = np.pad(graph.ndata['R'], ((0, 50 - n), (0, 47)), mode='constant')
    padded_feat = np.pad()
    if (count != 0 and count % 10_000 == 0):
        print(np.asarray(adj[idx]).shape)
        idx += 1  
    count += 1
    adj[idx].append(padded_adj)
    feat[idx].append(padded_feat)

# print(adj)
print(np.asarray(adj).shape)
# print(adj[0])
# print(adj[0].shape)
