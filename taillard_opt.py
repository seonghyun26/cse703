import numpy as np
import torch
from torch_geometric.data import Data


n_pair = [(15, 15), (20, 15), (20, 20), (30, 15), (30, 20), (50, 15), (50, 20), (100, 20)]

index = list(range(1, 81))
index.remove(62)

graph_dict = np.load('edge_opt.npy', allow_pickle=True, encoding='bytes').item()
final_temp = np.load('node_opt.npy', allow_pickle=True, encoding='bytes').item()


final_result = dict()
for i in index:
    edge_index = torch.tensor(graph_dict[i], dtype = torch.long)
    x = torch.tensor(final_temp[i], dtype = torch.float)
    data = Data(x=x, edge_index = edge_index.t().contiguous())
    final_result[i] = data


# for i in index:

#     n_job = n_pair[(i-1)//10][0]
#     n_mch = n_pair[(i-1)//10][1]

#     torch.save(final_result[i], 'jssp_opt\jssp_'+str(n_job)+'_'+str(n_mch)+'_'+str((i-1)%10)+'.pt')