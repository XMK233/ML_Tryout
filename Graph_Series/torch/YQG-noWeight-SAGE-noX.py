#!/usr/bin/env python
# coding: utf-8

# https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70
# 
# 这里，我们尝试，不要初始特征，让初始特征变成embedding。

# In[1]:


import os.path as osp
import torch
from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# In[2]:


import pandas as pd
import numpy as np


# In[ ]:





# In[3]:


# import torch_scatter, torch_sparse, torch_cluster, torch_spline_conv, torch_geometric


# In[4]:


# pip install torch==2.1.1 torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2 torch-geometric==2.4.0 --target=./


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


df_ori = pd.read_csv("sample.csv").head(10000)

node_order = sorted(list(set(df_ori.company_id.to_list() + df_ori.outcompany_id.to_list())))

mapping = {
    ci: idx for idx, ci in enumerate(node_order)
}
for col in df_ori:
    df_ori[col] = df_ori[col].map(mapping)


# In[6]:


from torch_geometric.data import Data

data = Data(
    num_nodes = len(mapping),
    edge_index=torch.tensor(
        df_ori.T.to_numpy(), 
        dtype = torch.long
    )
)


# In[7]:


data.is_directed()


# In[8]:


n_nodes = data.num_nodes
hidden_channels = 64


# In[9]:


transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,    
)
train_data, val_data, test_data = transform(data)


# In[10]:


from torch_geometric.loader import LinkNeighborLoader
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[
        -1,# 10, 
        -1, 5
    ],
    neg_sampling_ratio=2.0,
    batch_size=128,
    shuffle=True,
)


# In[11]:


from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 128)
        self.conv2 = SAGEConv(128, out_channels)
    def forward(self, x, edge_index) :
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class Classifier(torch.nn.Module):
    def forward(self, x_from, x_to,):
        return (x_from * x_to).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.emb = torch.nn.Embedding(n_nodes, in_channels)
        self.gnn = GNN(in_channels, out_channels)
        self.classifier = Classifier()
    def forward(self, data):
        x_out = self.gnn(
            self.emb(data.n_id), 
            data.edge_label_index
        )
        pred = self.classifier(
            x_out[data.edge_label_index[0]], ## 边的起始点。
            x_out[data.edge_label_index[-1]] ## 边的终结点。
        )
        return pred
        
model = Model(in_channels=hidden_channels, out_channels=64)

weights = model.emb.weight.detach().numpy()
pd.DataFrame(weights, columns = [f"col_{i}" for i in range(weights.shape[1])])


# In[12]:


# !pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu111.html
# import torch_geometric
# torch_geometric.typing.WITH_TORCH_SPARSE


# In[13]:


import tqdm
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, 10):
    total_loss = total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):        
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = model(sampled_data)
        ground_truth = sampled_data.edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    
#     break
    
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
    


# In[14]:


# !pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric


# In[ ]:





# In[15]:


test_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[10, 5],
    neg_sampling_ratio=2.0,
    batch_size=128,
    shuffle=True,
)


# In[16]:


from sklearn.metrics import roc_auc_score
preds = []
ground_truths = []
for sampled_data in tqdm.tqdm(test_loader):
    with torch.no_grad():
        sampled_data.to(device)
        preds.append(model(sampled_data))
        ground_truths.append(sampled_data.edge_label)
pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
auc = roc_auc_score(ground_truth, pred)
print()
print(f"Validation AUC: {auc:.4f}")


# In[17]:


model.emb


# In[18]:


weights = model.emb.weight.detach().numpy()
df_rst = pd.DataFrame(weights, columns = [f"col_{i}" for i in range(weights.shape[1])])
df_rst["company_id"] = node_order
df_rst.to_csv("embedding.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




