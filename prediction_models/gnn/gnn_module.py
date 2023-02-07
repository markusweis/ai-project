

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg




class GNNModel(torch.nn.Module):
    def __init__(self, num_nodes, input_dim, emb_features, num_gnn_layers, fc_features, num_fc_layers, dropout):
        super(GNNModel, self).__init__()
        conv_model = pyg.nn.DenseSAGEConv

        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, emb_features))
        self.dropout = dropout
        self.num_layers = num_gnn_layers

        assert (self.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(self.num_layers - 1):
            self.convs.append(conv_model(emb_features, emb_features))

        # post-message-passing
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(emb_features, fc_features))
        for _ in range(num_fc_layers - 2):
            self.lins.append(nn.Linear(fc_features, fc_features))
        self.lins.append(nn.Linear(fc_features, 1))

    def forward(self, x: torch.tensor):
        """
            :param x: tensor of one-hot encoded node features (N, 2) 
        """

        # x = x.type(torch.float)
        # edge_index = edge_index.type(torch.int64)
        
        # build fully connected graph
        adj = torch.ones(x.size(dim=1))

        # gnn pass
        for i in range(self.num_layers):
            x = self.convs[i](x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        
        # Take embeddings and for every edge pass FC layers 
        adj = torch.triu(adj, diagonal=1)
        edge_index = adj.nonzero().t().contiguous()

        edge_embeddings = x[edge_index[0]] * x[edge_index[1]] # (E, d)

        # FC Layers 
        for lin in self.lins[:-1]:
            edge_embeddings = lin(edge_embeddings)
            edge_embeddings = F.relu(edge_embeddings)
            edge_embeddings = F.dropout(edge_embeddings, p=self.dropout, training=self.training)
        edge_embeddings = self.lins[-1](edge_embeddings) # (E, )

        return torch.sigmoid(edge_embeddings)







