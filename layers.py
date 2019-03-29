import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    def __init__(self, in_features, out_feature_list, b_dim, dropout):
        super(GraphConvolution, self).__init__()
        # 5 [128, 64] 5 0.0
        self.in_features = in_features
        self.out_feature_list = out_feature_list

        self.linear1 = nn.Linear(in_features, out_feature_list[0]) # 5x128
        self.linear2 = nn.Linear(out_feature_list[0], out_feature_list[1]) # 128x64

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, adj, activation=None):
        # input : 16x9x5
        # adj : 16x4x9x9
        hidden = torch.stack([self.linear1(input) for _ in range(adj.size(1))], 1) # 16x4x9x128
        hidden = torch.einsum('bijk,bikl->bijl', (adj, hidden)) # 16x4x9x128
        hidden = torch.sum(hidden, 1) + self.linear1(input) # 16x9x128
        hidden = activation(hidden) if activation is not None else hidden # 16x9x128
        hidden = self.dropout(hidden)

        output = torch.stack([self.linear2(hidden) for _ in range(adj.size(1))], 1) # 16x4x9x64
        output = torch.einsum('bijk,bikl->bijl', (adj, output)) # 16x4x9x64
        output = torch.sum(output, 1) + self.linear2(hidden) # 16x9x64
        output = activation(output) if activation is not None else output # 16x9x64
        output = self.dropout(output)
        return output


class GraphAggregation(Module):

    def __init__(self, in_features, out_features, b_dim, dropout):
        super(GraphAggregation, self).__init__()
        # 64 128 5 0.0
        self.sigmoid_linear = nn.Sequential(nn.Linear(in_features+b_dim, out_features),
                                            nn.Sigmoid()) # 69x128
        self.tanh_linear = nn.Sequential(nn.Linear(in_features+b_dim, out_features),
                                         nn.Tanh()) # 69x128
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, activation):
        i = self.sigmoid_linear(input)
        j = self.tanh_linear(input)
        output = torch.sum(torch.mul(i,j), 1)
        output = activation(output) if activation is not None\
                 else output
        output = self.dropout(output)
        return output
