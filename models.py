import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphConvolution, GraphAggregation


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout):
        # [128, 256, 512] 8 9 5 5 0.0
        super(Generator, self).__init__()

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        layers = []
        # ffnn transform z_dim
        for c0, c1 in zip([z_dim]+conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x [16, 8]
        output = self.layers(x) # [16, 512]
        edges_logits = self.edges_layer(output)\
                       .view(-1, self.edges, self.vertexes, self.vertexes) # [16, 5, 9, 9]
        edges_logits = (edges_logits + edges_logits.permute(0,1,3,2))/2 # [16, 5, 9, 9] not sure what this op does
        edges_logits = self.dropoout(edges_logits.permute(0,2,3,1)) # [16, 9, 9, 5]
        nodes_logits = self.nodes_layer(output) # [16, 45]
        nodes_logits = self.dropoout(nodes_logits.view(-1, self.vertexes, self.nodes)) # [16, 9, 5]
        """
        edges_logits: defines bonds types [16, 9, 9, 5]
        nodes_logits: defines atom types [16, 9, 5]
        """
        return edges_logits, nodes_logits

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, conv_dim, m_dim, b_dim, dropout):
        super(Discriminator, self).__init__()
        # [[128, 64], 128, [128, 64]] 5 5 0.0
        graph_conv_dim, aux_dim, linear_dim = conv_dim # [128, 64] 128 [128, 64]

        # TODO
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, dropout) # 5, [128,64], 5
        self.agg_layer = GraphAggregation(graph_conv_dim[-1], aux_dim, b_dim, dropout) # 128, 64, 5

        # multi dense layer
        layers = []
        for c0, c1 in zip([aux_dim]+linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0,c1))
            layers.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, hidden, node, activatation=None):
        adj = adj[:,:,:,1:].permute(0,3,1,2) # slice and reorder dims
        annotations = torch.cat((hidden, node), -1) if hidden is not None else node
        h = self.gcn_layer(annotations, adj)
        annotations = torch.cat((h, hidden, node) if hidden is not None\
                                 else (h, node), -1)
        h = self.agg_layer(annotations, torch.tanh)
        h = self.linear_layer(h)

        # Need to implemente batch discriminator #
        ##########################################

        output = self.output_layer(h)
        output = activatation(output) if activatation is not None else output

        return output, h
