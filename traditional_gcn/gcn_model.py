import torch
import argparse
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

import sys
# import custom logger 
sys.path.insert(0,'/home/janulm/janulm_agent_nodes/utils') # for tikserver
sys.path.insert(0,'/home/janulm/gnn-bachelor-thesis/utils') # for tikserver
sys.path.insert(0,'/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/utils/') # for running local

from argparse import ArgumentParser

from logger import Logger


def add_model_args(parent_parser: ArgumentParser) -> ArgumentParser:
    if parent_parser is not None: 
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
    else:
        parser = ArgumentParser(add_help=False, conflict_handler='resolve')
    parser.add_argument('--device', type=int, default=0,help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_units',type=int,default=64)
    parser.add_argument('--epochs',type=int,default=2000)
    parser.add_argument('--total_num_layers',type=int,default=2)
    parser.add_argument('--training_dropout_rate',type=float,default=0.0)
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--weight_decay',type=float,default=0.01)
    # for running on server and logging: 
    parser.add_argument('--job_id',type=int, default=111)
     
    
    return parser

class GCN_Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, total_num_layers,
                 dropout, log_soft_max_last_layer=True):
        super(GCN_Model, self).__init__()

        assert(total_num_layers > 0)

        self.log_soft_max_last_layer = log_soft_max_last_layer

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        if total_num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels, cached=True))
        
        else:
            # fist layer
            self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
            for _ in range(total_num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
            # last layer
            self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout


    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        if self.log_soft_max_last_layer:
            x =  x.log_softmax(dim=-1)
        return x
