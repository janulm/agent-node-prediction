# page rank solver for a graph
# sources: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html
# Spearman correlation: https://discuss.pytorch.org/t/spearmans-correlation/91931/5
#

#import ogb
from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
import networkx as nx
import sys
import torch
import argparse
import torch.nn.functional as F
#from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def edge_index_to_graph(edge_index):
    """
    Convert edge index to graph.
    :param edge_index: np.ndarray, the edge index of the graph.
    :return: nx.Graph, the graph.
    """
    graph = nx.Graph()
    edge_tuple_list = zip(edge_index[0], edge_index[1])
    graph.add_edges_from(edge_tuple_list)
    return graph


def page_rank_solver(G, alpha=0.85, tol=1e-6, max_iter=1000):
    pr = nx.pagerank(G, alpha=alpha, tol=tol, max_iter=max_iter)
    
    result = torch.zeros(G.number_of_nodes(), dtype=torch.float)
    for k, v in pr.items():
        result[k] = v
    return result


def page_rank_cora():
    # get cora dataset
    #torch.set_printoptions(precision=15)
    dataset = Planetoid(root='data/Planetoid', name='Cora',transform=NormalizeFeatures())
    data = dataset[0]
    # convert to graph
    G = edge_index_to_graph(data.edge_index.numpy())
    # solve
    pr = page_rank_solver(G)
    del G
    # return x=features[vector of 1s], edge_index=adjacency matrix, y=page_rank, train_mask, val_mask, test_mask
    x = torch.ones((data.x.shape[0],1), dtype=torch.float)
    return {'x' : x,'edge_index' : data.edge_index,'y' :  pr,'train' : data.train_mask,'val' :  data.val_mask,'test' : data.test_mask}


def page_rank_pubmed():
    # get pubmed dataset
    #torch.set_printoptions(precision=15)
    dataset = Planetoid(root='data/Planetoid', name='PubMed',transform=NormalizeFeatures())
    data = dataset[0]
    # convert to graph
    G = edge_index_to_graph(data.edge_index.numpy())
    # solve
    pr = page_rank_solver(G)
    del G
    # return x=features[vector of 1s], edge_index=adjacency matrix, y=page_rank, train_mask, val_mask, test_mask
    x = torch.ones((data.x.shape[0],1), dtype=torch.float)
    return {'x' : x,'edge_index' : data.edge_index,'y' :  pr,'train' : data.train_mask,'val' :  data.val_mask,'test' : data.test_mask}

def page_rank_ogb_arxiv():
    # get ogb-arxiv dataset
    print("alive2",flush=True)
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
    # convert to graph
    G = edge_index_to_graph(data.edge_index.numpy())
    # solve
    pr = page_rank_solver(G)
    print("alive3")
    del G
    # return x=features[vector of 1s], edge_index=adjacency matrix, y=page_rank, train_mask, val_mask, test_mask
    x = torch.ones((data.x.shape[0],1), dtype=torch.float)
    idx_split = dataset.get_idx_split()
    return {'x' : x,'edge_index' : data.edge_index,'y' :  pr,'train' : idx_split['train'],'val' :  idx_split['valid'],'test' : idx_split['test']}


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp,device=x.device)
    ranks[tmp] = torch.arange(len(x),device=x.device)
    return ranks

def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    
    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)