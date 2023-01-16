import sys
from ogb.nodeproppred import PygNodePropPredDataset
import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
#from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np

# import custom logger 
sys.path.insert(0,'/home/janulm/janulm_agent_nodes/utils') # for tikserver
sys.path.insert(0,'/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/utils/') # for running local
sys.path.insert(0,'/home/janulm/janulm_agent_nodes/traditional_gcn') # for tikserver
sys.path.insert(0,'/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/traditional_gcn/') # for running local


from logger import Logger
from gcn_model import GCN_Model, add_model_args
from page_rank_solver import page_rank_cora, page_rank_pubmed, page_rank_ogb_arxiv, spearman_correlation

def train_step(model, x, edge_index, y, train_mask, optimizer, device, criterion):
    model.train()
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    optimizer.zero_grad()
    pred = model(x, edge_index)[train_mask]
    pred = pred.squeeze(1)
    loss = criterion(pred, y[train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def eval_step(model,criterion,x,y,edge_index,train_mask,val_mask,test_mask,device): # should return the loss. 
    model.eval()
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
        # this only works for the last batch, change if multiple batches
    with torch.no_grad():
        pred = model(x, edge_index)
        pred = pred.squeeze(1)
        # compute MSE loss
        train_loss = criterion(pred[train_mask], y[train_mask])
        valid_loss = criterion(pred[val_mask], y[val_mask])
        test_loss = criterion(pred[test_mask], y[test_mask])
        train_spear = spearman_correlation(pred[train_mask], y[train_mask])
        valid_spear = spearman_correlation(pred[val_mask], y[val_mask])
        test_spear = spearman_correlation(pred[test_mask], y[test_mask])

    return train_loss, valid_loss, test_loss, train_spear, valid_spear, test_spear




def main(args,x,y,edge_index,train_mask,val_mask,test_mask):
     # get run args
    # Seed 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #random.seed(args.seed)
    print(args,flush=True)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    #print("Device: ",device)
   
    log = Logger(args.dataset,args.job_id,use_loss=True)
    spear_log = Logger(args.dataset+str(" Spearman correlation coefficient "),args.job_id,use_acc=True)

    ## model, optimizer and 
    num_nodes =  x.size(0)  
    num_node_features = 1
    num_edge_features = 0
    num_out_classes = 1
        
    model = GCN_Model(num_node_features,args.hidden_units,num_out_classes,args.total_num_layers,args.training_dropout_rate,log_soft_max_last_layer=False)
    model = model.to(device)

    reg_criterion = nn.MSELoss(reduction='mean')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    for epoch in range(args.epochs):
        #print(device)
        if device.type != "cpu":
            #print("went here",device, device == 'cpu', device != "cpu")
            torch.cuda.reset_peak_memory_stats(0)
            
        #print("=====Epoch {}".format(epoch),flush=True)
        #print('Training...',flush=True)
    
        train_step(model, x, edge_index, y, train_mask, optimizer, device, reg_criterion)

        #print('Evaluating...',flush=True)
        train_loss, val_loss, test_loss, train_spear, valid_spear, test_spear = eval_step(model,reg_criterion,x,y,edge_index,train_mask,val_mask,test_mask,device)
        # these here are losses:
        log.log_data(train_loss=train_loss,val_loss=val_loss,test_loss=test_loss)
        spear_log.log_data(train_acc=train_spear,val_acc=valid_spear,test_acc=test_spear)
        #print("Epoch: ",epoch," Train loss: ",train_loss," Val loss: ",val_loss, "Train spear: ",train_spear, " Val spear: ",valid_spear,flush=True)
        
        # early stopping
        if log.early_stopping(min_count_epochs=1000,patience=200):
            break

    #log.save_plot_data()
    t_value = log.get_test_at_best_val()
    spear_t_value = spear_log.get_test_at_best_val()
    return t_value, spear_t_value

if __name__ == '__main__':                       
    # will result in 2592 queries:
    i = 0
    print("alive1")
    datasets = ['cora','pubmed','ogb_arxiv']
    dataset = datasets[2]

    # get dataset and evaluator
    if dataset == 'cora':
        data = page_rank_cora()
    elif dataset == 'pubmed':
        data = page_rank_pubmed()
    elif dataset == 'ogb_arxiv':
        data = page_rank_ogb_arxiv()
    else:
        raise ValueError("Dataset not found")
    print("alive4")
    dims = [4,8,16]
    dropouts =[0.0,0.3] 
    lrs = [0.01,0.001,0.0001]
    total_num_layers = [1,2,3,4]

    results = np.ndarray(shape=(len(dims),len(dropouts),len(lrs),len(total_num_layers),2))


    for id_dims, dim in enumerate(dims):
        for id_dropout, dropout in enumerate(dropouts):
           for id_lr, lr in enumerate(lrs):
                for id_layers, layer_count in enumerate(total_num_layers):
                    # try and log result:
                    query = str(dim)+" "+str(lr)+" "+str(layer_count)+" "+str(dropout)
                    idx_loss = (id_dims,id_dropout,id_lr,id_layers,0)
                    idx_spear = (id_dims,id_dropout,id_lr,id_layers,1)
                    print("running ",i," : ",query,flush=True)
                    i = i + 1
                    
                    #try:
                    parser = argparse.ArgumentParser(description='PageRank (GCN-Model)')
                    parser = add_model_args(parser)
                    args = parser.parse_args()
                    # MANUAL FLAG OVERRIDE
                    
                    args.hidden_units = dim
                    args.training_dropout_rate = dropout
                    args.lr = lr
                    args.total_num_layers = layer_count
                    args.dataset = dataset
                    args.epochs = 5000

                    loss, spear_coeff = main(args,x=data['x'],edge_index=data['edge_index'],y=data['y'],train_mask=data['train'],val_mask=data['val'],test_mask=data['test'])
                    print("test_loss: ",loss,"spear coeff: ",spear_coeff,flush=True)
                    results[idx_loss] = loss
                    results[idx_spear] = spear_coeff

                    #except (RuntimeError,ValueError) as err:
                    #    results[idx_loss] = -1
                    #    results[idx_spear] = -1
                    #    print("got err",err,flush=True)
                                
    #print(results)
    #print("max result test acc: ",results[np.argmax(results)],np.argmax(results))
    print("--------------------")
    print(dims,dropouts,lrs,total_num_layers)
    print("--------------------")
    print(results)
