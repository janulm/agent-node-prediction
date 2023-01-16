#   
# gcn_ogb_arxiv.py written by Jannek Ulm 11.10.2022 (Bachelor Thesis)
# Resources: https://colab.research.google.com/drive/1LJir3T6M6Omc2Vn2GV2cDW_GV2YfI53_?usp=sharing#scrollTo=hxlhrODKz_W5
# https://blog.devgenius.io/how-to-train-a-graph-convolutional-network-on-the-cora-dataset-with-pytorch-geometric-847ed5fab9cb
#   

import sys
import torch
import argparse
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import torch_geometric.transforms as T

# import custom logger 
sys.path.insert(0,'/home/janulm/janulm_agent_nodes/utils') # for tikserver
sys.path.insert(0,'/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/utils/') # for running local


from logger import Logger
from gcn_model import GCN_Model, add_model_args



from tqdm import tqdm
#import numpy as np otherwise ogb import maybe gets stuck



def train_step(model, data, optimizer, device, criterion,train_mask):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    pred = model(data.x, data.adj_t)[train_mask]
    loss = criterion(pred, target = data.y.squeeze(1)[train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def eval_step(model,data,device,evaluator,split_idx): # should return the acc. 
    model.eval()
    data = data.to(device)
        # this only works for the last batch, change if multiple batches
    with torch.no_grad():
        pred = model(data.x, data.adj_t)
        y_pred = pred.argmax(dim=-1,keepdim=True)
        train = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
        })
        valid = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
        })
        test = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
        })
    return train['acc'], valid['acc'], test['acc']




def main(args):
     # get run args
    # Seed 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #random.seed(args.seed)
    print(args,flush=True)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    #print("Device: ",device)
   
    

    # get dataset and evaluator
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
    evaluator = Evaluator(name='ogbn-arxiv')
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)
    
    #args.num_agents =  2708 # data.num_nodes

    log = Logger(args.dataset,args.job_id,only_acc=True)

    ## model, optimizer and 
    num_nodes =  data.x.size(0)  
    num_node_features = dataset.num_features
    num_edge_features = 0
    num_out_classes = dataset.num_classes
        
    model = GCN_Model(num_node_features,args.hidden_units,num_out_classes,args.total_num_layers,args.training_dropout_rate)
    model = model.to(device)

    cls_criterion = F.nll_loss

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    for epoch in range(args.epochs):
        #print(device)
        if device.type != "cpu":
            #print("went here",device, device == 'cpu', device != "cpu")
            torch.cuda.reset_peak_memory_stats(0)
            
        #print("=====Epoch {}".format(epoch),flush=True)
        #print('Training...',flush=True)
    
        train_step(model, data, optimizer, device, cls_criterion,split_idx['train'])

        #print('Evaluating...',flush=True)
        train_acc, val_acc, test_acc = eval_step(model,data,device,evaluator,split_idx)

        log.log_data(val_acc=val_acc,train_acc=train_acc,test_acc=test_acc)
        #print("Epoch: ",epoch," Train acc: ",train_acc," Val acc: ",val_acc,flush=True)

        # early stopping
        if log.early_stopping(min_count_epochs=1000,patience=200):
            break

    #log.save_plot_data()
    t_acc = log.get_test_at_best_val()
    return t_acc

if __name__ == '__main__':                       
    # 
    i = 0
    
    dims = [256, 32,64,128]
    dropouts =[0.5,0.3,0.0] 
    lrs = [0.01,0.001,0.0001]
    total_num_layers = [1,2,3]

    results = np.ndarray(shape=(len(dims),len(dropouts),len(lrs),len(total_num_layers)))


    for id_dims, dim in enumerate(dims):
        for id_dropout, dropout in enumerate(dropouts):
           for id_lr, lr in enumerate(lrs):
                for id_layers, layer_count in enumerate(total_num_layers):
                    # try and log result:
                    query = str(dim)+" "+str(lr)+" "+str(layer_count)+" "+str(dropout)
                    idx = (id_dims,id_dropout,id_lr,id_layers)
                    print("running ",i," : ",query,flush=True)
                    i = i + 1
                    
                    #try:
                    parser = argparse.ArgumentParser(description='OGB-Arxiv (GCN-Model)')
                    parser = add_model_args(parser)
                    args = parser.parse_args()
                    args.dataset = "ogb-arxiv"
                    # MANUAL FLAG OVERRIDE
                    
                    args.hidden_units = dim
                    args.training_dropout_rate = dropout
                    args.lr = lr
                    args.total_num_layers = layer_count
                    args.weight_decay = 5e-4

                    args.epochs = 2000

                    r = main(args)
                    print("test_acc: ",r,flush=True)
                    results[idx] = r

                    #except (RuntimeError,ValueError) as err:
                    #    results[idx] = -1
                    #    print("got err",err,flush=True)
                                
    #print(results)
    print("max result test acc: ",results[np.argmax(results)],np.argmax(results))
    print("--------------------")
    print(dims,dropouts,lrs,total_num_layers)
    print("--------------------")
    print(results)