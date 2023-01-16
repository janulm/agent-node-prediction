#   
## ogb_arxiv.py written by Jannek Ulm 11.10.2022 (Bachelor Thesis)
#   https://github.com/KarolisMart/AgentNet
#   source: https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py


#import ogb
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch
import argparse
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import sys
import torch_geometric.transforms as T

# import custom logger 
sys.path.insert(0,'/home/janulm/janulm_agent_nodes/utils') # for tikserver
sys.path.insert(0,'/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/utils/') # for running local


from util import cos_anneal, get_cosine_schedule_with_warmup
from logger import Logger
from agent_net_model import AgentNet, add_model_args
from agent_net_model import TransitionStrategy, InitStrategy, ReadOutStrategy


from tqdm import tqdm
import numpy as np


def train_step_AgentNet(model, data, optimizer, device, criterion, train_idx):
    model.train()
    
    data = data.to(device)
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index, data.edge_attr)[train_idx]
    loss = criterion(pred, data.y[train_idx].squeeze(1))
    loss.backward()
    optimizer.step()

@torch.no_grad()
def eval_step_AgentNet(model,data,device,evaluator,split_idx): # should return the acc. 
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        pred = model(data.x, data.edge_index, data.edge_attr)
        y_pred = pred.argmax(dim=-1, keepdim=True)
        train_acc = evaluator.eval({
            'y_true': data.y[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': data.y[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': data.y[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
        })['acc']

    return train_acc, valid_acc, test_acc




def main(args):
    # Seed 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #random.seed(args.seed)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    #print("Device: ",device)
   
    # get dataset and evaluator
    evaluator = Evaluator(name=args.dataset)
    dataset = PygNodePropPredDataset(name='ogbn-arxiv') #, transform=T.ToSparseTensor())

    data = dataset[0]
    #data.adj_t = data.adj_t.to_symmetric()

    #row, col, edge_attr = data.adj_t.t().coo()
    #edge_index = torch.stack([row, col], dim=0)
    #data.edge_index = edge_index
    #del data.adj_t
    #del data.node_year
    data = data.to(device)

    #args.num_agents =  100000 # data.num_nodes

    split_idx = dataset.get_idx_split()
    
    log = Logger(args.dataset,args.job_id,only_acc=True)

    ## model, optimizer, loss, lr-scheduler
    
    num_nodes =  data.num_nodes 
    num_node_features = data.x.size(1) 
    num_edge_features = 0
    num_out_classes = 40
        
    model = AgentNet(args,num_nodes,num_node_features,num_edge_features,num_out_classes)
    model = model.to(device)

    cls_criterion = F.nll_loss

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.epochs, min_lr_mult=args.min_lr_mult)
    
    # dataloader for batches  
          
    for epoch in range(args.epochs):
        #print(device)
        if device.type != "cpu":
            #print("went here",device, device == 'cpu', device != "cpu")
            torch.cuda.reset_peak_memory_stats(0)
            
        #print("=====Epoch {}".format(epoch),flush=True)
        #print('Training...',flush=True)
        lr = scheduler.optimizer.param_groups[0]['lr']
        if args.gumbel_warmup < 0:
            gumbel_warmup = args.warmup
        else:
            gumbel_warmup = args.gumbel_warmup
        model.temp = cos_anneal(gumbel_warmup, gumbel_warmup + args.gumbel_decay_epochs, args.gumbel_temp, args.gumbel_min_temp, epoch)
        train_step_AgentNet(model, data, optimizer, device, cls_criterion,split_idx['train'])
        scheduler.step()

        #print('Evaluating...',flush=True)
        train_acc, val_acc, test_acc = eval_step_AgentNet(model,data,device,evaluator,split_idx)

        log.log_data(val_acc=val_acc,train_acc=train_acc,test_acc=test_acc)
        #print("Epoch: ",epoch," Train acc: ",train_acc," Val acc: ",val_acc,flush=True)

        # early stopping
        if log.early_stopping(min_count_epochs=210,patience=200):
            break

    #log.save_plot_data()
    t_acc = log.get_test_at_best_val()
    return t_acc

if __name__ == '__main__':                       
    # will result in 2592 queries:
    i = 0
    
    dims = [32,64,128]
    dropouts =[0.0,0.3] 
    steps = [3,6,9,18]
    lrs = [0.01,0.001,0.0001]
    reads = [ReadOutStrategy.agent_start,ReadOutStrategy.last_agent_visited, ReadOutStrategy.node_embedding]
    transs = [TransitionStrategy.attention, TransitionStrategy.attention_reset, TransitionStrategy.bias_attention, TransitionStrategy.bias_attention_reset]
    inits = [InitStrategy.one_to_one]
    neighborhood_sizes = [1,2,3]

    results = np.ndarray(shape=(len(dims),len(dropouts),len(steps),len(lrs),len(reads),len(transs),len(inits),len(neighborhood_sizes)))

    for id_dims, dim in enumerate(dims):
        for id_dropout, dropout in enumerate(dropouts):
            for id_step, step in enumerate(steps):
                for id_lr, lr in enumerate(lrs):
                    for id_read, read in enumerate(reads):
                        for id_trans, trans in enumerate(transs):
                            for id_init, init in enumerate(inits):
                                for id_neighborhood, neighborhood_size in enumerate(neighborhood_sizes):
                                    
                                    # try and log result:
                                    query = str(dim)+" "+init.value+" "+trans.value+" "+read.value+" "+str(lr)+" "+str(step)+" "+str(dropout)+" "+str(neighborhood_size)
                                    idx = (id_dims,id_dropout,id_step,id_lr,id_read,id_trans,id_init,id_neighborhood)
                                    print("running ",i," : ",query,flush=True)
                                    i = i + 1
                                    
                                    # break if no transition reset strategy and id > 0
                                    if (trans != TransitionStrategy.attention_reset and trans != TransitionStrategy.bias_attention_reset)and id_neighborhood > 0:
                                        results[idx] = -1
                                        print("skipped because reset not used")
                                        continue

                                    try:
                                        parser = argparse.ArgumentParser(description='OGBN-Arxiv (AgentNet)')
                                        parser = add_model_args(parser)
                                        args = parser.parse_args()
                                        args.dataset = "ogbn-arxiv"
                                        # MANUAL FLAG OVERRIDE
                                        args.use_time = True
                                        args.self_loops = True; # this is necessary since it seems there are isolated nodes, sometimes agents just can stay on the same node   
                                        args.use_mlp_input = True
                                        args.hidden_units = dim
                                        args.init_strat = init
                                        args.transition_strat = trans
                                        args.readout_strat = read
                                        args.num_steps = step
                                        args.training_dropout_rate = dropout
                                        args.num_agents = 169343
                                        args.lr = lr
                                        args.reset_neighbourhood_size = neighborhood_size

                                        args.epochs = 2000

                                        r = main(args)
                                        print("test_acc: ",r,flush=True)
                                        results[idx] = r

                                    except (RuntimeError,ValueError) as err:
                                        results[idx] = -1
                                        print("got err",err,flush=True)
                                
    #print(results)
    print("max result test acc: ",results[np.argmax(results)],np.argmax(results))
    print("--------------------")
    print(dims,dropouts,steps,lrs,reads,transs,inits)
    print("--------------------")
    print(results)                        