#   
# planetoid_cora.py written by Jannek Ulm 11.10.2022 (Bachelor Thesis)
# Resources: https://colab.research.google.com/drive/1LJir3T6M6Omc2Vn2GV2cDW_GV2YfI53_?usp=sharing#scrollTo=hxlhrODKz_W5
# https://blog.devgenius.io/how-to-train-a-graph-convolutional-network-on-the-cora-dataset-with-pytorch-geometric-847ed5fab9cb
#   

import sys
import torch
import argparse
import torch.nn.functional as F
#from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np

# import custom logger 
sys.path.insert(0,'/home/janulm/janulm_agent_nodes/utils') # for tikserver
sys.path.insert(0,'/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/utils/') # for running local


from util import cos_anneal, get_cosine_schedule_with_warmup
from logger import Logger
from agent_net_model import AgentNet, add_model_args
from agent_net_model import TransitionStrategy, InitStrategy, ReadOutStrategy


from tqdm import tqdm
#import numpy as np otherwise ogb import maybe gets stuck



def train_step_AgentNet(model, data, optimizer, device, criterion):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index, None)[data.train_mask]
    loss = criterion(pred, data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def eval_step_AgentNet(model,data,device): # should return the acc. 
    model.eval()
    data = data.to(device)
        # this only works for the last batch, change if multiple batches
    with torch.no_grad():
        pred = model(data.x, data.edge_index, None)
        y_pred = pred.argmax(dim=-1)
        train_acc = float((y_pred[data.train_mask] == data.y[data.train_mask]).sum()) / float(data.train_mask.sum())
        valid_acc = float((y_pred[data.val_mask] == data.y[data.val_mask]).sum()) / float(data.val_mask.sum())
        test_acc = float((y_pred[data.test_mask] == data.y[data.test_mask]).sum()) / float(data.test_mask.sum())
    return train_acc, valid_acc, test_acc




def main(args):
     # get run args
    # Seed 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #random.seed(args.seed)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    #print("Device: ",device)
   
    

    # get dataset and evaluator
    dataset = Planetoid(root='data/Planetoid', name='Cora',transform=NormalizeFeatures())
    #print('Number of graphs:',len(dataset))
    #print('Number of features',dataset.num_features)
    #print('Number of classes',dataset.num_classes)


    data = dataset[0]
    data = data.to(device)
    
    #args.num_agents =  2708 # data.num_nodes

    log = Logger(args.dataset,args.job_id,use_acc=True)

    ## model, optimizer and 
    num_nodes =  data.x.size(0)  
    num_node_features = dataset.num_features
    num_edge_features = 0
    num_out_classes = dataset.num_classes
        
    model = AgentNet(args,num_nodes,num_node_features,num_edge_features,num_out_classes)
    model = model.to(device)

    cls_criterion = F.nll_loss

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.epochs, min_lr_mult=args.min_lr_mult)
    
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
        train_step_AgentNet(model, data, optimizer, device, cls_criterion)
        scheduler.step()

        #print('Evaluating...',flush=True)
        train_acc, val_acc, test_acc = eval_step_AgentNet(model,data,device)

        log.log_data(val_acc=val_acc,train_acc=train_acc,test_acc=test_acc)
        #print("Epoch: ",epoch," Train acc: ",train_acc," Val acc: ",val_acc,flush=True)

        # early stopping
        if log.early_stopping(min_count_epochs=1500,patience=400):
            break

    #log.save_plot_data()
    t_acc = log.get_test_at_best_val()
    return t_acc

if __name__ == '__main__':                       
    # will result in 2592 queries:
    i = 0
    
    dims = [16,32,64]
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
                                        parser = argparse.ArgumentParser(description='Planetoid-Cora (AgentNet)')
                                        parser = add_model_args(parser)
                                        args = parser.parse_args()
                                        args.dataset = "planetoid-cora"
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
                                        args.num_agents = 2708
                                        args.lr = lr
                                        args.reset_neighbourhood_size = neighborhood_size

                                        args.epochs = 4000

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
