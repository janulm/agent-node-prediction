# -*- coding: utf-8 -*-

# written by Jannek Ulm, inspired by Martinkus Karolis Agent Net Github: 
# https://github.com/KarolisMart/AgentNet/blob/main/model.py

from enum import Enum
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops, coalesce
from torch_scatter import scatter_max, scatter_add
from math import sqrt, log
from typing import List, Optional
from argparse import ArgumentParser, Namespace
import torch_sparse.tensor

from util import gumbel_softmax, spmm, scatter

# ----------------------------------------------------------------------------------------------------
# Different Strategies that the model can explore
class InitStrategy(Enum):
    random = 'random' # places the num_agents randomly on the field
    one_to_one = 'one_to_one' # overrides num_agents with num_nodes and starts with one agent per node

    def __str__(self):
        return self.value

class TransitionStrategy(Enum):
    random = 'random'
    bias_attention = 'bias_attention'
    random_reset = 'random_reset' # resets position after a few number of steps to starting position
    bias_attention_reset = "bias_attention_reset" # resets position after a few number of steps to starting position
    attention = "attention"
    attention_reset = "attention_reset"

    def __str__(self):
        return self.value


class ReadOutStrategy(Enum):
    node_embedding = 'node_embedding' # use the node_embedding itself to read out node property prediction 
    #(this strategy is valid for all transition and init strategies)

    agent_start = 'agent_start' # uses the agents embedding for the node, on which the agent started
    # (only nodes that an agent starts on are valid for readout, requires one_to_one init)

    last_agent_visited = 'last_agent_visited' # uses the lasts visited agents embedding on which the agent ended, stores this intermediate agent embedding in an out vector
    
    def __str__(self):
        return self.value

# ----------------------------------------------------------------------------------------------------
# other settings for the model
def add_model_args(parent_parser: ArgumentParser) -> ArgumentParser:
    if parent_parser is not None: 
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
    else:
        parser = ArgumentParser(add_help=False, conflict_handler='resolve')
    parser.add_argument('--init_strat',default=InitStrategy.random,type=InitStrategy,choices=list(InitStrategy))
    parser.add_argument('--transition_strat',default=TransitionStrategy.attention,type=TransitionStrategy,choices=list(TransitionStrategy))
    parser.add_argument('--readout_strat',default=ReadOutStrategy.node_embedding,type=ReadOutStrategy,choices=list(ReadOutStrategy))
    parser.add_argument('--classification',type=bool,default=True)

    parser.add_argument('--device', type=int, default=0,help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_units',type=int,default=64)
    parser.add_argument('--num_agents',type=int,default=10)
    parser.add_argument('--num_steps',type=int,default=10)
    parser.add_argument('--self_loops',action='store_true',default=True)
    parser.add_argument('--epochs',type=int,default=1000)
    parser.add_argument('--reset_neighbourhood_size',type=int,default=2)
    parser.add_argument('--training_dropout_rate',type=float,default=0.0)
    parser.add_argument('--reduce_function', type=str, default='log', choices=['sum','mean','max','log','sqrt'],  help="Options are ['sum', 'mean', 'max', 'log', 'sqrt']")
    parser.add_argument('--use_time',action='store_true',default=False)
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--weight_decay',type=float,default=0.01)

    parser.add_argument('--leakyRELU_neg_slope',type=float,default=0.01)
    parser.add_argument('--leakyRELU_edge_neg_slope',type=float,default=0.2)
    parser.add_argument('--use_mlp_input',action='store_true',default=False)
    parser.add_argument('--post_ln',action='store_true',default=False)
    parser.add_argument('--activation_function', type=str, default='leaky_relu',choices=['leaky_relu','gelu','relu'])
    parser.add_argument('--global_agent_node_update',action='store_true',default=False) # use global agent for node update
    parser.add_argument('--global_agent_agent_update',action='store_true',default=False)
    parser.add_argument('--sparse_conv',action='store_true',default=False)
    parser.add_argument('--mlp_width_mult',type=int,default=2)
    parser.add_argument('--attn_width_mult',type=int,default=2)
    parser.add_argument('--num_workers',type=int, default=0)
    parser.add_argument('--visited_decay',type=float, default=0.9)
    
    
    parser.add_argument('--test_argmax',action='store_true',default=False)
    parser.add_argument('--num_pos_attention_heads',type=int, default=1)

    # cosine learning rate schedule
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--gumbel_temp', type=float, default=0.66666667)
    parser.add_argument('--gumbel_min_temp', type=float, default=0.66666667)
    parser.add_argument('--gumbel_warmup', type=int, default=-1)
    parser.add_argument('--gumbel_decay_epochs', type=int, default=100)
    parser.add_argument('--min_lr_mult', type=float, default=1e-7)

    # for running on server and logging: 
    parser.add_argument('--job_id',type=int, default=111)

     
    
    return parser

    
def printf(arg):
    print(arg,flush=True)

# ----------------------------------------------------------------------------------------------------
# Time Embedding for Model
class TimeEmbedding(nn.Module):
    # https://github.com/w86763777/pytorch-ddpm/blob/master/model.py
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb

# ----------------------------------------------------------------------------------------------------
# Model 
class AgentNet(nn.Module):
    def __init__(self, args: Namespace, num_nodes: int, num_node_features: int, num_edge_features: int, num_out_classes: int) -> None:
        super(AgentNet, self).__init__()
        # check if strategies are compliant, adjust values of namespace
        if args.readout_strat == ReadOutStrategy.agent_start and not args.init_strat == InitStrategy.one_to_one:
            raise ValueError("Strategies are not compliant")

        if args.init_strat == InitStrategy.one_to_one and not args.num_agents == num_nodes: # make sure that one agent one node
            raise ValueError("One to one init not possible: num_agents == num_nodes")
        # print out strategies and namespace:
        #printf(args)
        #  save config to self.
        self.init_strat = args.init_strat
        self.transition_strat = args.transition_strat
        self.readout_strat = args.readout_strat
        self.hidden_units = args.hidden_units
        self.dim = self.hidden_units
        self.num_agents = args.num_agents
        self.num_nodes = num_nodes
        self.num_steps = args.num_steps
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.num_out_classes = num_out_classes
        self.num_steps_reset =  args.reset_neighbourhood_size + 1  # r-hop neighbourhood on which agent is allowed to go from its original position
        assert(args.reset_neighbourhood_size >=0 )
        self.training_dropout_rate = args.training_dropout_rate
        self.reduce_function = args.reduce_function # I think this is the the global communication reduce operation
        self.use_time = args.use_time # (always use time)
        self.lr = args.lr
        self.leakyRELU_neg_slope = args.leakyRELU_neg_slope
        self.leakyRELU_edge_neg_slope = args.leakyRELU_edge_neg_slope
        self.use_mlp_input = args.use_mlp_input
        self.activation_function = args.activation_function
        self.mlp_width_mult = args.mlp_width_mult
        self.attn_width_mult = args.attn_width_mult
        self.post_ln = args.post_ln
        self.temp = 2.0/3.0
        # global agent functionality:
        self.global_agent_agent_update = args.global_agent_agent_update
        self.global_agent_node_update = args.global_agent_node_update
        self.self_loops = args.self_loops
        self.sparse_conv = args.sparse_conv
        self.visited_decay = args.visited_decay

        # not sure if below ones are needed and for what exactly
        self.test_argmax = args.test_argmax
        self.num_pos_attention_heads = args.num_pos_attention_heads # TODO not sure what exactly this means
        
        if self.activation_function == 'gelu':
            activation = nn.GELU() 
        elif self.activation_function == 'relu':
            activation = nn.ReLU() 
        elif self.activation_function == 'leaky_relu':
            activation = nn.LeakyReLU(negative_slope=self.leakyRELU_neg_slope) # NOTE make negative_slope into a param


        # Embed time step
        if self.use_time: #  True = always use time emb) # self.use_time:
            self.time_emb = TimeEmbedding(self.num_steps + 1, self.hidden_units, self.hidden_units * self.mlp_width_mult)

        # Input projection: 
        if self.use_mlp_input:
            self.input_proj = nn.Sequential(nn.Linear(self.num_node_features, self.hidden_units*2), activation, nn.Linear(self.hidden_units*2, self.hidden_units))
        else:
            self.input_proj = nn.Sequential(nn.Linear(self.num_node_features, self.hidden_units))
        # Edge Input projection
        if self.num_edge_features > 0:
            self.edge_input_proj = nn.Sequential(nn.Linear(self.num_edge_features, self.hidden_units*2), activation, nn.Linear(self.hidden_units*2, self.hidden_units))
            edge_dim = self.hidden_units
        else:
            self.edge_input_proj = nn.Sequential(nn.Identity()) # for jit
            edge_dim = 0

         # Agent embeddings
        self.agent_emb = nn.Embedding(self.num_agents, self.hidden_units) # Initialized from N(0,1) normal distribution
        # does it learn -> shuffle agent positions to account for agent learning something about start node

        # parameters to "steer" agent movement (bias part in bias_attention)
        if self.transition_strat == TransitionStrategy.bias_attention_reset or self.transition_strat == TransitionStrategy.bias_attention:
            self.back_param = nn.Parameter(torch.tensor([1.0], requires_grad=True))
            self.stay_param = nn.Parameter(torch.tensor([1.0], requires_grad=True))
            self.explored_param = nn.Parameter(torch.tensor([1.0], requires_grad=True))
            self.unexplored_param = nn.Parameter(torch.tensor([1.0], requires_grad=True))

        # Layer norms
        if self.post_ln:
            self.agent_ln = nn.LayerNorm(self.hidden_units)
            self.node_ln = nn.LayerNorm(self.hidden_units)
            self.conv_ln = nn.LayerNorm(self.hidden_units)
        else: 
            self.agent_ln = nn.Identity()
            self.node_ln = nn.Identity()
            self.conv_ln = nn.Identity()
        # Global Agent communication
        global_agent_node_update_dim = self.hidden_units if self.global_agent_node_update else 0
        global_agent_agent_update_dim = self.hidden_units if self.global_agent_agent_update else 0
        # Agent_node layer norm
        if edge_dim > 0:
            self.agent_node_ln = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm(self.hidden_units+edge_dim)), nn.Linear(self.hidden_units+edge_dim, self.hidden_units), (nn.LeakyReLU(self.leakyRELU_edge_neg_slope) if self.leakyRELU_edge_neg_slope > 0 else nn.ReLU()))
        else:
            self.agent_node_ln = nn.Identity()
        # MLP for message passing
        self.message_val = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm(self.hidden_units+edge_dim)), nn.Linear(self.dim+edge_dim, self.dim), (nn.LeakyReLU(self.leakyRELU_edge_neg_slope) if self.leakyRELU_edge_neg_slope > 0 else nn.ReLU()))
        self.conv_mlp = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm(self.hidden_units*2)), nn.Linear(self.hidden_units*2, self.hidden_units*2 * self.mlp_width_mult), activation, nn.Dropout(self.training_dropout_rate), nn.Linear(self.hidden_units*2 * self.mlp_width_mult, self.hidden_units), nn.Dropout(self.training_dropout_rate))
        self.node_mlp = nn.Sequential(nn.Identity() if self.post_ln else (nn.LayerNorm(self.hidden_units*2 + global_agent_node_update_dim)), nn.Linear(self.hidden_units*2 + global_agent_node_update_dim, self.hidden_units*2 * self.mlp_width_mult), activation, nn.Dropout(self.training_dropout_rate), nn.Linear(self.hidden_units*2 * self.mlp_width_mult, self.dim), nn.Dropout(self.training_dropout_rate)) 
        self.agent_mlp = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm(self.hidden_units*2 + edge_dim + global_agent_agent_update_dim)), nn.Linear(self.dim*2 + edge_dim + global_agent_agent_update_dim, self.hidden_units*2 * self.mlp_width_mult), activation, nn.Dropout(self.training_dropout_rate), nn.Linear(self.dim*2 * self.mlp_width_mult, self.dim), nn.Dropout(self.training_dropout_rate)) 
        
        # Add time emb projections
        if self.use_time:
            self.node_mlp_time = nn.Sequential(activation, nn.Linear(self.dim * self.mlp_width_mult, self.dim*2 + (self.dim if self.global_agent_node_update else 0)))
            self.agent_mlp_time = nn.Sequential(activation, nn.Linear(self.dim * self.mlp_width_mult, self.dim*2 + edge_dim + (self.dim if self.global_agent_agent_update else 0))) #  + extra_global_dim
            self.step_readout_mlp_time = nn.Sequential(activation, nn.Linear(self.dim * self.mlp_width_mult, self.dim))
            self.conv_mlp_time = nn.Sequential(activation, nn.Linear(self.dim * self.mlp_width_mult, self.dim*2))
        
        # Agent jump
        self.key = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm((self.dim)*2 + edge_dim)), nn.Linear((self.dim)*2 + edge_dim, self.dim*self.attn_width_mult*self.num_pos_attention_heads), nn.Identity())
        self.query = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm(self.dim)), nn.Linear(self.dim, self.dim*self.attn_width_mult*self.num_pos_attention_heads))
        self.attn_lin = nn.Sequential(nn.Linear(self.num_pos_attention_heads, 1))
    
        
        # Readout params: 
        self.classification = args.classification
        if self.readout_strat == ReadOutStrategy.agent_start:
            # this ensures (count agents == count nodes :)
            self.readout_mlp_agent = nn.Sequential(nn.Linear(self.dim,self.dim*2),activation, nn.Linear(self.dim*2, self.num_out_classes))
        elif self.readout_strat == ReadOutStrategy.last_agent_visited:
            # use agents embedding at each node after agent update to readout this nodes props: 
            self.readout_mlp_agent = nn.Sequential(nn.Linear(self.dim,self.dim*2),activation, nn.Linear(self.dim*2, self.num_out_classes))
        elif self.readout_strat == ReadOutStrategy.node_embedding: 
            self.readout_mlp_node = nn.Sequential(nn.Linear(self.dim,self.dim*2),activation, nn.Linear(self.dim*2, self.num_out_classes))
        else:
            raise ValueError("no valid readout strat")

        self.reset_parameters()        

    

    @torch.jit.ignore
    def reset_parameters(self):
        # Have learnable global BSEU [back, stay, explored, unexplored] params
        if self.transition_strat == TransitionStrategy.bias_attention or self.transition_strat == TransitionStrategy.bias_attention_reset: 
            # Bias the parameters towards exploration
            nn.init.constant_(self.back_param, 0.0)
            nn.init.constant_(self.stay_param, -1.0)
            nn.init.constant_(self.explored_param, 0.0)
            nn.init.constant_(self.unexplored_param, 5.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation_function == 'gelu':
                    nn.init.xavier_uniform_(m.weight)
                elif self.activation_function == 'relu':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                else:
                    nn.init.kaiming_uniform_(m.weight, a=self.leakyRELU_neg_slope, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                m.reset_parameters()
            elif isinstance(m, nn.Embedding):
                m.reset_parameters()


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_feat: Optional[torch.Tensor] = None):
        #
        # 4 Steps:
        # node update, neighborhood aggregation, agent update, agent transition
        # 
        assert(self.num_nodes == x.size(0))
        # out -> where the result will be stored. 
        out_emb = torch.zeros(self.num_nodes, self.num_out_classes,device=x.device)
        # get node, edge storage and preprocess feature vectors
        node_emb = self.input_proj(x) 
        edge_emb = self.edge_input_proj(edge_feat) if edge_feat is not None else None
        # Add self loops to let agent stay on a current node
        edge_index_sl = edge_index.clone()
        edge_emb_sl = edge_emb if edge_emb is None else edge_emb.clone()
        if self.self_loops:
            edge_index_sl, edge_emb_sl = add_self_loops(edge_index_sl, edge_attr=edge_emb_sl, num_nodes=self.num_nodes)
        if edge_emb_sl is not None:
            edge_index_sl, edge_emb_sl = coalesce(edge_index_sl, edge_emb_sl, self.num_nodes)
            edge_index, edge_emb = coalesce(edge_index, edge_emb, self.num_nodes)
        else:
            edge_index_sl = coalesce(edge_index_sl, None, self.num_nodes)
            edge_index = coalesce(edge_index, None, self.num_nodes)
        # Agent Embedding
        agent_emb = self.agent_emb(torch.arange(self.num_agents, device=edge_index.device)) # .unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, self.hidden_units)

        # Initialize Agent position: 
        # define agent_pos
        if self.init_strat == InitStrategy.random:
            #printf("set positions randomly")
            agent_pos = torch.randint(low=0,high=self.num_nodes,size=(self.num_agents,),device=edge_index.device)
            # could also adapt to using the first (num_agents) entries of random permutation instead of completely random nodes
        elif self.init_strat == InitStrategy.one_to_one:
            #printf("one node one agent init")
            # random permutation for one to one init and random starting position
            assert(self.num_agents == self.num_nodes and "one to one init requires agents=nodes count")
            agent_pos = torch.randperm(self.num_nodes,device=edge_index.device,dtype=torch.int64)
        else: 
            raise ValueError("error in init strategy")
        # store initial starting position
        init_agent_pos = agent_pos.clone()
        #print("init_agent_pos",init_agent_pos)
        # get agent to node mapping and get agent node attention value
        if agent_pos.max() >= self.num_nodes:
            raise ValueError # Make sure agents are not placed 'out of bounds'
        agent_node = torch.stack([torch.arange(agent_pos.size(0), device=agent_pos.device), agent_pos], dim=0) # 2 x num_agents matrix: agent id (0 to numag-1) to node id
        agent_node_attention_value = torch.ones(agent_pos.size(0), dtype=torch.float, device=agent_pos.device)
        # node to agent id ? yet to confirm
        node_agent = torch.stack([agent_node[1], agent_node[0]]) # Transpose with no coalesce
        node_agent_attn_value = agent_node_attention_value
        if edge_feat is not None:
            edge_taken_emb = torch.zeros(agent_pos.size(0), edge_emb.size(-1), device=edge_emb.device)
        # sparse adjacency matrix:
        adj = torch_sparse.tensor.SparseTensor(row=edge_index_sl[0], col=edge_index_sl[1], value=edge_emb_sl,
                                sparse_sizes=(self.num_nodes, self.num_nodes), is_sorted=False)
        del edge_index_sl
        if self.sparse_conv:
            adj_no_self_loop = torch_sparse.tensor.SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_emb,
                                sparse_sizes=(self.num_nodes, self.num_nodes), is_sorted=False)
            del edge_index
        # # Fastest option to get agent -> neighbour adjacency?
        agent_neighbour_coo = torch_sparse.tensor.__getitem__(adj, agent_node[1]).coo()
        agent_neighbour = torch.stack(agent_neighbour_coo[:2])
        agent_neighbour_edge_emb = agent_neighbour_coo[2]
        del agent_neighbour_coo
        if agent_neighbour[0].unique().size(0) != self.num_agents: # Check if all agents have a neighbor
            raise ValueError
        # Track visited nodes
        if self.transition_strat == TransitionStrategy.bias_attention or self.transition_strat == TransitionStrategy.bias_attention_reset:
            # make this thing here sparse -> adapt things later on
            # or even better make this thing size self.num_agents x 
            visited_nodes = torch.zeros(self.num_nodes, self.num_agents, dtype=torch.float, device=agent_neighbour.device)
            visited_nodes.scatter_(0, agent_pos.view(1, self.num_agents), torch.ones(1, self.num_agents, device=agent_pos.device))
            # this should replace both steps above
            #visited_nodes_sparse = torch.sparse_coo_tensor(indices=node_agent,values=torch.ones(self.num_agents, device=agent_pos.device),size=(self.num_nodes, self.num_agents),device=agent_neighbour.device,dtype=torch.float)
            #visited_nodes_sparse =visited_nodes_sparse.coalesce()

        for i in range(self.num_steps+1):
            # Get time for current step
            if self.use_time:
                time_emb = self.time_emb(torch.tensor([i], device=node_emb.device, dtype=torch.long))
            # In the first iteration just update the starting node/agent embeddings
            if i > 0:
                # Fastest option to get agent -> neighbour adjacency?
                agent_neighbour_coo = torch_sparse.tensor.__getitem__(adj, agent_node[1]).coo()
                agent_neighbour = torch.stack(agent_neighbour_coo[:2])
                agent_neighbour_edge_emb = agent_neighbour_coo[2]
                del agent_neighbour_coo
                if agent_neighbour[0].unique().size(0) != self.num_agents: # Check if all agents have a neighbor
                    #printf(i, agent_neighbour[0].unique().size(0), agent_neighbour[0], agent_neighbour[1])
                    raise ValueError 
                # find attention values for neighbors using different transition strategies: 
                # -----------------------------
                # compute agent_neighbour_attention_value
                if self.transition_strat == TransitionStrategy.random or self.transition_strat == TransitionStrategy.random_reset:
                    #printf("random transition")
                    agent_neighbour_attention_value = torch.rand(agent_neighbour.size(1), dtype=torch.float, device=agent_neighbour.device)
                elif self.transition_strat == TransitionStrategy.bias_attention or self.transition_strat == TransitionStrategy.bias_attention_reset:
                    #printf("bias attention transition")
                    # Fill in neighbor attention scores using the learned logits for [back, stay, explored, unexplored]
                    attn_score = torch.zeros_like(agent_neighbour[0], dtype=torch.float)
                    # Update tracked positions
                    previous_visited = visited_nodes[agent_neighbour[1]].gather(1, (agent_neighbour[0] % self.num_agents).unsqueeze(1)).squeeze(1)
                    # TODO sparse version of the above compute previous_visited from the sparse version (# NOTE visited_nodes_sparse.index_select(0,agent_neighbour[1]) seems to be the same as visited_nodes[agent_neighbour[1]])
                    #previous_visited_sparse = torch.zeros(agent_neighbour[1].size(0))
                    #count_active_nodes = agent_neighbour[1].size(0)
                    #for i in range(count_active_nodes):
                    #    previous_visited_sparse[i] = visited_nodes[agent_neighbour[1][i]][agent_neighbour[0][i]]
                    #assert((previous_visited_sparse == previous_visited_sparse).min() == True) # both previous visited should be identical

                    visited_nodes = visited_nodes * self.visited_decay
                    # TODO: sparse version of the above
                    #visited_nodes_sparse = visited_nodes_sparse * self.visited_decay

                    visited_nodes.scatter_(0, agent_pos.view(1, self.num_agents), torch.ones(1, self.num_agents, device=agent_pos.device))
                    # TODO: sparse replacing of above (watch out to not only add but really replace)
                    #raise ValueError("not implemented yet")

                    #for i in range (self.num_agents):
                    #    # TODO need to add 1 - the current value for each of the indices of interest: index= agent_pos[i],i
                        # first construct to add sparse tensor, then add and coalesce
                    #    visited_nodes_sparse.add()
                    #    visited_nodes_sparse.coalesce()

                    # Get tracked values for new neighbors
                    neighbors_visited = visited_nodes[agent_neighbour[1]].gather(1, (agent_neighbour[0] % self.num_agents).unsqueeze(1)).squeeze(1)

                    mask_old = neighbors_visited < 1.0 # Disregard the current node

                    # Move agents
                    Q = self.query(agent_emb).reshape(agent_emb.size(0), self.num_pos_attention_heads, -1)
                    if edge_feat is not None:
                        K = torch.cat([node_emb[agent_neighbour[1]], agent_neighbour_edge_emb, node_emb[agent_node[1]][agent_neighbour[0]]], dim=-1)
                    else:
                        K = torch.cat([node_emb[agent_neighbour[1]], node_emb[agent_node[1]][agent_neighbour[0]]], dim=-1)   
                    K = self.key(K).reshape(agent_neighbour.size(1), self.num_pos_attention_heads, -1)
                    attn_score = (Q[agent_neighbour[0]] * K).sum(dim=-1).view(-1) / sqrt(Q.size(-1))
                    del K, Q
                    if self.num_pos_attention_heads > 1:
                        attn_score = self.attn_lin(attn_score.view(agent_neighbour.size(1), self.num_pos_attention_heads))
                    # Bias attention part: 
                    attn_score[mask_old] += (neighbors_visited[mask_old] / self.visited_decay) * self.explored_param # remove one decay step, such that previous neighbor would have visited = 1.0
                    attn_score[mask_old] += ((1 - (neighbors_visited[mask_old] / self.visited_decay)) * self.unexplored_param) # Same
                    if self.self_loops:
                        attn_score[neighbors_visited==1.0] += self.stay_param # Current node has visited = 1.0
                    if i > 1: # No previous node at first step # (use this for reset as well)
                        attn_score[previous_visited==1.0] += self.back_param # Previous node has visited = 1.0 * decay
                    agent_neighbour_attention_value = gumbel_softmax(attn_score, agent_neighbour[0], num_nodes=self.num_agents, hard=True, tau=(self.temp if self.training or not self.test_argmax else 1e-6), i=i)
                    del attn_score
                elif self.transition_strat == TransitionStrategy.attention or self.transition_strat == TransitionStrategy.attention_reset:
                    #printf("attention transition")
                    # basic attention 
                    Q = self.query(agent_emb).reshape(agent_emb.size(0), self.num_pos_attention_heads, -1)
                    if edge_feat is not None:
                        K = torch.cat([node_emb[agent_neighbour[1]], agent_neighbour_edge_emb, node_emb[agent_node[1]][agent_neighbour[0]]], dim=-1)
                    else:
                        K = torch.cat([node_emb[agent_neighbour[1]], node_emb[agent_node[1]][agent_neighbour[0]]], dim=-1)  
                    K = self.key(K).reshape(agent_neighbour.size(1), self.num_pos_attention_heads, -1)
                    attn_score = (Q[agent_neighbour[0]] * K).sum(dim=-1).view(-1) / sqrt(Q.size(-1)) # dim Q
                    del K, Q
                    if self.num_pos_attention_heads > 1:
                        attn_score = self.attn_lin(attn_score.view(agent_neighbour.size(1), self.num_pos_attention_heads))
                    # zero one's the vector to define next neighbour    
                    agent_neighbour_attention_value = gumbel_softmax(attn_score, agent_neighbour[0], num_nodes=self.num_agents, hard=True, tau=(self.temp if self.training or not self.test_argmax else 1e-6), i=i)
                    del attn_score
                else: 
                    raise ValueError("no valid transition strat.")
                
                # -----------------------------
                # Get updated agent positions
                # either use some sort of randomness or use the attention value to select the next node
                reset_flag = self.transition_strat == TransitionStrategy.random_reset and i % self.num_steps_reset == 0
                reset_flag = reset_flag or self.transition_strat == TransitionStrategy.attention_reset and i % self.num_steps_reset == 0
                reset_flag = reset_flag or self.transition_strat == TransitionStrategy.bias_attention_reset and i % self.num_steps_reset == 0
                #print("reset_flag: ", reset_flag, "step:", i, self.num_steps_reset)


                if reset_flag:
                    #printf("resetting agent pos to starting location:", no edge taken)
                    #printf("performing reset")
                    agent_pos = init_agent_pos
                else: # setting neighbour using attention value: (actual edge taken)
                    indices = scatter_max(agent_neighbour_attention_value, agent_neighbour[0], dim=0, dim_size=self.num_agents)[1] # NOTE: could convert this to boolean tensor instead
                    if indices.max() >= agent_neighbour_attention_value.size(0):
                        raise ValueError # Make sure agents are not placed 'out of bounds', this should not be possible here
                    if indices.size(0) != self.num_agents: # Check if all agents have a neighbor
                        print(i, agent_pos.unique().size(0), agent_pos)
                        raise ValueError
                    agent_pos = agent_neighbour[1][indices]
                
                if agent_pos.max() >= self.num_nodes:
                    print(i, agent_pos, agent_neighbour)
                    raise ValueError # Make sure agents are not placed 'out of bounds', this should not be possible here
                    
                
                #print("agent_pos: ", agent_pos)

                agent_node = torch.stack([torch.arange(agent_pos.size(0), device=agent_pos.device), agent_pos], dim=0) # N_agents x N_nodes adjacency
                
                # TODO: does this do justice? 
                # when resetting dont attach gradients of neighbour attention to node agent
                rand_agent_eq = self.transition_strat == TransitionStrategy.random or self.transition_strat == TransitionStrategy.random_reset
                rand_agent_eq = rand_agent_eq or reset_flag

                agent_node_attention_value = torch.ones_like(agent_pos, dtype=torch.float,device=agent_pos.device) if rand_agent_eq else agent_neighbour_attention_value[indices] # NOTE multiply node emb with this to attach gradients when getting node agent is on
                node_agent = torch.stack([agent_node[1], agent_node[0]]) # Transpose with no coalesce
                node_agent_attn_value = agent_node_attention_value
                if edge_feat is not None:
                    # TODO, this doesn't work when resetting
                    edge_taken_emb = agent_neighbour_edge_emb.clone()[indices]
                if not reset_flag:
                    # indices was used -> ok to delete
                    del indices
            #
            # ---------------------------
            # Update node embeddings
            active_nodes = torch.unique(agent_pos)
            if edge_feat is not None:
                agent_cat = torch.cat([agent_emb, edge_taken_emb * agent_node_attention_value.unsqueeze(-1)], dim=-1)
            else:
                agent_cat = agent_emb
            agent_ln = self.agent_node_ln(agent_cat)
            del agent_cat
            if self.global_agent_node_update:
                global_agent = agent_ln.view(1, self.num_agents, -1).mean(dim=1, keepdim=False)
                node_update = torch.cat([node_emb[active_nodes], spmm(node_agent, node_agent_attn_value, self.num_nodes, self.num_agents, agent_ln, reduce=self.reduce_function)[active_nodes], global_agent[active_nodes]], dim=-1)
                node_update = node_update + self.node_mlp_time(time_emb)
            else:
                node_update = torch.cat([node_emb[active_nodes], spmm(node_agent, node_agent_attn_value, self.num_nodes, self.num_agents, agent_ln, reduce=self.reduce_function)[active_nodes]], dim=-1) + self.node_mlp_time(time_emb)
            del agent_ln
            node_emb[active_nodes] = self.node_ln(node_emb[active_nodes] + self.node_mlp(node_update))
            del node_update
            # Do a convolution to get neighborhood info:
            if self.sparse_conv:
                active_edge_index_coo = torch_sparse.tensor.__getitem__(adj_no_self_loop, active_nodes).coo()
                active_edge_index = torch.stack(active_edge_index_coo[:2])
                active_edge_emb = active_edge_index_coo[2]
                del active_edge_index_coo
            else: # For small graphs just doing a conv on everything, but updating only active nodes is faster
                active_edge_index = edge_index
                active_edge_emb = edge_emb
            
            if edge_feat is not None:
                # Pre-process edge embeddings
                node_cat = torch.cat([node_emb[active_edge_index[1]], active_edge_emb], dim=-1)
                # Need to materialize the messages if we have edge features
                message_val = self.message_val(node_cat)
                node_update = torch.cat([node_emb[active_nodes], scatter(message_val, active_edge_index[0], dim=0, dim_size=self.num_nodes, reduce=self.reduce_function)[active_nodes]], dim=-1)
                del message_val
            else:
                node_update = torch.cat([node_emb[active_nodes], spmm(active_edge_index, torch.ones_like(active_edge_index[0], dtype=torch.float), self.num_nodes, self.num_nodes, node_emb, reduce=self.reduce_function)[active_nodes]], dim=-1)
            # update node embedding: 
            node_emb[active_nodes] = self.conv_ln(node_emb[active_nodes] + self.conv_mlp(node_update))
            del node_update
        
            # ---------------------
            # Update Agent Embeddings: 
            # 

            if edge_feat is not None:
                agent_cat = torch.cat([agent_emb, node_emb[agent_pos] * agent_node_attention_value.unsqueeze(-1), edge_taken_emb * agent_node_attention_value.unsqueeze(-1)], dim=-1)
            else:
                agent_cat = torch.cat([agent_emb, node_emb[agent_pos] * agent_node_attention_value.unsqueeze(-1)], dim=-1)
            if self.global_agent_agent_update:
                agent_cat = torch.cat([agent_cat, global_agent.unsqueeze(1).expand(-1, self.num_agents, -1).reshape(self.num_agents, -1)], dim=-1)
            agent_emb = self.agent_ln(agent_emb + self.agent_mlp(agent_cat + self.agent_mlp_time(time_emb)))
            del agent_cat
            # done with agent 

            # ---------------------
            # intermediate readout (last_agent_visited) 
            if self.readout_strat == ReadOutStrategy.last_agent_visited:
                out_emb[agent_pos] = self.readout_mlp_agent(agent_emb)
                
        # After all steps: 
        # readout (that's not already computed on the go): 

        if self.readout_strat == ReadOutStrategy.agent_start:
            #print("read out agent at starting position")
            
            # this ensures one to one initialisation of agents
            out_emb[init_agent_pos] = self.readout_mlp_agent(agent_emb)

        elif self.readout_strat == ReadOutStrategy.node_embedding:
            #print("read out node embeddings directly")
            out_emb = self.readout_mlp_node(node_emb)

        if self.classification:
            out_emb = F.log_softmax(out_emb, dim=-1)
        return out_emb
