# plots.py written by Jannek Ulm to make parallel coordinate plots
# paxplot: https://kravitsjacob.github.io/paxplot/advanced_usage.html

import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import paxplot
import pylatex as pl
sys.path.insert(0,'/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/page_rank/') # for running local
from experiments_to_df import exp_run_to_df,filter_df
from page_rank_solver import page_rank_cora, page_rank_pubmed, page_rank_ogb_arxiv


def df_to_latex_longtable(df,title,filename,list_of_cols_to_drop=None):
    if list_of_cols_to_drop is not None:
        df.drop(list_of_cols_to_drop,axis=1,inplace=True)

    doc = pl.Document()
    doc.packages.append(pl.Package('booktabs'))
    doc.packages.append(pl.Package('longtable'))

    with doc.create(pl.Section('Table: Global Faults ')):
        doc.append(pl.NoEscape(df.to_latex(longtable=True,caption=title)))

    doc.generate_pdf(filepath=filename, clean_tex=False)


def parallel_coords_plot(df,title,filename,cols_to_display=None,drop_duplicates=True,show=False,best_color='gold',other_color='winter',last_ax_label="last ax label"):
    # filter for cols to display (if cols_to_display is not None)
    df_cols = df.columns
    n_cols = len(df_cols)
    if cols_to_display is not None:
        cols_to_display.append(df_cols[n_cols-1])
        df = df[cols_to_display]
    df_cols = df.columns
    n_cols = len(df_cols)
    # drop duplicates expect last col (if drop_duplicates is True)
    if drop_duplicates:
        df.sort_values(df_cols[n_cols-1], ascending=False, inplace=True)
        df.drop_duplicates(subset=df.columns.difference([df_cols[n_cols-1]]),inplace=True)
        df.sort_values(df_cols[n_cols-1], ascending=True, inplace=True)
    
    # special case: to consider if reset not used as strat, show this 
    # if transition strat. in df_cols 

    if 'transition strat.' in df_cols and 'neighborhood reset' in df_cols:
        #df[df['transition strat.'].str.contains("reset")]
        options = ['attention','random','bias_attention']
        df.loc[df['transition strat.'].isin(options), 'neighborhood reset'] = 'not used'

    # change cols in df that contain strings to ints and change tick labels accordingly
    # for each col in df (except last col)
    # set tick labels to the occuring values in the col 
    # if contains string, then set tick labels to the strings and convert values to 0,1,2,3,4
    
    ax_ticks = []
    for i in range(n_cols-1):
        # if col contains strings
        if any (isinstance(x,str) for x in df[df_cols[i]]):
            # need conversion
            # get unique values of col
            # have them sorted string first, then numbers
            unique_values = df[df_cols[i]].unique()
            unique_strings = [x for x in unique_values if isinstance(x,str)]
            unique_numbers = [x for x in unique_values if not isinstance(x,str)]
            unique_numbers.sort()
            unique_strings.sort()
            unique_values = unique_strings + unique_numbers
            ticks = []
            labels = []
            z = 0
            for val in unique_values:
                ticks.append(z)
                labels.append(val)
                # replace in df
                df[df_cols[i]].replace(val,z,inplace=True)
                z += 1
            ax_ticks.append([ticks,labels])
        else: # col contains numbers
            # set tick labels to the unique values
            ax_ticks.append([df[df_cols[i]].unique()])

    # set tick labels for last col
    last_ax_min, last_ax_max = df[df_cols[n_cols-1]].min(), df[df_cols[n_cols-1]].max()
    
    # float("{:.2f}".format(13.949999999999999))
    last_ax_vals = [last_ax_min,last_ax_min*0.75+last_ax_max*0.25,last_ax_min*0.5+last_ax_max*0.5,last_ax_min*0.25+last_ax_max*0.75,last_ax_max]
    last_ax_vals_formatted = [float("{:.2f}".format(x)) for x in last_ax_vals]
    ax_ticks.append([last_ax_vals_formatted])

    # sort for right painting order (increasing test acc/spear coeff.)
    df.sort_values(df_cols[n_cols-1], ascending=True, inplace=True)
    
    # split df into best and other df
    metric = df_cols[n_cols-1]
    df_best = df[df[metric] == df[metric].max()]
    df_other = df[df[metric] < df[metric].max()]
    # plot other df with other_color
    paxfig = paxplot.pax_parallel(n_axes=n_cols)
    #plt.gcf().set_size_inches(10, 5)
    paxfig.plot(df_other.to_numpy(),line_kwargs={'alpha':1})
    paxfig.add_colorbar(ax_idx=n_cols-1,cmap=other_color,colorbar_kwargs={'label': last_ax_label})
    # plot best df with best_color
    paxfig.plot(df_best.to_numpy(),line_kwargs={'alpha':1,'color': best_color,'zorder':100,'linewidth':2.5})
    # set title
    mid_axis_idx = int(n_cols/2)
    paxfig.axes[mid_axis_idx].set_title(title,fontsize=12,pad=20)
    
    # set other axis lims (add buffer)
    for i in range(n_cols):
        val_range = df[df_cols[i]].max() - df[df_cols[i]].min()
        # add range buffer
        pad = val_range*0.1
        paxfig.set_lim(ax_idx=i, bottom=df[df_cols[i]].min()-pad, top=df[df_cols[i]].max()+pad)
    

    # set axis labels:
    paxfig.set_labels(df_cols)
    for i in range(len(ax_ticks)):
        # set ticks: 
        if len(ax_ticks[i]) == 1:
            paxfig.set_ticks(ax_idx=i, ticks=ax_ticks[i][0])
        else:
            paxfig.set_ticks(ax_idx=i, ticks=ax_ticks[i][0],labels=ax_ticks[i][1])

    # white background on axis ticks
    bbox = dict(boxstyle="round", ec="white", fc="white", alpha=0.7)
    for i in range(n_cols):
        plt.setp(paxfig.axes[i].get_yticklabels(), bbox=bbox)
    if show:
        plt.show()
    plt.tight_layout()
    paxfig.savefig(filename+'.png',dpi=300)
    plt.clf()
    return


def histogram(array, title,xlabel, filename, show=False): 
    plt.hist(array, bins='auto')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(filename+'.png',dpi=300)
    if show:
        plt.show()
    plt.clf()
    return

def triple_hist(p_cora, p_pubmed, p_ogb, show=False):
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(8, 5)
    axes[0].hist(p_cora, bins='auto')
    axes[1].hist(p_pubmed, bins='auto')
    axes[2].hist(p_ogb, bins='auto')

    axes[0].set_title("Cora")
    axes[1].set_title("PubMed")
    axes[2].set_title("OGB-Arxiv")
    fig.suptitle("Page Rank Histograms")
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[2].set_yscale('log')

    axes[0].set_xlabel("PageRank value")
    axes[1].set_xlabel("PageRank value")
    axes[2].set_xlabel("PageRank value")
    axes[0].set_ylabel("count")
    axes[1].set_ylabel("count")
    axes[2].set_ylabel("count")

    #plt.xlabel("xlabel")
    #plt.ylabel("count")
    #axes[0].yscale('log')
    plt.tight_layout()
    plt.savefig('triple_hist.png',dpi=300)
    if show:
        plt.show()
    plt.clf()
    return

if __name__ == "__main__":

    print("starting")
    #### NODE CLASSIFICATION DATASETS ####
    if True:
        node_gcn_cora = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/traditional_gcn/slurm_log/583503.out",
                            name_space_args=[['hidden_units','hidden units'], ['training_dropout_rate','dropout'], ['lr'], 
                            ['total_num_layers','num layers']],
                            res_args=[['test_acc',"test acc"]],
                            filter_flag=True)
        
        node_gcn_pubmed = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/traditional_gcn/slurm_log/583505.out",
                            name_space_args=[['hidden_units','hidden units'], ['training_dropout_rate','dropout'], ['lr'], 
                            ['total_num_layers','num layers']],
                            res_args=[['test_acc',"test acc"]],
                            filter_flag=True)

        node_gcn_ogb = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/traditional_gcn/slurm_log/583567.out",
                            name_space_args=[['hidden_units','hidden units'], ['training_dropout_rate','dropout'], ['lr'], 
                            ['total_num_layers','num layers']],
                            res_args=[['test_acc',"test acc"]],
                            filter_flag=True)

        node_agent_cora = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/agent-net/slurm_log/581259.out",
                            name_space_args=[['hidden_units','hidden units'], ['init_strat','init strat.'], 
                            ['transition_strat','transition strat.'],['readout_strat','readout strat.']
                            ,['num_steps','agent steps'], ['training_dropout_rate','dropout'], ['lr'], ['num_agents','num agents'], ['reset_neighbourhood_size','n.hood reset']
                            ],
                            res_args=[['test_acc',"test acc"]],
                            filter_flag=True)

        node_agent_pubmed = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/agent-net/slurm_log/581258.out",
                            name_space_args = [ ['hidden_units','hidden units'], ['init_strat','init strat.'], 
                            ['transition_strat','transition strat.'],['readout_strat','readout strat.']
                            ,['num_steps','agent steps'], ['training_dropout_rate','dropout'], ['lr'], ['num_agents','num agents'], ['reset_neighbourhood_size','n.hood reset']
                            ],
                            res_args=[['test_acc',"test acc"]],
                            filter_flag=True)

        node_agent_ogb_part1 = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/agent-net/slurm_log/581260.out",
                            name_space_args=[['hidden_units','hidden units'], ['init_strat','init strat.'], 
                            ['transition_strat','transition strat.'],['readout_strat','readout strat.']
                            ,['num_steps','agent steps'], ['training_dropout_rate','dropout'], ['lr'], ['num_agents','num agents'], ['reset_neighbourhood_size','n.hood reset']
                            ],
                            res_args=[['test_acc',"test acc"]],
                            filter_flag=True)
            
        node_agent_ogb_part2 = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/agent-net/slurm_log/583482.out",
                            name_space_args=[['hidden_units','hidden units'], ['init_strat','init strat.'], 
                            ['transition_strat','transition strat.'],['readout_strat','readout strat.']
                            ,['num_steps','agent steps'], ['training_dropout_rate','dropout'], ['lr'], ['num_agents','num agents'], ['reset_neighbourhood_size','n.hood reset']
                            ],
                            res_args=[['test_acc',"test acc"]],
                            filter_flag=True)

        node_agent_ogb = pd.concat([node_agent_ogb_part1,node_agent_ogb_part2])
        del node_agent_ogb_part1,node_agent_ogb_part2
        node_agent_ogb.sort_values('test acc', ascending=True, inplace=True)

    print("done") 
    

    ### PAGE RANK DATASETS ###
      ## GCN
    if True:
        page_gcn_cora = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/page_rank/slurm_log/583753.out",
                            name_space_args=[['hidden_units','hidden units'], ['training_dropout_rate','dropout'], ['lr'], ['total_num_layers','num layers']],
                            res_args=[['spear coeff','spear coeff.'],['test_loss',"test loss"]],
                            filter_flag=False)
        
        page_gcn_pubmed = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/page_rank/slurm_log/583772.out",
                            name_space_args=[['hidden_units','hidden units'], ['training_dropout_rate','dropout'], ['lr'], ['total_num_layers','num layers']],
                            res_args=[['spear coeff','spear coeff.'],['test_loss',"test loss"]],
                            filter_flag=False)
        
        page_gcn_ogb = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/page_rank/slurm_log/583781.out",
                            name_space_args=[['hidden_units','hidden units'], ['training_dropout_rate','dropout'], ['lr'], ['total_num_layers','num layers']],
                            res_args=[['spear coeff','spear coeff.'],['test_loss',"test loss"]],
                            filter_flag=False)
        
        ## AGENT CORA 
        page_agent_cora_part1 = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/page_rank/slurm_log/583802.out",
                            name_space_args=[['hidden_units','hidden units'], ['init_strat','init strat.'], ['transition_strat','transition strat.'],['readout_strat','readout strat.'],['num_steps','agent steps'], ['training_dropout_rate','dropout'], ['lr'], ['num_agents','num agents'], ['reset_neighbourhood_size','n.hood reset']],
                            res_args=[['spear coeff','spear coeff.'],['test_loss',"test loss"]],
                            filter_flag=False)

        page_agent_cora_part2 = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/page_rank/slurm_log/584219.out",
                            name_space_args=[['hidden_units','hidden units'], ['init_strat','init strat.'], ['transition_strat','transition strat.'],['readout_strat','readout strat.'],['num_steps','agent steps'], ['training_dropout_rate','dropout'], ['lr'], ['num_agents','num agents'], ['reset_neighbourhood_size','n.hood reset']],
                            res_args=[['spear coeff','spear coeff.'],['test_loss',"test loss"]],
                            filter_flag=False)

        page_agent_cora = pd.concat([page_agent_cora_part1,page_agent_cora_part2])
        del page_agent_cora_part1,page_agent_cora_part2
        page_agent_cora.sort_values('spear coeff.', ascending=True, inplace=True)
        
        # AGENT PUBMED

        page_agent_pubmed_part1 = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/page_rank/slurm_log/583807.out",
                            name_space_args=[['hidden_units','hidden units'], ['init_strat','init strat.'], ['transition_strat','transition strat.'],['readout_strat','readout strat.'],['num_steps','agent steps'], ['training_dropout_rate','dropout'], ['lr'], ['num_agents','num agents'], ['reset_neighbourhood_size','n.hood reset']],
                            res_args=[['spear coeff','spear coeff.'],['test_loss',"test loss"]],
                            filter_flag=False)

        page_agent_pubmed_part2 = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/page_rank/slurm_log/584223.out",
                            name_space_args=[['hidden_units','hidden units'], ['init_strat','init strat.'], ['transition_strat','transition strat.'],['readout_strat','readout strat.'],['num_steps','agent steps'], ['training_dropout_rate','dropout'], ['lr'], ['num_agents','num agents'], ['reset_neighbourhood_size','n.hood reset']],
                            res_args=[['spear coeff','spear coeff.'],['test_loss',"test loss"]],
                            filter_flag=False)

        page_agent_pubmed = pd.concat([page_agent_pubmed_part1,page_agent_pubmed_part2])
        page_agent_pubmed.sort_values('spear coeff.', ascending=True, inplace=True)
        del page_agent_pubmed_part1,page_agent_pubmed_part2
        # AGENT OGB

        page_agent_ogb_part1 = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/page_rank/slurm_log/583809.out",
                            name_space_args=[['hidden_units','hidden units'], ['init_strat','init strat.'], ['transition_strat','transition strat.'],['readout_strat','readout strat.'],['num_steps','agent steps'], ['training_dropout_rate','dropout'], ['lr'], ['num_agents','num agents'], ['reset_neighbourhood_size','n.hood reset']],
                            res_args=[['spear coeff','spear coeff.'],['test_loss',"test loss"]],
                            filter_flag=False)

        page_agent_ogb_part2 = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/page_rank/slurm_log/584225.out",
                            name_space_args=[['hidden_units','hidden units'], ['init_strat','init strat.'], ['transition_strat','transition strat.'],['readout_strat','readout strat.'],['num_steps','agent steps'], ['training_dropout_rate','dropout'], ['lr'], ['num_agents','num agents'], ['reset_neighbourhood_size','n.hood reset']],
                            res_args=[['spear coeff','spear coeff.'],['test_loss',"test loss"]],
                            filter_flag=False)

        page_agent_ogb_part3 = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/page_rank/slurm_log/584609.out",
                            name_space_args=[['hidden_units','hidden units'], ['init_strat','init strat.'], ['transition_strat','transition strat.'],['readout_strat','readout strat.'],['num_steps','agent steps'], ['training_dropout_rate','dropout'], ['lr'], ['num_agents','num agents'], ['reset_neighbourhood_size','n.hood reset']],
                            res_args=[['spear coeff','spear coeff.'],['test_loss',"test loss"]],
                            filter_flag=False)
        
        page_agent_ogb_part4 = exp_run_to_df(file_path="/Users/jannekulm/Documents/eth/SM7/bachelor_thesis/janulm_agent_nodes/page_rank/slurm_log/584682.out",
                            name_space_args=[['hidden_units','hidden units'], ['init_strat','init strat.'], ['transition_strat','transition strat.'],['readout_strat','readout strat.'],['num_steps','agent steps'], ['training_dropout_rate','dropout'], ['lr'], ['num_agents','num agents'], ['reset_neighbourhood_size','n.hood reset']],
                            res_args=[['spear coeff','spear coeff.'],['test_loss',"test loss"]],
                            filter_flag=False)

        
        page_agent_ogb = pd.concat([page_agent_ogb_part1,page_agent_ogb_part2,page_agent_ogb_part3,page_agent_ogb_part4])
        page_agent_ogb.sort_values('spear coeff.', ascending=True, inplace=True)
        del page_agent_ogb_part1,page_agent_ogb_part2,page_agent_ogb_part3,page_agent_ogb_part4
    
    if True: 
        d1 = filter_df(page_gcn_cora,['hidden units','num layers'])
        d2 = filter_df(page_gcn_pubmed,['hidden units','num layers'])
        d3 = filter_df(page_gcn_ogb,['hidden units','num layers'])
        d4 = filter_df(page_agent_cora,['num agents','n.hood reset','transition strat.','readout strat.'])
        d5 = filter_df(page_agent_pubmed,['num agents','n.hood reset','transition strat.','readout strat.'])
        d6 = filter_df(page_agent_ogb,['num agents','n.hood reset','transition strat.','readout strat.'])

        df_to_latex_longtable(d1,"Computation results of Baseline-GCN on Cora","tex_page_gcn_cora")
        df_to_latex_longtable(d2,"Computation results of Baseline-GCN on PubMed","tex_page_gcn_pubmed")
        df_to_latex_longtable(d3,"Computation results of Baseline-GCN on OGB","tex_page_gcn_ogb")
        df_to_latex_longtable(d4,"Computation results of AgentNet on Cora","tex_page_agent_cora",['hidden units','init strat.','dropout','lr','test loss'])
        df_to_latex_longtable(d5,"Computation results of AgentNet on PubMed","tex_page_agent_pubmed",['hidden units','init strat.','dropout','lr','test loss'])
        df_to_latex_longtable(d6,"Computation results of AgentNet on OGB","tex_page_agent_ogb",['hidden units','init strat.','dropout','lr','test loss'])

        e1 = filter_df(node_gcn_cora,['hidden units','num layers'])
        e2 = filter_df(node_gcn_pubmed,['hidden units','num layers'])
        e3 = filter_df(node_gcn_ogb,['hidden units','num layers'])
        e4 = filter_df(node_agent_cora,['agent steps','n.hood reset','transition strat.','readout strat.'])
        e5 = filter_df(node_agent_pubmed,['agent steps','n.hood reset','transition strat.','readout strat.'])
        e6 = filter_df(node_agent_ogb,['agent steps','n.hood reset','transition strat.','readout strat.'])

        df_to_latex_longtable(e1,"Computation results of Baseline-GCN on Cora","tex_node_gcn_cora")
        df_to_latex_longtable(e2,"Computation results of Baseline-GCN on PubMed","tex_node_gcn_pubmed")
        df_to_latex_longtable(e3,"Computation results of Baseline-GCN on OGB","tex_node_gcn_ogb")
        df_to_latex_longtable(e4,"Computation results of AgentNet on Cora","tex_node_agent_cora",['hidden units','init strat.','dropout','lr', 'num agents'])
        df_to_latex_longtable(e5,"Computation results of AgentNet on PubMed","tex_node_agent_pubmed",['hidden units','init strat.','dropout','lr', 'num agents'])
        df_to_latex_longtable(e6,"Computation results of AgentNet on OGB","tex_node_agent_ogb",['hidden units','init strat.','dropout','lr', 'num agents'])
        

    print("Done")
    
    #### PLOTS #####
    #print("Plotting")
    if False:
        filter_df(node_agent_cora,['agent steps'])
        filter_df(node_agent_pubmed,['agent steps'])
        filter_df(node_agent_ogb,['agent steps'])

        filter_df(node_gcn_cora,['num layers'])
        filter_df(node_gcn_pubmed,['num layers'])
        filter_df(node_gcn_ogb,['num layers'])

    if False:

        parallel_coords_plot(page_agent_cora,"PageRank - AgentNet on Cora","page_agent_cora",cols_to_display=['num agents','transition strat.','readout strat.'],drop_duplicates=True,show=False,best_color='green',other_color='plasma',last_ax_label="Spearman rank coefficient")
        parallel_coords_plot(page_agent_pubmed,"PageRank - AgentNet on Pubmed","page_agent_pubmed",cols_to_display=['num agents','transition strat.','readout strat.'],drop_duplicates=True,show=False,best_color='green',other_color='plasma',last_ax_label="Spearman rank coefficient")
        parallel_coords_plot(page_agent_ogb,"PageRank - AgentNet on OGB","page_agent_ogb",cols_to_display=['num agents','transition strat.','readout strat.'],drop_duplicates=True,show=False,best_color='green',other_color='plasma',last_ax_label="Spearman rank coefficient")
        
    if False: 
        parallel_coords_plot(node_agent_cora,"Node Classification - AgentNet on Cora","node_agent_cora",cols_to_display=[
        'agent steps',
        #'neighborhood reset',
        'transition strat.','readout strat.'],drop_duplicates=True,show=False,best_color='green',other_color='plasma',last_ax_label="node classification test accuracy")
        parallel_coords_plot(node_agent_pubmed,"Node Classification - AgentNet on Pubmed","node_agent_pubmed",cols_to_display=[
        'agent steps',
        #'neighborhood reset',
        'transition strat.','readout strat.'],drop_duplicates=True,show=False,best_color='green',other_color='plasma',last_ax_label="node classification test accuracy")
        parallel_coords_plot(node_agent_ogb,"Node Classification - AgentNet on OGB","node_agent_ogb",cols_to_display=['agent steps',
        #'neighborhood reset',
        'transition strat.','readout strat.'],drop_duplicates=True,show=False,best_color='green',other_color='plasma',last_ax_label="node classification test accuracy")
        
    if False:
        parallel_coords_plot(page_gcn_cora,"PageRank - GCN on Cora","page_gcn_cora",cols_to_display=['hidden units','num layers'],drop_duplicates=True,show=False,best_color='green',other_color='plasma',last_ax_label="Spearman rank coefficient")
        parallel_coords_plot(page_gcn_pubmed,"PageRank - GCN on Pubmed","page_gcn_pubmed",cols_to_display=['hidden units','num layers'],drop_duplicates=True,show=False,best_color='green',other_color='plasma',last_ax_label="Spearman rank coefficient")
        parallel_coords_plot(page_gcn_ogb,"PageRank - GCN on OGB","page_gcn_ogb",cols_to_display=['hidden units','num layers'],drop_duplicates=True,show=False,best_color='green',other_color='plasma',last_ax_label="Spearman rank coefficient")

        parallel_coords_plot(node_gcn_cora,"Node Classification - GCN on Cora","node_gcn_cora",cols_to_display=['hidden units','num layers'],drop_duplicates=True,show=False,best_color='green',other_color='plasma',last_ax_label="node classification test accuracy")
        parallel_coords_plot(node_gcn_pubmed,"Node Classification - GCN on Pubmed","node_gcn_pubmed",cols_to_display=['hidden units','num layers'],drop_duplicates=True,show=False,best_color='green',other_color='plasma',last_ax_label="node classification test accuracy")
        parallel_coords_plot(node_gcn_ogb,"Node Classification - GCN on OGB","node_gcn_ogb",cols_to_display=['hidden units','num layers'],drop_duplicates=True,show=False,best_color='green',other_color='plasma',last_ax_label="node classification test accuracy")

    if False:
        # Hisograms of page rank datasets 
        p_cora = page_rank_cora()['y'].numpy()
        p_pubmed = page_rank_pubmed()['y'].numpy()
        p_ogb = page_rank_ogb_arxiv()['y'].numpy() 

        histogram(p_cora, "Histogram PageRank Cora","PageRank value","hist_cora", show=False)
        histogram(p_pubmed, "Histogram PageRank PubMed","PageRank value","hist_pubmed", show=False)
        histogram(p_ogb, "Histogram PageRank OGB-Arxiv","PageRank value","hist_ogb", show=False)

        triple_hist(p_cora, p_pubmed, p_ogb, show=False)

    # Latex tables

    #node_gcn_cora.to_latex("node_gcn_cora.tex",index=False)