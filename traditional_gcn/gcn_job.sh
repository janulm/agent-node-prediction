#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --job-name=agent-net
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --account=tik-highmem
#SBATCH --gres=gpu:1
#SBATCH --mail-type=NONE                           # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=slurm_log/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=slurm_log/%j.err                  # where to store error messages
#SBATCH --constraint='geforce_rtx_3090|titan_rtx|tesla_v100'

# --constraint='geforce_rtx_3090|titan_rtx|tesla_v100'

# 
# |titan_rtx|tesla_v100
# get shell: srun --gres=gpu:1 --constraint='geforce_rtx_3090'  --account=tik-highmem --pty bash

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID

# exit on errors
set -o errexit
# binary to execute

. /itet-stor/janulm/net_scratch/conda/etc/profile.d/conda.sh


conda activate AgentGNN

#python /home/janulm/janulm_agent_nodes/traditional_gcn/gcn_cora.py
python /home/janulm/janulm_agent_nodes/traditional_gcn/gcn_ogb_arxiv.py
#python /home/janulm/janulm_agent_nodes/traditional_gcn/gcn_pubmed.py

echo finished at: `date`
exit 0;