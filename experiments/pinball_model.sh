#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4  # This needs to match Trainer(devices=...)
#SBATCH --mem=32g
#SBATCH --time=0-23:00:00
#SBATCH -p 3090-gcondo
#SBATCH --output experiments/logs/pinball-log-%J.txt
#SBATCH --exclude=gpu2114,gpu2102

module load python/3.9.0
module load cuda/10.2
module load cudnn/7.6.5
module load tmux/2.8
source /users/rrodri19/abs-mdp/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/users/rrodri19/abs-mdp/:/users/rrodri19/abs-mdp/src/:/users/rrodri19/abs-mdp/tests/


# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo
