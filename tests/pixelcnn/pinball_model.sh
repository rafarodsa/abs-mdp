#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-node=3   # This needs to match Trainer(devices=...)
#SBATCH --mem=32g
#SBATCH --time=0-12:00:00
#SBATCH -p 3090-gcondo
#SBATCH --output pinball-log-%J.txt

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

# run script from above
srun python tests/pixelcnn/pinball_model.py --config experiments/pb_obstacles/pixel/config/test_decoder.yaml --devices 3 --epochs 600 --batch-size 64 --lr 1e-4 --strategy ddp --save-path ./ckpts-25-pixel-models-256
