#!/bin/sh
#SBATCH -n 32
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH --output exp_results/antmaze/antmaze-umaze-v2/data/generate_data.log
#SBATCH -p 3090-gcondo
# Load modules

module load python/3.9.0
module load cuda/11.7.1
module load cudnn/8.6.0
module load tmux/2.8
module load gcc/10.2
module load opengl/mesa-12.0.6
module load glew/2.1.0
source /users/rrodri19/abs-mdp/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/users/rrodri19/abs-mdp/:/users/rrodri19/abs-mdp/src/:/users/rrodri19/abs-mdp/tests/
export SLURM_NTASKS_PER_NODE=10
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/rrodri19/.mujoco/mujoco210/bin


sizes=(128 512 1024 4096 8192)


ENV='antmaze-umaze-v2'
while getopts e: flag
do
    case "${flag}" in
        e) ENV=${OPTARG};;
    esac
done

for size in "${sizes[@]}"
do  
    echo "Generating data for $size"
    python experiments/antmaze/utils/generate_trajectories.py --max-horizon 64 --n-jobs 32 --num-traj $size --max-exec-time 100 --save-path exp_results/antmaze/${ENV}/data/trajectories_$size.zip --env $ENV
done

for size in "${sizes[@]}"
do  
    echo "Generating uniform data for $size"
    python experiments/antmaze/utils/generate_trajectories.py --max-horizon 64 --n-jobs 32 --num-traj $size --max-exec-time 100 --save-path exp_results/antmaze/${ENV}/data/trajectories_${size}_uniform.zip --uniform --env $ENV
done