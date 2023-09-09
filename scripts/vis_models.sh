#!/bin/bash -l

#SBATCH --nodes=4             # This needs to match Trainer(num_nodes=...)
#SBATCH --mem=16g
#SBATCH --time=2:00:00
#SBATCH -p batch


module load python/3.9.0
module load cuda/11.7.1
module load cudnn/8.6.0
module load tmux/2.8
module load gcc/10.2
module load opengl/mesa-12.0.6
module load glew/2.1.0
source /users/rrodri19/abs-mdp/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/users/rrodri19/abs-mdp/:/users/rrodri19/abs-mdp/src/:/users/rrodri19/abs-mdp/tests/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/rrodri19/.mujoco/mujoco210/bin

VIS=''
MAKE_MDP=''
RSSM=''
OUTDIR=''
BASEDIR=''
DEVICE='cpu'
DATASET=''

while getopts d:vmro:g:s: flag
do
    case "${flag}" in
        d) BASEDIR=${OPTARG};;
        v) VIS='TRUE';;
        m) MAKE_MDP='TRUE';;
        r) RSSM='--rssm';;
        o) OUT_DIR=$OPTARG;;
        g) DEVICE=$OPTARG;;
        s) DATASET=$OPTARG;;
    esac
done

if [[ -z $BASEDIR ]]; then
    echo 'ERROR: Missing Experiment dir'
    exit -1
fi


for i in $(echo "${BASEDIR}*"); do
    echo "$i"
    if [[ -n $MAKE_MDP ]]; then
        if [[ -z $DATASET ]]; then
            python scripts/make_absmdp_experiments.py --experiment $i $RSSM
        else
            python scripts/make_absmdp_experiments.py --experiment $i --dataset $DATASET $RSSM
        fi
        
    fi

    if [[ -n $VIS ]]; then

     if [[ -z $DATASET ]]; then
            python scripts/visualize_experiment.py --experiment $i --device $DEVICE $RSSM
        else
            python scripts/visualize_experiment.py --experiment $i --device $DEVICE $RSSM --dataset $DATASET
        fi
    fi
done


# for each one
    # run vis generation
    # create mdp model
    # plan?


