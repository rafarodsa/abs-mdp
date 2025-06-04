# Abstract MDP Learning

This repository contains implementations for learning abstract Markov Decision Processes (MDPs) from the paper "Learning Abstract World Models for Value-preserving Planning with Options"

## Overview


- **Data Generation**: Collecting trajectory data using option-based policies
- **Model Pretraining**: Training abstract world models using temporal predictive coding (TPC)
- **Planning & Training**: Online planning and training of agents using the learned abstractions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd abs-mdp
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── experiments/          # Main experiment scripts
│   ├── antmaze/          # AntMaze environment experiments
│   │   ├── utils/        # Data generation utilities
│   │   └── config/       # Configuration files
│   └── pretrain_mdp.py   # Model pretraining script
├── src/                  # Source code
│   ├── absmdp/           # Abstract MDP implementations
│   ├── agents/           # RL agents
│   ├── models/           # Neural network models
│   └── options/          # Option/skill implementations
├── envs/                 # Environment implementations
└── exp_results/          # Experiment results and data
```

## Usage

### 1. Data Generation

Generate trajectory data for training abstract models:

```bash
python experiments/antmaze/utils/generate_trajectories.py \
    --max-horizon 64 \
    --n-jobs 32 \
    --num-traj 32 \
    --max-exec-time 100 \
    --save-path exp_results/reproduce/antmaze-medium-play-v2/data/trajectories.zip \
    --env antmaze-medium-play-v2
```

**Parameters:**
- `--max-horizon`: Maximum trajectory length (default: 64)
- `--n-jobs`: Number of parallel jobs for data collection (default: 32)
- `--num-traj`: Number of trajectories to generate
- `--max-exec-time`: Maximum execution time per option (default: 100)
- `--save-path`: Path to save the generated trajectory data
- `--env`: Environment name (e.g., `antmaze-medium-play-v2`, `antmaze-umaze-v2`)

**Example:**
```bash
# Generate 32 trajectories for antmaze-medium-play-v2
python experiments/antmaze/utils/generate_trajectories.py \
    --max-horizon 64 \
    --n-jobs 32 \
    --num-traj 32 \
    --max-exec-time 100 \
    --save-path exp_results/reproduce/antmaze-medium-play-v2/data/trajectories.zip \
    --env antmaze-medium-play-v2
```

### 2. Model Pretraining


```bash
python experiments/pretrain_mdp.py \
    --config experiments/antmaze/config/tpc_cfg_critic_mixture.yaml \
    --experiment_cwd exp_results/reproduce/antmaze-medium-play-v2/mdps \
    --data.data_path exp_results/reproduce/antmaze-medium-play-v2/data/trajectories.zip \
    --accelerator gpu \
    --epochs 1 \
    --model.latent_dim 16 \
    --tag test_reproduce
```

**Parameters:**
- `--config`: Configuration file for the model
- `--experiment_cwd`: Experiment working directory for saving results
- `--data.data_path`: Path to the trajectory data generated in step 1
- `--accelerator`: Training accelerator (`gpu` or `cpu`)
- `--epochs`: Number of training epochs
- `--model.latent_dim`: Dimensionality of the latent representation
- `--tag`: Experiment tag for organization

**Available Models:**
- TPC Critic Mixture: `experiments/antmaze/config/tpc_cfg_critic_mixture.yaml`
- Other configurations available in `experiments/antmaze/config/`

### 3. Planning & Training

Train agents using the pretrained abstract models for online planning:

```bash
python experiments/antmaze/plan_and_train.py \
    --config experiments/antmaze/online_planner.yaml \
    --experiment_name plan_antmaze \
    --planner.agent.replay_buffer_size 1000000 \
    --experiment.steps 100000 \
    --experiment.eval_interval 5000 \
    --experiment.seed 19 \
    --planner.agent.lr 1e-5 \
    --exp_id test \
    --world_model.ckpt exp_results/reproduce/antmaze-medium-play-v2/mdps/test_reproduce/phi_train/ckpts/last.ckpt \
    --world_model.data_path exp_results/reproduce/antmaze-medium-play-v2/data/trajectories.zip \
    --planner.env.envname antmaze-medium-play-v2
```

**Parameters:**
- `--config`: Configuration file for the planner
- `--experiment_name`: Name for the experiment
- `--planner.agent.replay_buffer_size`: Size of the replay buffer
- `--experiment.steps`: Total training steps
- `--experiment.eval_interval`: Evaluation frequency
- `--experiment.seed`: Random seed for reproducibility
- `--planner.agent.lr`: Learning rate for the agent
- `--exp_id`: Experiment identifier
- `--world_model.ckpt`: Path to the pretrained world model checkpoint from step 2
- `--world_model.data_path`: Path to the trajectory data used for training
- `--planner.env.envname`: Environment name to use for planning

**Available Environments:**
- AntMaze: `experiments/antmaze/online_planner.yaml`

**File Flow Between Steps:**
- **Step 1** → **Step 2**: `trajectories.zip`
- **Step 2** → **Step 3**: `last.ckpt` + `trajectories.zip`

## Pinball Pixels Workflow

### 1. Data Generation

Generate pixel-based trajectory data for Pinball environment:

```bash
python scripts/generate_trajectories.py \
    --observation pixel \
    --image-size 50 \
    --uniform \
    --save-path exp_results/reproduce/pinball.zip \
    --num-traj 32 \
    --n-jobs 4
```

**Parameters:**
- `--observation`: Observation type (`pixel` for image-based observations)
- `--image-size`: Size of the pixel observations (50x50)
- `--uniform`: Use uniform sampling for data collection
- `--save-path`: Path to save the generated trajectory data
- `--num-traj`: Number of trajectories to generate
- `--n-jobs`: Number of parallel jobs for data collection

### 2. Model Pretraining

Train abstract world models for pixel-based Pinball:

```bash
python experiments/pretrain_mdp.py \
    --config experiments/pb_obstacles/pixel/config/tpc_cfg_critic.yaml \
    --experiment_cwd exp_results/reproduce/pinball \
    --data.data_path exp_results/reproduce/pinball.zip \
    --accelerator gpu \
    --epochs 1 \
    --model.latent_dim 16 \
    --tag test_reproduce
```

**Parameters:**
- `--config`: Configuration file for pixel-based Pinball model
- `--experiment_cwd`: Experiment working directory for saving results
- `--data.data_path`: Path to the trajectory data generated in step 1
- `--accelerator`: Training accelerator (`gpu` or `cpu`)
- `--epochs`: Number of training epochs
- `--model.latent_dim`: Dimensionality of the latent representation
- `--tag`: Experiment tag for organization

### 3. Planning & Training

Train agents using the pretrained abstract models for Pinball navigation:

```bash
python experiments/pb_obstacles/plan_and_train.py \
    --config experiments/pb_obstacles/pixel/config/online_planner.yaml \
    --experiment_name pinball \
    --planner.agent.replay_buffer_size 1000000 \
    --experiment.steps 100000 \
    --experiment.eval_interval 5000 \
    --planner.env.goal 9 \
    --experiment.seed 19 \
    --planner.agent.lr 1e-5 \
    --exp_id test \
    --experiments_cwd exp_results/reproduce/pinball/ \
    --world_model.ckpt exp_results/reproduce/pinball/mdps/test_reproduce/phi_train/ckpts/last.ckpt \
    --world_model.data_path exp_results/reproduce/pinball.zip
```

**Parameters:**
- `--config`: Configuration file for the Pinball planner
- `--experiment_name`: Name for the experiment
- `--planner.agent.replay_buffer_size`: Size of the replay buffer
- `--experiment.steps`: Total training steps
- `--experiment.eval_interval`: Evaluation frequency
- `--planner.env.goal`: Goal configuration ID (9 for specific goal location)
- `--experiment.seed`: Random seed for reproducibility
- `--planner.agent.lr`: Learning rate for the agent
- `--exp_id`: Experiment identifier
- `--experiments_cwd`: Experiment working directory
- `--world_model.ckpt`: Path to the pretrained world model checkpoint from step 2
- `--world_model.data_path`: Path to the trajectory data used for training

**Pinball File Flow Between Steps:**
- **Step 1** → **Step 2**: `pinball.zip`
- **Step 2** → **Step 3**: `last.ckpt` + `pinball.zip`

## Configuration

The project uses YAML configuration files to manage experiment parameters. Key configuration directories:

- `experiments/antmaze/config/`: AntMaze experiment configurations

## Environment Variables

Make sure to set appropriate environment variables for your setup:

```bash
export ENV=antmaze-medium-play-v2  # or other environment names
export size=32  # number of trajectories to generate
```

## Supported Environments

### AntMaze
- `antmaze-umaze-v2`
- `antmaze-medium-play-v2`

## Results

Experiment results are saved in the `exp_results/` directory with the following structure:
```
exp_results/
└── reproduce/
    └── antmaze-medium-play-v2/
        ├── data/
        │   ├── trajectories.zip              # Generated trajectory data (Step 1)
        │   └── trajectories_stats.yaml       # Dataset statistics
        └── mdps/
            └── test_reproduce/               # Tag from --tag parameter (Step 2)
                └── phi_train/
                    ├── ckpts/
                    │   ├── last.ckpt                    # Latest checkpoint (Step 3 input)
                    │   └── infomax-pb-epoch=XX-val_infomax=X.XX.ckpt
                    ├── logs/                            # TensorBoard logs
                    ├── csv_logs/                        # CSV training logs
                    └── test_results.yaml               # Final test metrics
```

## Citation

```bibtex
@article{rodriguez-sanchez2024learning,
    title={Learning Abstract World Models for Value-preserving Planning with Options},
    author={Rodriguez-Sanchez, Rafael and Konidaris, George},
    journal={Reinforcement Learning Journal},
    volume={4},
    pages={1733--1758},
    year={2024}
}
```

## Contact

For questions or issues, please contact:
- Rafael Rodriguez-Sanchez (rrs@brown.edu)