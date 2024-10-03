#!/bin/bash
#SBATCH -p csi
#SBATCH -t 2:00:00
#SBATCH --account csiml
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --qos csi
#SBATCH --gres=gpu:1

wandb login e48870a815eed7ccfabbb6d1a0e40f9a618bfa88
wandb offline
cd /hpcgpfs01/scratch/gzhao/rl-prompt/examples/few-shot-classification
srun python -u run_fsc.py dataset=sst-2 dataset_seed=1 \
prompt_length=5 task_lm=roberta-large  \
dpo_loss_config.name=ipo dpo_loss_config.reference_learning_rate=0.001 \
report_to_wandb=true dpo_training=false training_device='cuda' max_train_steps=6000 run_name=roberta-large-rl\