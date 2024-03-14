#!/bin/bash
#SBATCH -p csi
#SBATCH -t 15:00:00
#SBATCH --account csiml
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --qos csi
#SBATCH --gres=gpu:1

wandb login e48870a815eed7ccfabbb6d1a0e40f9a618bfa88
wandb offline

cd /hpcgpfs01/scratch/gzhao/rl-prompt/examples/few-shot-classification

srun python -u run_fsc.py dataset=sst-2 dataset_seed=0 \
prompt_length=2 task_lm=roberta-base  \
report_to_wandb=true dpo_training=false training_device='cpu' max_train_steps=15