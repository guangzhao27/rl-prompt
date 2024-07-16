#!/bin/bash
#SBATCH -p csi
#SBATCH -t 10:00:00
#SBATCH --account csiml
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --qos csi
#SBATCH --gres=gpu:1

model_path=$1
run_name=$2
epoch_list=$3

wandb login e48870a815eed7ccfabbb6d1a0e40f9a618bfa88
wandb offline
cd /hpcgpfs01/scratch/gzhao/rl-prompt/examples/text-style-transfer/load_model
srun python -u run_tst_load_model.py dataset=yelp \
prompt_length=2 task_lm=gpt2-xl direction=0_to_1 \
dpo_loss_config.name=ipo dpo_loss_config.reference_learning_rate=0.001 \
report_to_wandb=false dpo_loss_config.dpo_training=true dpo_loss_config.multi_optimize=true \
training_device='cuda' max_train_steps=12000 project_name=tst-task \
prompt_train_batch_size=64 \
model_path=$model_path \
run_name=$run_name \
epoch_list="$epoch_list"

