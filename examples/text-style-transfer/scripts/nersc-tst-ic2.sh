#!/bin/bash
#SBATCH -p csi
#SBATCH -t 24:00:00
#SBATCH --account csiml
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --qos csi
#SBATCH --gres=gpu:1

run_name=$1
loss_name=$2
multi_bool=$3
nd_name=$4


wandb login e48870a815eed7ccfabbb6d1a0e40f9a618bfa88
wandb offline
cd /hpcgpfs01/scratch/gzhao/rl-prompt/examples/text-style-transfer
srun python -u run_tst.py dataset=yelp report_to_wandb=true \
prompt_length=2 task_lm=gpt2-xl direction=0_to_1 \
dpo_loss_config.name=$loss_name dpo_loss_config.dpo_training=true dpo_loss_config.multi_optimize=$multi_bool \
dpo_loss_config.nondominate_punishment=$nd_name dpo_loss_config.epsilon=0.1 \
training_device='cuda' max_train_steps=4000 project_name=tst-task run_name=tst-ipo-multi-prob_diff-gpt2-xl-e0.1-tus$run_name \
eval_steps=1000 dominate_evaluate_num=64 \
target_update_method="copy" target_update_steps=200 \
model_path=null load_step=0 
# max_size=16
