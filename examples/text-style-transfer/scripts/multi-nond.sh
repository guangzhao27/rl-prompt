#!/bin/bash
#SBATCH -p csi
#SBATCH -t 24:00:00
#SBATCH --account csiml
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --qos csi
#SBATCH --gres=gpu:1

wandb login e48870a815eed7ccfabbb6d1a0e40f9a618bfa88
wandb offline
cd /hpcgpfs01/scratch/gzhao/rl-prompt/examples/text-style-transfer
srun python -u run_tst.py dataset=yelp report_to_wandb=true \
prompt_length=2 task_lm=gpt2-xl direction=0_to_1 \
dpo_loss_config.name=dpo dpo_loss_config.dpo_training=true dpo_loss_config.multi_optimize=true \
dpo_loss_config.nondominate_punishment=null dpo_loss_config.epsilon=1 \
training_device='cuda' max_train_steps=12000 project_name=tst-task run_name=tst-dpo-multi-noND-gpt2-xl-0.1lr \
eval_steps=1000 dominate_evaluate_num=64 \
target_update_method="polyak" target_learning_rate=0.1 \
model_path=null load_step=0 
# max_size=16
