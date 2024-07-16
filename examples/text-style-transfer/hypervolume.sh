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
dpo_loss_config.name=ipo dpo_loss_config.dpo_training=false dpo_loss_config.multi_optimize=false reward_type=hypervolume \
dpo_loss_config.nondominate_punishment=null dpo_loss_config.epsilon=1 \
training_device='cuda' max_train_steps=6000 project_name=tst-task run_name=tst-hypervolume \
eval_steps=1000 dominate_evaluate_num=64 \
target_update_method="polyak" target_learning_rate=0.001 \
model_path=null load_step=0 \
logit_bias=-10
# max_size=16
