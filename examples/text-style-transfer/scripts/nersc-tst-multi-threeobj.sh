#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m4259_g

algorithm_name=$1
dataset=$2
direction=$3

wandb login e48870a815eed7ccfabbb6d1a0e40f9a618bfa88
cd /pscratch/sd/g/gzhao27/rl-prompt/examples/text-style-transfer
srun python -u run_tst.py \
dataset=yelp \
report_to_wandb=true \
prompt_length=5 task_lm=gpt2-xl direction=0_to_1 \
dpo_loss_config.epsilon=0.1 \
training_device='cuda' max_train_steps=4000 project_name=iclr-tst run_name=$dataset-pl5-$direction$algorithm_name \
algorithm_name=$algorithm_name \
eval_steps=1000 dominate_evaluate_num=64 \
target_update_method="copy" target_update_steps=200 \
model_path=null load_step=0 \
three_objectives=true
# max_size=16
