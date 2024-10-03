#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m4259_g

wandb login e48870a815eed7ccfabbb6d1a0e40f9a618bfa88
wandb online
# wandb offline
cd /pscratch/sd/g/gzhao27/rl-prompt/examples/few-shot-classification

algorithm_name=RlPrompt
dataset=sst-5 
echo $algorithm_name
echo $dataset
srun python -u run_fsc.py dataset=$dataset dataset_seed=1 \
report_to_wandb=true \
prompt_length=5 task_lm=roberta-large \
dpo_loss_config.epsilon=0.1 \
training_device='cuda' max_train_steps=6000 project_name=fsc-multi run_name=$dataset-pl5_$algorithm_name \
dominate_evaluate_num=64 \
target_update_method="copy" target_update_steps=200 \
algorithm_name=$algorithm_name


# algorithm_dict = {z
#     "RlPrompt": {"dpo_training":False, "name":"dpo", "multi_optimize":False, "nondominate_punishment":None}, 
#     "Reward-Guided-DPO": {"dpo_training":True, "name":"dpo", "multi_optimize":False, "nondominate_punishment":None}, 
#     "Reward-Guided-IPO": {"dpo_training":True, "name":"ipo", "multi_optimize":False, "nondominate_punishment":None}, 
#     "Dominance-Only-DPO": {"dpo_training":True, "name":"dpo", "multi_optimize":True, "nondominate_punishment":None}, 
#     "Dominance-Only-IPO": {"dpo_training":True, "name":"ipo", "multi_optimize":True, "nondominate_punishment":None}, 
#     "ParetoPrompt-DPO": {"dpo_training":True, "name":"dpo", "multi_optimize":True, "nondominate_punishment":"prob_diff"}, 
#     "ParetoPrompt-IPO": {"dpo_training":True, "name":"ipo", "multi_optimize":True, "nondominate_punishment":"prob_diff"}, 
# }
