#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m4259_g
#2956_g
conda activate Prompt
wandb login e48870a815eed7ccfabbb6d1a0e40f9a618bfa88
cd /pscratch/sd/g/gzhao27/rl-prompt/examples/text-style-transfer
srun python -u run_tst.py dataset=yelp direction=0_to_1 \
prompt_length=2 task_lm=gpt2-xl \
report_to_wandb=true training_device='cuda' max_train_steps=12000 \
model_path=null load_step=0 \
target_update_method="copy" target_update_steps=200 \
eval_steps=100 dominate_evaluate_num=64 \
algorithm_name='Dominance-Only-IPO' dpo_loss_config.epsilon=0.1 \
project_name=tst-task run_name=yelp-dominanceonly-ipo-200update \
top_k=50 dpo_loss_config.reference_free=true
# max_size=16

# dataset=shakespeare dataset_seed=0

    # "RlPrompt": {"dpo_training":False, "name":"dpo", "multi_optimize":False, "nondominate_punishment":None}, 
    # "Reward-Guided-DPO": {"dpo_training":True, "name":"dpo", "multi_optimize":False, "nondominate_punishment":None}, 
    # "Reward-Guided-IPO": {"dpo_training":True, "name":"ipo", "multi_optimize":False, "nondominate_punishment":None}, 
    # "Dominance-Only-DPO": {"dpo_training":True, "name":"dpo", "multi_optimize":True, "nondominate_punishment":None}, 
    # "Dominance-Only-IPO": {"dpo_training":True, "name":"ipo", "multi_optimize":True, "nondominate_punishment":None}, 
    # "ParetoPrompt-DPO": {"dpo_training":True, "name":"dpo", "multi_optimize":True, "nondominate_punishment":"prob_diff"}, 
    # "ParetoPrompt-IPO": {"dpo_training":True, "name":"ipo", "multi_optimize":True, "nondominate_punishment":"prob_diff"}, 