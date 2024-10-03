#!/bin/bash
echo hello world
# conda activate Prompt
# model_name_list=("Reward-Guided-DPO" "Reward-Guided-IPO" "Dominance-Only-DPO" "Dominance-Only-IPO" "ParetoPrompt-DPO" "ParetoPrompt-IPO")
# epsilon_list=(0.1 0.3 0.5 0.7 0.9)
epsilon_list=(0.7 )

cd /pscratch/sd/g/gzhao27/rl-prompt/examples/text-style-transfer/scripts/
conda init
conda activate Prompt

loss_name_list=(dpo ipo dpo ipo dpo ipo)
multi_bool_list=(false false true true true true)
nd_name_list=(null null null null prob_diff prob_diff)
for j in {0..0}; do
    for i in "${!epsilon_list[@]}" 
    do
    
        sbatch nersc-tst-epsilon.sh ${epsilon_list[i]} 
        echo ${j} ${epsilon_list[i]} 
        sleep 30
    done
done

    # "RlPrompt": {"dpo_training":False, "name":"dpo", "multi_optimize":False, "nondominate_punishment":None}, 
    # "Reward-Guided-DPO": {"dpo_training":True, "name":"dpo", "multi_optimize":False, "nondominate_punishment":None}, 
    # "Reward-Guided-IPO": {"dpo_training":True, "name":"ipo", "multi_optimize":False, "nondominate_punishment":None}, 
    # "Dominance-Only-DPO": {"dpo_training":True, "name":"dpo", "multi_optimize":True, "nondominate_punishment":None}, 
    # "Dominance-Only-IPO": {"dpo_training":True, "name":"ipo", "multi_optimize":True, "nondominate_punishment":None}, 
    # "ParetoPrompt-DPO": {"dpo_training":True, "name":"dpo", "multi_optimize":True, "nondominate_punishment":"prob_diff"}, 
    # "ParetoPrompt-IPO": {"dpo_training":True, "name":"ipo", "multi_optimize":True, "nondominate_punishment":"prob_diff"}, 