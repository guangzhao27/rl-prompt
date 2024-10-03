#!/bin/bash
# conda activate Prompt
dataset_name_list=("agnews" "cr" "dbpedia" "mr" "sst-5" "subj" "trec" "yahoo" "yelp-5")
algorithm_name_list=("RlPrompt" "Product" "Reward-Guided-IPO" "Reward-Guided-DPO" "ParetoPrompt-DPO" "ParetoPrompt-IPO")
dataset_name_list=("mr" "sst-5")
# dataset_name_list=('dbpedia' 'subj' 'trec' 'yahoo') "yelp-2"  
# epsilon_list=(0.1 0.3 0.5 0.7 0.9)
# epsilon_list=(0.7 )
run_name=gpt2ppt

cd /pscratch/sd/g/gzhao27/rl-prompt/examples/few-shot-classification/scripts/
conda init
conda activate Prompt

for j in {0..0}; do
    for k in "${!dataset_name_list[@]}" 
    do
        for i in "${!algorithm_name_list[@]}"
        do
            sbatch nersc-fsc-multi-gpt2ppt.sh ${algorithm_name_list[i]} ${dataset_name_list[k]} $run_name
            echo ${algorithm_name_list[i]} ${dataset_name_list[k]} 
            sleep 10
        done
    done
done

    # "RlPrompt": {"dpo_training":False, "name":"dpo", "multi_optimize":False, "nondominate_punishment":None}, 
    # "Reward-Guided-DPO": {"dpo_training":True, "name":"dpo", "multi_optimize":False, "nondominate_punishment":None}, 
    # "Reward-Guided-IPO": {"dpo_training":True, "name":"ipo", "multi_optimize":False, "nondominate_punishment":None}, 
    # "Dominance-Only-DPO": {"dpo_training":True, "name":"dpo", "multi_optimize":True, "nondominate_punishment":None}, 
    # "Dominance-Only-IPO": {"dpo_training":True, "name":"ipo", "multi_optimize":True, "nondominate_punishment":None}, 
    # "ParetoPrompt-DPO": {"dpo_training":True, "name":"dpo", "multi_optimize":True, "nondominate_punishment":"prob_diff"}, 
    # "ParetoPrompt-IPO": {"dpo_training":True, "name":"ipo", "multi_optimize":True, "nondominate_punishment":"prob_diff"}, 