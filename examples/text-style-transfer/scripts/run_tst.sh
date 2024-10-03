#!/bin/bash

algorithm_name_list=("RlPrompt" "Product" "Reward-Guided-IPO" "ParetoPrompt-DPO" "ParetoPrompt-IPO")
algorithm_name_list=("Dominance-Only-DPO" "Dominance-Only-IPO")
algorithm_name_list=("HVI")

dataset_name_list=('yelp')
direction=0_to_1

cd /pscratch/sd/g/gzhao27/rl-prompt/examples/text-style-transfer/scripts/
conda init
conda activate Prompt

for j in {0..0}; do
    for k in "${!dataset_name_list[@]}" 
    do
        for i in "${!algorithm_name_list[@]}"
        do
            sbatch nersc-tst-multi-threeobj.sh ${algorithm_name_list[i]} ${dataset_name_list[k]} $direction
            echo ${algorithm_name_list[i]} ${dataset_name_list[k]} 
            sleep 10
        done
    done
done