#!/bin/bash


model_name_list=(single_dpo single_ipo multi_nond_dpo multi_nond_ipo multi_probdiff_dpo multi_probdiff_ipo)
loss_name_list=(dpo ipo dpo ipo dpo ipo)
multi_bool_list=(false false true true true true)
nd_name_list=(null null null null prob_diff prob_diff)
for j in {0..3}; do
    for i in "${!model_name_list[@]}" 
    do
    
        sbatch nersc-tst.sh ${model_name_list[i]} ${loss_name_list[i]} ${multi_bool_list[i]} ${nd_name_list[i]}
        echo j ${model_name_list[i]} ${loss_name_list[i]} ${multi_bool_list[i]} ${nd_name_list[i]}
        sleep 30
    done
done