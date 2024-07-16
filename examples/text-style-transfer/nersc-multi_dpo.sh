#!/bin/bash


model_name_list=(single_dpo single_ipo multi_nond_dpo multi_nond_ipo multi_probdiff_dpo multi_probdiff_ipo)
loss_name_list=(dpo ipo dpo ipo dpo ipo)
multi_bool_list=(false false true true true true)
nd_name_list=(null null null null prob_diff prob_diff)
for j in {0..4}; do
    
    sbatch rlprompt.sh 
    echo ${j}
    sleep 30

done