#!/bin/bash


model_name_list=(outputs/2024-05-19/15-25-23 outputs/2024-05-19/16-13-58 outputs/2024-05-20/06-42-10 outputs/2024-05-20/05-59-11 outputs/2024-05-20/11-46-13)

for i in "${!model_name_list[@]}" 
do

    sbatch multi-nond.sh ${model_name_list[i]} 
    echo $i
    sleep 30
done