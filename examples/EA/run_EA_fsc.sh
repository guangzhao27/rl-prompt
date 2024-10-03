#!/bin/bash

conda activate Prompt

dataset_name_list=("agnews" "cr" "dbpedia" "mr" "sst-5" "subj" "trec" "yahoo" "yelp-5")
dataset_name_list=( "mr" "sst-5" "yahoo" "yelp-5")

cd /pscratch/sd/g/gzhao27/rl-prompt/examples/EA

for j in {0..2}; do
    for k in "${!dataset_name_list[@]}" 
    do
        sbatch nersc-EA-fsc.sh ${dataset_name_list[k]} gpt2
        echo $j ${dataset_name_list[k]} 
        sleep 10
    done
done