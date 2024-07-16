#!/bin/bash

model_path_list=(outputs/2024-04-26/15-41-51 outputs/2024-04-26/15-43-03 outputs/2024-04-26/15-43-05)
run_name_list=(
outputs/2024-04-26/15-41-51tst-ipo-multi-prob_diff-gpt2-xl-e0.1-tus100 \
outputs/2024-04-26/15-43-03tst-ipo-multi-prob_diff-gpt2-xl-e0.1-tus200 \
outputs/2024-04-26/15-43-05tst-ipo-multi-prob_diff-gpt2-xl-e0.1-tus500 \
)
epoch_list=([200, 500, 1000, 4000, 6700])

for i in "${!model_path_list[@]}" 
do
    sbatch tst-model-load.sh ${model_path_list[i]} ${run_name_list[i]} "${epoch_list[0]}"
done

# 'outputs/2024-03-27/15-29-39tst-rl-gpt2-xl-64', 
# 'outputs/2024-03-27/15-30-10tst-ipo-gpt2-xl-64', 
# 'outputs/2024-03-18/12-54-12tst-ipo-multi-gpt2-xl-64',
# 'outputs/2024-03-30/12-41-56tst-ipo-multi-prob_diff-gpt2-xl', 
# 'outputs/2024-03-30/12-42-07tst-ipo-multi-gpt2-xl-0.1lr', 
# "outputs/2024-04-04/04-07-46tst-ipo-multi-prob_diff-gpt2-xl-0.1", 
# "outputs/2024-04-04/05-24-39tst-ipo-single-gpt2-xl-0.1lr"
# “outputs/2024-04-16/13-23-28tst-ipo-multi-prob_diff-gpt2-xl-0.1”
# "outputs/2024-04-18/20-11-52tst-ipo-multi-prob_diff-gpt2-xl-e0.1"
# "outputs/2024-04-18/21-18-38tst-ipo-multi-prob_diff-gpt2-xl-e0.5"
# outputs/2024-04-26/15-41-51tst-ipo-multi-prob_diff-gpt2-xl-e0.1-tus100 \
# outputs/2024-04-26/15-43-03tst-ipo-multi-prob_diff-gpt2-xl-e0.1-tus200 \
# outputs/2024-04-26/15-43-05tst-ipo-multi-prob_diff-gpt2-xl-e0.1-tus500 \

# model_path_list=(outputs/2024-04-16/13-23-28 outputs/2024-04-16/13-23-28)
# outputs/2024-04-26/15-43-03tst-ipo-multi-prob_diff-gpt2-xl-e0.1-tus200
