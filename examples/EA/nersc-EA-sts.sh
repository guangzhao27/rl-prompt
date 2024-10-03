#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m4259

cd /pscratch/sd/g/gzhao27/rl-prompt/examples/EA/
conda activate Prompt


python -u run_EA_sts.py \
    dataset=yelp \
    top_k=200 \
    direction=0_to_1 \
    prompt_length=2 \
    task_lm=distilgpt2 \
    report_to_wandb=false \
    dpo_training=true \
    eval_steps=5 \
    max_train_steps=60 \
    dpo_loss_config.reference_free=true \
    train_batch_size=2 \
    run_name=newrun \
    model_path=null \
    training_device=cpu \
    load_step=8000 \
    random_seed=42 \
    logit_bias=-10 \
    algorithm_name=ParetoPrompt-IPO \
    dataset_name=$dataset