#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=cpu
#SBATCH --account=m2956

cd /pscratch/sd/g/gzhao27/rl-prompt/examples/few-shot-classification/
conda activate Prompt

dataset=$1
fluency_model_name=$2

python /pscratch/sd/g/gzhao27/rl-prompt/examples/EA/run_EA_fsc.py \
    dataset=$dataset \
    dataset_seed=1 \
    prompt_length=5 \
    task_lm=roberta-large \
    dpo_loss_config.name=ipo \
    dpo_loss_config.reference_learning_rate=0.001 \
    report_to_wandb=true \
    dpo_training=false \
    training_device=cpu \
    max_train_steps=6000 \
    run_name=roberta-large-rl \
    report_to_wandb=false \
    algorithm_name=ParetoPrompt-IPO \
    eval_batch_size=2 \
    dominate_evaluate_num=2 \
    dataset_name=$fluency_model_name$dataset
    fluency_model_name=$fluency_model_name