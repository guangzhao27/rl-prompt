import os
import dataclasses
import hydra
from hydra.core.config_store import ConfigStore
from typing import Optional
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.append('../../')
from rlprompt.trainers import TrainerConfig, make_trainer
from rlprompt.modules import SQLModuleConfig, make_sql_module
from rlprompt.models import (LMAdaptorModelConfig, SinglePromptModelConfig,
                             make_lm_adaptor_model, make_single_prompt_model)
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)
from tst_helpers import (PromptedTextStyleTransferRewardConfig,
                         TextStyleTransferDatasetConfig,
                         make_prompted_text_style_transfer_reward,
                         make_text_style_transfer_datasets,
                         get_style_classifier, 
                         algorithm_set_config, 
                         )
from dataclasses import dataclass

import random
import numpy as np
import torch

# import os
# os.environ['HF_HOME'] = './llm_cache_dir'


@dataclass
class LoadConfig:
    model_path: Optional[str] = None
    algorithm_name: str="RlPrompt"
    load_step: int = 0
    few_shot: int = -1
    dominate_evaluate_num: int = 16

# Compose default config
config_list = [PromptedTextStyleTransferRewardConfig,
                TextStyleTransferDatasetConfig, LMAdaptorModelConfig,
                SinglePromptModelConfig, SQLModuleConfig, TrainerConfig, LoadConfig]
cs = compose_hydra_config_store('base_tst', config_list)


@hydra.main(version_base=None, config_path="./configs", config_name="tst_config")
def main(config: "DictConfig"):
    
    config.prompt_train_batch_size = config.num_repeats*config.train_batch_size ## TODO: this should be fixed, too many free configs
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    output_dir = get_hydra_output_dir()

    train_dataset, val_dataset, test_dataset = \
        make_text_style_transfer_datasets(config)
    print('Train Size:', len(train_dataset))
    print('Examples:', train_dataset[:5])
    print('Val Size', len(val_dataset))
    print('Examples:', val_dataset[:5])

    print("Algorithm Name: ", config.algorithm_name)
    algorithm_set_config(config)


    policy_model = make_lm_adaptor_model(config)
    prompt_model = make_single_prompt_model(policy_model, config)
    config.style_classifier = get_style_classifier('train', config)
    reward = make_prompted_text_style_transfer_reward(config) # reward top_k decide the text generation, set as 1 for deterministic generation
    algo_module = make_sql_module(prompt_model, reward, config) #module top_k decide the prompt generation, set as 0 for diverse generation

    config.save_dir = os.path.join(output_dir, config.save_dir)
    if config.run_name:
        config.run_name = output_dir+config.run_name
    else:
        config.run_name = output_dir
    trainer = make_trainer(algo_module, train_dataset, val_dataset, config)
    
    if config.model_path:
        load_ckpt_path = os.path.join('./',
            config.model_path, 
            f"outputs/ckpt/ckpt.step.{config.load_step}.pth")
        trainer.load_pretrain(load_ckpt_path, config.training_device)
        reward._counter=config.load_step
    
    trainer.train(config=config)


if __name__ == "__main__":
    main()
    
"""
/scratch/gzhao/rl-prompt/examples/text-style-transfer/tst_modules/output_selector.py  compute_sample_rewards
content_rewards uses 'roberta-large' BERTScorer to calculate two sentence similarity  [0, 100]
style_rewards uses './style_classifiers/yelp-bert-base-uncased-train/' to classify styles [0, 100]


idx| prompt list        |prompt str |original sentence      | new sentence         | content reward     | style reward     | sum reward max    | mean of 4 max of bootstramp
1 | ['Info', 'Effect'] | InfoEffect | i was sadly mistaken. | it was a good idea. | Top Content: 42.06 | Top Style: 99.86 | Top Reward: 70.96 | Reward: 64.41

implementation: 'v2_v2r_v3_v3r'

raw_reward:  mean of 4 max of bootstramp
shaped_reward: raw_reward * (new_max - new_min) + new_min


wandb explain:

SQL_ON/rewards:
sum_reward: mean of max of bootstrap of sum_reward
mean_reward: mean of all sum_reward
top_content: max of content score
top_style: max of style prob

/scratch/gzhao/rl-prompt/rlprompt/modules/sql_module.py 
raw reward and shaped reward the mean is 0, but they used to calculate q_loss



During training
In one epoch
2 inputs of negtive comments
each negtive comments repeat 4 times, so there are 8 prompts in total, 
each prompts generate num_samples(4) *num_bootraps (4) sentences to get the transformed sentences.     ## why don't use different negtive comments for the same prompt?


The policy-model with adaptor first generate 8 prompts
Then each prompt combined with input as the formatted_template:
    formatted_template: 'DeliverySquare "i was sadly mistaken." "'

The generated text is just keep generating new text:
    'DeliverySquare "i was sadly mistaken." "I think that\'s just not true," he said, pointing out that he'


In Eval
score is just sum_reward


The generate prompts are irrelvent to the input sentence


## Change reward from mean of max bootstramp to mean_reward in 
#   /scratch/gzhao/rl-prompt/examples/text-style-transfer/tst_reward.py line 112

/scratch/gzhao/rl-prompt/examples/text-style-transfer/tst_helpers.py
what are style_batch_size and sample_size, they should be the same?

what is the mask for generating sentence, what is the prompt position?

How to evaluate the RL performance for multi-objective learning

How to avoid the mode clapse problem??





1 | ['Shadow', 'Stats'] | ShadowStats | i was sadly mistaken. | he is a very intelligent person. | Top Content: 39.06 | Top Style: 99.98 | Top Reward: 69.52 | Reward: 23.97
1 | ['Description', 'Spider'] | DescriptionSpider | i was sadly mistaken. | i was very surprised. | Top Content: 54.31 | Top Style: 99.02 | Top Reward: 76.67 | Reward: 27.21
1 | ['Pope', 'Rated'] | PopeRated | i was sadly mistaken. | it was a good sign. | Top Content: 37.54 | Top Style: 99.85 | Top Reward: 68.69 | Reward: 24.34
1 | ['Shadow', 'Report'] | ShadowReport | i was sadly mistaken. | i'm sure he's right. | Top Content: 35.3 | Top Style: 99.67 | Top Reward: 67.48 | Reward: 18.12
1 | ['ĠA', 'Player'] |  APlayer | so on to the hoagies, the italian is general run of the mill. | he was a master of the mill. | Top Content: 19.69 | Top Style: 99.34 | Top Reward: 59.52 | Reward: 24.47
1 | ['Israel', 'ĠImage'] | Israel Image | so on to the hoagies, the italian is general run of the mill. | we have all the best of the best at the mill. | Top Content: 17.3 | Top Style: 99.99 | Top Reward: 58.64 | Reward: 20.27
1 | ['ĠImage', 'Microsoft'] |  ImageMicrosoft | so on to the hoagies, the italian is general run of the mill. | and the italian is the most important part of the economy. | Top Content: 26.33 | Top Style: 99.57 | Top Reward: 62.95 | Reward: 26.19
1 | ['Effect', 'Senate'] | EffectSenate | so on to the hoagies, the italian is general run of the mill. | the most popular is a lot of the old man. | Top Content: 17.68 | Top Style: 99.77 | Top Reward: 58.73 | Reward: 15.8
tensor(0.1733, grad_fn=<MeanBackward0>)



"""
