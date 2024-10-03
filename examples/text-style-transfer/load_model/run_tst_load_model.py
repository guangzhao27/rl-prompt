import os
from dataclasses import dataclass, field
import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader
import sys
from typing import List
sys.path.append('../../../')
sys.path.append('../')
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
                         get_style_classifier)
import json
from dataclasses import dataclass

@dataclass
class LoadConfig:
    model_path: str = "../outputs/2024-03-18/12-54-12/outputs/ckpt"
    epoch_list: List[int] = field(default_factory=lambda: [1000, 4000, 7000])
# Compose default config
config_list = [PromptedTextStyleTransferRewardConfig,
                TextStyleTransferDatasetConfig, LMAdaptorModelConfig,
                SinglePromptModelConfig, SQLModuleConfig, TrainerConfig, LoadConfig]
cs = compose_hydra_config_store('base_tst', config_list)


@hydra.main(version_base=None, 
            config_path="../configs", 
            config_name="load_config")
def main(config: "DictConfig"):
    # config.prompt_train_batch_size = config.num_repeats*config.train_batch_size ## TODO: this should be fixed, too many free configs
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    output_dir = get_hydra_output_dir()

    train_dataset, val_dataset, test_dataset = \
        make_text_style_transfer_datasets(config)

    print('Train Size:', len(train_dataset))
    print('Examples:', train_dataset[:5])
    print('Val Size', len(val_dataset))
    print('Examples:', val_dataset[:5])
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))

    policy_model = make_lm_adaptor_model(config)
    prompt_model = make_single_prompt_model(policy_model, config)
    config.style_classifier = get_style_classifier('train', config)
    # config.style_classifier = "."+ config.style_classifier 
    config.style_classifier = "../style_classifiers/yelp-bert-base-uncased-train/"
    reward = make_prompted_text_style_transfer_reward(config)
    algo_module = make_sql_module(prompt_model, reward, config)
    
    performance_list = {}
    epoch_list = config.epoch_list

    
    
    model_path = "../" + \
        config.model_path + "/outputs/ckpt"
    for epoch in epoch_list:
        print(epoch)
        performance_list[f'{epoch}'] = {}
        algo_module = make_sql_module(prompt_model, reward, config)
        ckpt_path = os.path.join(model_path, f"ckpt.step.{epoch}.pth")
        if config.training_device=="cpu":
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        elif config.training_device=='cuda':
            checkpoint = torch.load(ckpt_path)
        algo_module.load_state_dict(checkpoint['model_state_dict'])
        # algo_module._top_k = 200
        for batch in val_loader:
            (logits, logits_, output_tokens, output_ids, sequence_lengths) = algo_module._decode_sampling(batch=batch)
            # generate {prompt_train_batch_size} prompts 
        content_list = []
        style_list = []
        cola_list = []
        for prompt in output_tokens:    
            output_token_list = [prompt]*len(batch['source_texts'])
            sum_reward, multi_rewards_dict, tokens_list, rewards_log= algo_module.compute_rewards(batch=batch, output_tokens=output_token_list, multi_optimize=True)
            style_reward = multi_rewards_dict['style']
            content_reward = multi_rewards_dict['content']
            if len(multi_rewards_dict) == 3:
                cola = multi_rewards_dict['cola']
                cola_list.append(cola.mean().item())
            # for each prompt, calculate the average reward, the reward average over 16 validation sentence, 
            # each sentence and prompt generating {num_samples}*{num_bootstraps} sentences 
            # the reward average over the 16*{num_samples}*{num_bootstraps} rewards
            
            content_list.append(content_reward.mean().item())
            style_list.append(style_reward.mean().item())
        performance_list[f'{epoch}']['content'] = content_list
        performance_list[f'{epoch}']['style'] = style_list
        performance_list[f'{epoch}']['cola'] = cola_list
    
    file_name ="../"+config.run_name+".json"
    
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_name, "w") as file:
        json.dump(performance_list, file)
    print('finish loading')
    return 


if __name__ == "__main__":
    main()
    
"""
/hpcgpfs01/scratch/gzhao/rl-prompt/examples/text-style-transfer/tst_modules/output_selector.py  compute_sample_rewards
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

/hpcgpfs01/scratch/gzhao/rl-prompt/rlprompt/modules/sql_module.py 
raw reward and shaped reward the mean is 0, but they used to calculate q_loss



During training
In one epoch
2 inputs of negtive comments
each negtive comments repeat 4 times, and repectively combined with 8 prompts, 
each prompts generate num_samples(4)*num_bootraps(4) sentences to get the transformed sentences.     ## why don't use different negtive comments for the same prompt?


The policy-model with adaptor first generate 8 prompts
Then each prompt combined with input as the formatted_template:
    formatted_template: 'DeliverySquare "i was sadly mistaken." "'

The generated text is just keep generating new text:
    'DeliverySquare "i was sadly mistaken." "I think that\'s just not true," he said, pointing out that he'


In Eval
score is just sum_reward


The generate prompts are irrelvent to the input sentence


/hpcgpfs01/scratch/gzhao/rl-prompt/examples/text-style-transfer/tst_helpers.py
what are style_batch_size and sample_size, they should be the same?


How to evaluate the RL performance for multi-objective learning

How to avoid the mode clapse problem??
"""
