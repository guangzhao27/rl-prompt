import os
import hydra
from typing import Optional
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.append('/pscratch/sd/g/gzhao27/rl-prompt')
from rlprompt.trainers import TrainerConfig, make_trainer
from rlprompt.modules import SQLModuleConfig, make_sql_module
from rlprompt.models import (LMAdaptorModelConfig, SinglePromptModelConfig,
                             make_lm_adaptor_model, make_single_prompt_model)
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)
sys.path.append('/pscratch/sd/g/gzhao27/rl-prompt/examples/text-style-transfer')
from tst_helpers import (PromptedTextStyleTransferRewardConfig,
                         TextStyleTransferDatasetConfig,
                         make_prompted_text_style_transfer_reward,
                         make_text_style_transfer_datasets,
                         get_style_classifier, 
                         )
from tst_reward import PromptedTextStyleTransferReward
from dataclasses import dataclass

# TODO: generation number and populaiton number
import random
import numpy
import torch

sys.path.append("/pscratch/sd/g/gzhao27/rl-prompt/InstOptima")

from entity.population import Population
from entity.instruction import Instruction
from operators.instruction_operators import InstructOperator
from entity.individual import Individual
from objectives.objective import o2_objective_generation
from evo_core.nsga2 import nsga2

from torch.utils.data import DataLoader, Dataset

@dataclass
class LoadConfig:
    model_path: Optional[str] = None
    algorithm_name: str="RlPrompt"
    load_step: int = 0
    few_shot: int = -1
    dominate_evaluate_num: int = 16
    temperature: float = 0.3
    dataset_name: str = 'yelp'
    population_size: int=10
    num_generations: int=30
    initial_file: str='./sts_initial_review.txt'

# Compose default config
config_list = [PromptedTextStyleTransferRewardConfig,
                TextStyleTransferDatasetConfig, LMAdaptorModelConfig,
                SinglePromptModelConfig, SQLModuleConfig, TrainerConfig, LoadConfig]
cs = compose_hydra_config_store('base_tst', config_list)


@hydra.main(version_base=None, config_path="../text-style-transfer/configs", config_name="tst_config")
def main(config: "DictConfig"):
    
    pre_path = '../text-style-transfer/'
    config.base_path = os.path.join(pre_path, config.base_path)
    config.max_size = 16
    config.prompt_train_batch_size = config.num_repeats*config.train_batch_size ## TODO: this should be fixed, too many free configs
    config.dataset_name = 'yelp'
    output_dir = get_hydra_output_dir()

    train_dataset, val_dataset, test_dataset = \
        make_text_style_transfer_datasets(config)
    print('Train Size:', len(train_dataset))
    print('Examples:', train_dataset[:5])
    print('Val Size', len(val_dataset))
    print('Examples:', val_dataset[:5])
    
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    config.style_classifier = get_style_classifier('train', config)
    config.style_classifier = os.path.join(pre_path, config.style_classifier)
    # reward = make_prompted_text_style_transfer_reward(config) # reward top_k decide the text generation, set as 1 for deterministic generation
    
    
    # task_lm = 'distilgpt2'
    # task_top_k = 10
    # style_classifier = './style_classifiers/yelp-bert-base-uncased-train/'
    # config.style_classifier = '../text-style-transfer/'+config.style_classifier
    # style_tokenizer = 'bert-base-uncased'
    # style_batch_size = 32
    with open("config.yaml", "w") as file:
        OmegaConf.save(config, file)
    # torch.save({'cfg': config}, 'config.pth')
    
    reward = PromptedTextStyleTransferReward(
        config.task_lm, config.task_top_k, config.style_classifier, 
        config.style_tokenizer,
        config.style_batch_size, config.pad_token, config.num_repeats, 
        config.num_samples, config.num_bootstraps, config.compute_zscore, 
        config.lower_outputs, config.control_output_length,
        config.template, config.end_punct, config.training_device, config.dpo_loss_config.multi_optimize)
    
    # prompt1 = "Transform the following negative review into a positive one while keeping the focus on the same restaurant details."
    
    # p = Instruction(definition=prompt1, example='', dataset=config.dataset_name)
    
    
    
    # def sts_objective_generation(batch, p):
    #     results_dict = reward(**batch, output_tokens=[[p.prompt]]*16, to_tensor=True, mode='infer', multi_optimize=True)
    #     style = results_dict[1].mean().item()
    #     content = results_dict[2].mean().item()
    #     return style, content
    
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    for batch in train_loader:
        pass
    
    objective_func = lambda p: o2_objective_generation(batch, p, reward)
    
    # style, content= objective_func(p)
    
    # inst_op = InstructOperator(dataset=config.dataset_name)

    # p2 = inst_op.evolve(p)
    
    # ind = Individual(p2)
    
    # ind.update_objectives(objective_func)
    
    # file_path = 'sts_initial_review.txt'
    
    # with open(file_path, 'r') as file:
    #     lines = file.readlines()
    
    # for line in lines:
    #     if line:
    #         p = Instruction(definition=line, example='', dataset=config.dataset_name)

    evo_new = nsga2(config.population_size, config.num_generations, output_dir=output_dir, initial_file=config.initial_file, dataset=config.dataset_name, objective_func=objective_func)

if __name__ == "__main__":
    main()